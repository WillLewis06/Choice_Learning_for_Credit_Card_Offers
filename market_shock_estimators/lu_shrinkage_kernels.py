from __future__ import annotations
import tensorflow as tf

"""
MCMC kernels for the Lu–Shimizu (2025) shrinkage estimator.

This module contains *market-level* update kernels that are used by the
Lu shrinkage estimator but are factored out for clarity and testability.

Included kernels
----------------
1. mh_toggle_gamma_market
   Metropolis–Hastings kernel that jointly updates:
     - gamma_{jt} ∈ {0,1}
     - n_{jt} (latent market–product shocks)
   using the Lu spike-and-slab prior and the market-level log posterior.

2. gibbs_phi_market
   Gibbs update for the market-level inclusion probability phi_t:
     phi_t | gamma_t ~ Beta(a_phi + sum_j gamma_{jt},
                            b_phi + J − sum_j gamma_{jt})

3. sample_beta_tf
   Stateless Beta sampler implemented via two Gamma draws, compatible
   with tf.random.Generator and reproducible seeding.

Design notes
------------
- These kernels operate on a *single market t* at a time.
- They mutate estimator state (tf.Variables) and therefore expect to be
  called from an owning estimator object that provides:
    - posterior object (with market_logpost and hyperparameters)
    - tf.random.Generator (rng)
    - model dimensions and data tensors
- No orchestration, iteration control, or diagnostics are included here.
- The code follows the blocking scheme described in Lu & Shimizu (2025),
  Section 3–4, and is intended to be unit-testable at the kernel level.

This file contains no simulation logic and no estimator API.
"""


def mh_toggle_gamma_market(
    shrink: LuShrinkageEstimator,
    t: int,
) -> tuple[tf.Tensor, tf.Tensor]:

    gamma_t = tf.cast(shrink.gamma[t], tf.int32)
    njt_t = tf.cast(shrink.njt[t], tf.float64)
    phi_t = tf.cast(shrink.phi[t], tf.float64)

    for j in range(shrink.J):
        g_old = int(gamma_t[j].numpy())
        gamma_old = gamma_t
        njt_old = njt_t * tf.cast(gamma_old, tf.float64)

        if g_old == 0:
            var = tf.cast(shrink.posterior.sigma_n_sq, tf.float64)
            sd = tf.sqrt(var)
            njt_j_prop = sd * shrink.rng.normal([], dtype=tf.float64)
            gamma_new = tf.tensor_scatter_nd_update(gamma_old, [[j]], [1])
            njt_new = tf.tensor_scatter_nd_update(njt_old, [[j]], [njt_j_prop])
        else:
            gamma_new = tf.tensor_scatter_nd_update(gamma_old, [[j]], [0])
            njt_new = tf.tensor_scatter_nd_update(
                njt_old, [[j]], [tf.cast(0.0, tf.float64)]
            )

        njt_new = njt_new * tf.cast(gamma_new, tf.float64)

        lp_old = shrink.posterior.market_logpost(
            qjt_t=shrink.qjt[t],
            q0t_t=shrink.q0t[t],
            pjt_t=shrink.pjt[t],
            wjt_t=shrink.wjt[t],
            beta_p=shrink.beta_p,
            beta_w=shrink.beta_w,
            r=shrink.r,
            E_bar_t=shrink.E_bar[t],
            njt_t=njt_old,
            gamma_t=gamma_old,
            phi_t=phi_t,
        )
        lp_new = shrink.posterior.market_logpost(
            qjt_t=shrink.qjt[t],
            q0t_t=shrink.q0t[t],
            pjt_t=shrink.pjt[t],
            wjt_t=shrink.wjt[t],
            beta_p=shrink.beta_p,
            beta_w=shrink.beta_w,
            r=shrink.r,
            E_bar_t=shrink.E_bar[t],
            njt_t=njt_new,
            gamma_t=gamma_new,
            phi_t=phi_t,
        )

        var = tf.cast(shrink.posterior.sigma_n_sq, tf.float64)
        if g_old == 0:
            log_q_forward = (
                -0.5 * tf.math.log(shrink.posterior.two_pi * var)
                - 0.5 * tf.square(njt_j_prop) / var
            )
            log_q_reverse = tf.cast(0.0, tf.float64)
        else:
            njt_j_old = tf.cast(njt_old[j], tf.float64)
            log_q_forward = tf.cast(0.0, tf.float64)
            log_q_reverse = (
                -0.5 * tf.math.log(shrink.posterior.two_pi * var)
                - 0.5 * tf.square(njt_j_old) / var
            )

        log_alpha = (lp_new - lp_old) + (log_q_reverse - log_q_forward)
        u = shrink.rng.uniform([], dtype=tf.float64)
        if bool((tf.math.log(u) < log_alpha).numpy()):
            gamma_t = tf.cast(gamma_new, tf.int32)
            njt_t = njt_new

    return gamma_t, njt_t


def gibbs_phi_market(
    gamma_t: tf.Tensor,
    a_phi,
    b_phi,
    J: int,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Gibbs update:
      phi_t | gamma_t ~ Beta(a_phi + sum(gamma_t),
                             b_phi + J - sum(gamma_t))
    """
    gamma_t = tf.cast(gamma_t, tf.float64)

    a_post = tf.cast(a_phi, tf.float64) + tf.reduce_sum(gamma_t)
    b_post = (
        tf.cast(b_phi, tf.float64) + tf.cast(J, tf.float64) - tf.reduce_sum(gamma_t)
    )

    phi_t_new = _sample_beta_tf(rng, a_post, b_post)
    return phi_t_new


def _sample_beta_tf(rng: tf.random.Generator, a: tf.Tensor, b: tf.Tensor):
    a = tf.convert_to_tensor(a, dtype=tf.float64)
    b = tf.convert_to_tensor(b, dtype=tf.float64)

    seeds = rng.make_seeds(2)
    seed_x = seeds[0]
    seed_y = seeds[1]

    x = tf.random.stateless_gamma(
        shape=[],
        seed=seed_x,
        alpha=a,
        beta=tf.cast(1.0, tf.float64),
        dtype=tf.float64,
    )
    y = tf.random.stateless_gamma(
        shape=[],
        seed=seed_y,
        alpha=b,
        beta=tf.cast(1.0, tf.float64),
        dtype=tf.float64,
    )
    return x / (x + y)
