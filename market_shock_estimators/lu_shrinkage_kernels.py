from __future__ import annotations

import tensorflow as tf

"""
MCMC kernels for the Lu–Shimizu (2025) shrinkage estimator.

This module contains *market-level* update kernels aligned with the
two-normal spike-and-slab prior in the Lu paper.

Included kernels
----------------
1. sample_gamma_given_n_phi_market
   Gibbs update for gamma_t given n_t and phi_t:
     gamma_jt | n_jt, phi_t ~ Bernoulli(p_jt)
   where p_jt follows from the two-normal mixture:
     n_jt | gamma_jt=1 ~ N(0, T1_sq)
     n_jt | gamma_jt=0 ~ N(0, T0_sq)

2. gibbs_phi_market
   Gibbs update for the market-level inclusion probability phi_t:
     phi_t | gamma_t ~ Beta(a_phi + sum_j gamma_jt,
                            b_phi + J − sum_j gamma_jt)

3. _sample_beta_tf
   Stateless Beta sampler implemented via two Gamma draws, compatible
   with tf.random.Generator.

Design notes
------------
- Kernels operate on a *single market t* at a time.
- No masking of n_jt by gamma_jt is performed.
- No Metropolis–Hastings birth/death logic is used.
- These kernels are mathematically aligned with Lu & Shimizu (2025).
"""


def sample_gamma_given_n_phi_market(
    *,
    njt_t: tf.Tensor,
    phi_t: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    log_T0_sq: tf.Tensor,
    log_T1_sq: tf.Tensor,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Gibbs update for gamma_t given n_t and phi_t under the Lu spike-and-slab:

      p(gamma_j=1 | n_j, phi)
        ∝ phi * N(n_j; 0, T1_sq)
      p(gamma_j=0 | n_j, phi)
        ∝ (1-phi) * N(n_j; 0, T0_sq)

    Parameters
    ----------
    njt_t : (J,) tf.Tensor
        Latent market–product shocks for market t.
    phi_t : scalar tf.Tensor
        Inclusion probability for market t.
    T0_sq, T1_sq : scalar tf.Tensor
        Spike and slab variances.
    log_T0_sq, log_T1_sq : scalar tf.Tensor
        Precomputed logs of the variances.
    rng : tf.random.Generator

    Returns
    -------
    gamma_t : (J,) tf.Tensor, dtype int32
    """
    njt_t = tf.cast(njt_t, tf.float64)
    phi_t = tf.cast(phi_t, tf.float64)

    # Numerical safety
    eps = tf.constant(1e-15, dtype=tf.float64)
    phi_t = tf.clip_by_value(phi_t, eps, 1.0 - eps)

    two_pi = tf.constant(2.0 * 3.141592653589793, dtype=tf.float64)

    # log N(n_j; 0, T1_sq)
    logp1 = (
        tf.math.log(phi_t)
        - 0.5 * (tf.math.log(two_pi) + log_T1_sq)
        - 0.5 * tf.square(njt_t) / T1_sq
    )

    # log N(n_j; 0, T0_sq)
    logp0 = (
        tf.math.log(1.0 - phi_t)
        - 0.5 * (tf.math.log(two_pi) + log_T0_sq)
        - 0.5 * tf.square(njt_t) / T0_sq
    )

    # p = sigmoid(logp1 - logp0)
    logit_p = logp1 - logp0
    p = tf.math.sigmoid(logit_p)

    u = rng.uniform(shape=tf.shape(p), dtype=tf.float64)
    gamma_t = tf.cast(u < p, tf.int32)

    return gamma_t


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


def _sample_beta_tf(
    rng: tf.random.Generator,
    a: tf.Tensor,
    b: tf.Tensor,
) -> tf.Tensor:
    """
    Stateless Beta(a,b) sampler using two Gamma draws.
    """
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
