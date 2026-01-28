from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def tmh_step(
    theta0,
    logp_fn,
    ridge,
    max_lbfgs_iters,
    rng,
    k,
):
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    # ------------------------------------------------------------
    # 1. Dimension and kappa (Lu default)
    # ------------------------------------------------------------
    k = tf.cast(tf.convert_to_tensor(k), dtype)

    # ------------------------------------------------------------
    # 2. Mode finding via LBFGS
    # ------------------------------------------------------------
    def val_and_grad(x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            val = -logp_fn(x)
        grad = tape.gradient(val, x)
        return val, grad

    res = tfp.optimizer.lbfgs_minimize(
        val_and_grad,
        initial_position=theta0,
        max_iterations=max_lbfgs_iters,
    )

    mu = tf.where(res.converged, res.position, theta0)

    # ------------------------------------------------------------
    # 3. Hessian of logp at the mode
    # ------------------------------------------------------------
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(mu)
        with tf.GradientTape() as t1:
            t1.watch(mu)
            lp = logp_fn(mu)
        grad = t1.gradient(lp, mu)
    H = t2.jacobian(grad, mu)
    del t2

    # ------------------------------------------------------------
    # 4. Precision matrix and Cholesky
    #     P = (-H + ridge I) / kappa^2
    # ------------------------------------------------------------
    ridge = tf.cast(ridge, dtype)
    I = tf.eye(tf.size(mu), dtype=dtype)
    P = (-H + ridge * I) / (k * k)

    L = tf.linalg.cholesky(P)

    # ------------------------------------------------------------
    # 5. Laplace proposal draw
    #     theta' = mu + L^{-T} eps
    # ------------------------------------------------------------
    eps = rng.normal(tf.shape(mu), dtype=dtype)
    delta = tf.linalg.triangular_solve(L, eps[:, None], adjoint=True)[:, 0]
    prop = mu + delta

    # ------------------------------------------------------------
    # 6. Proposal log density under precision parameterization
    # ------------------------------------------------------------
    def log_q(theta):
        diff = theta - mu
        z = tf.matmul(L, diff[:, None], transpose_a=True)
        quad = tf.reduce_sum(z * z)
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return 0.5 * logdet - 0.5 * quad

    # ------------------------------------------------------------
    # 7. Independence MH accept/reject
    # ------------------------------------------------------------
    log_alpha = logp_fn(prop) + log_q(theta0) - logp_fn(theta0) - log_q(prop)

    u = rng.uniform([], dtype=dtype)
    accepted = tf.math.log(u) < log_alpha

    theta_new = tf.where(accepted, prop, theta0)

    return theta_new, accepted


def rw_mh_step(theta0, logp_fn, k, rng: tf.random.Generator):
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    k = tf.cast(tf.convert_to_tensor(k), dtype)

    # ------------------------------------------------------------
    # 1) Propose
    # ------------------------------------------------------------
    # Identity covariance: step = k * z, z ~ N(0, I) with shape(theta0)
    z = rng.normal(tf.shape(theta0), dtype=dtype)
    theta_prop = theta0 + k * z
    logp_curr = tf.cast(logp_fn(theta0), dtype)
    logp_prop = tf.cast(logp_fn(theta_prop), dtype)
    log_alpha = logp_prop - logp_curr

    # ------------------------------------------------------------
    # 3) Accept / reject
    # ------------------------------------------------------------
    u = rng.uniform([], dtype=dtype)
    accepted = tf.math.log(u) < log_alpha
    theta_new = tf.where(accepted, theta_prop, theta0)

    return theta_new, accepted, log_alpha


def tune_k(
    theta0: tf.Tensor,
    step_fn,
    k0: float | tf.Tensor,
    pilot_length: int,
    target_low: float = 0.3,
    target_high: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Minimal (Option A) pilot tuner for a scalar proposal scale k.

    Runs ONE pilot chain of length `pilot_length`, computes acceptance rate,
    then adjusts k once:
      - if acc > target_high: k <- 1.2 k
      - if acc < target_low:  k <- 0.8 k
      - else: keep k

    Assumes `step_fn(theta, k)` returns (theta_new, accepted[, ...]).

    Returns (k_new, acc_rate, theta_end).
    """
    if pilot_length <= 0:
        raise ValueError("pilot_length must be positive.")
    if not (0.0 < target_low < target_high < 1.0):
        raise ValueError("Require 0 < target_low < target_high < 1.")

    # Hard-coded multiplicative adjustments (minimal).
    factor_up = 1.2
    factor_down = 0.8

    theta = tf.convert_to_tensor(theta0)
    dtype = theta.dtype

    k = tf.cast(tf.convert_to_tensor(k0), dtype)
    eps_k = tf.cast(1e-12, dtype)
    k = tf.maximum(k, eps_k)

    acc_sum = tf.cast(0.0, dtype)
    for _ in range(int(pilot_length)):
        out = step_fn(theta, k)
        theta = out[0]
        accepted = out[1]
        acc_sum += tf.cast(accepted, dtype)

    acc_rate = acc_sum / tf.cast(pilot_length, dtype)

    if acc_rate > tf.cast(target_high, dtype):
        k_new = tf.maximum(k * tf.cast(factor_up, dtype), eps_k)
    elif acc_rate < tf.cast(target_low, dtype):
        k_new = tf.maximum(k * tf.cast(factor_down, dtype), eps_k)
    else:
        k_new = k

    return k_new, acc_rate, theta


def sample_gamma_given_n_phi_market(
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
