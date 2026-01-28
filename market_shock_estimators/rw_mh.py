from __future__ import annotations

import tensorflow as tf


def rw_mh_step(theta0, logp_fn, k, rng: tf.random.Generator):
    """
    Perform one Random-Walk Metropolis–Hastings (RW-MH) step for a single block.

    This matches Lu-style Gaussian random-walk proposals:
      - If s is None:
          theta' = theta + k * z,   z ~ N(0, I)   (scalar or identity covariance)
      - If s is provided (Cholesky factor of S):
          theta' = theta + k * (s @ z),   z ~ N(0, I),  s s^T = S

    Parameters
    ----------
    theta0 : tf.Tensor
        Current state. Scalar or vector (d,).
    logp_fn : callable
        Function mapping theta -> scalar log density (tf.Tensor).
    k : float or tf.Tensor
        Scalar step scale.
    rng : tf.random.Generator
        TensorFlow RNG.
    s : tf.Tensor or None
        Optional Cholesky factor (d, d). If provided, theta0 must be rank-1 (d,).

    Returns
    -------
    theta_new : tf.Tensor
        Updated state (same shape/dtype as theta0).
    accepted : tf.Tensor
        Scalar bool indicating whether the proposal was accepted.
    log_alpha : tf.Tensor
        Scalar log acceptance ratio (dtype matches theta0).
    """
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
