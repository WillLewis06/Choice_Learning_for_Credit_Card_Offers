from __future__ import annotations

import tensorflow as tf


def rw_mh_step(
    *,
    theta0,
    logp_fn,
    step_size,
    rng,
):
    """
    Perform one Random-Walk Metropolis–Hastings (RW-MH) step
    for a single parameter block.

    Parameters
    ----------
    theta0 : tf.Tensor, shape (d,) or scalar
        Current state of the parameter block.
    logp_fn : callable
        Function mapping theta -> scalar log posterior (tf.Tensor).
    step_size : float or tf.Tensor
        Scalar or vector step size for the Gaussian random walk.
    rng : tf.random.Generator
        TensorFlow RNG.

    Returns
    -------
    theta_new : tf.Tensor
        Updated parameter block.
    accepted : tf.Tensor, scalar bool
        Whether the proposal was accepted.
    """
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    step_size = tf.convert_to_tensor(step_size, dtype=dtype)

    # ------------------------------------------------------------
    # 1. Propose: theta' = theta + eps
    #    eps ~ N(0, step_size^2 I)   (scalar or diagonal)
    # ------------------------------------------------------------
    eps = rng.normal(tf.shape(theta0), dtype=dtype)
    theta_prop = theta0 + step_size * eps

    # ------------------------------------------------------------
    # 2. Log acceptance ratio (symmetric proposal)
    #    Ensure dtype consistency for tf.function compatibility.
    # ------------------------------------------------------------
    logp_curr = tf.cast(logp_fn(theta0), dtype)
    logp_prop = tf.cast(logp_fn(theta_prop), dtype)

    log_alpha = logp_prop - logp_curr

    # ------------------------------------------------------------
    # 3. Accept / reject
    # ------------------------------------------------------------
    u = rng.uniform([], dtype=dtype)
    accepted = tf.math.log(u) < log_alpha

    theta_new = tf.where(accepted, theta_prop, theta0)

    return theta_new, accepted
