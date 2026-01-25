# market_shock_estimators/tmh.py

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def tmh_step(
    *,
    theta0,
    logp_fn,
    ridge,
    max_lbfgs_iters,
    rng,
):
    """
    Perform one Laplace independence Metropolis–Hastings (TMH) step
    for a single parameter block.

    Parameters
    ----------
    theta0 : tf.Tensor, shape (d,)
        Current state of the parameter block.
    logp_fn : callable
        Function mapping theta -> scalar log posterior (tf.Tensor).
    ridge : float or tf.Tensor
        Ridge added to -Hessian for numerical stability.
    max_lbfgs_iters : int
        Maximum number of LBFGS iterations for mode finding.
    rng : tf.random.Generator
        TensorFlow RNG.

    Returns
    -------
    theta_new : tf.Tensor, shape (d,)
        Updated parameter block.
    accepted : tf.Tensor, scalar bool
        Whether the proposal was accepted.
    """
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    # ------------------------------------------------------------
    # 1. Dimension and kappa (Lu default)
    # ------------------------------------------------------------
    d = tf.cast(tf.size(theta0), dtype)
    kappa = tf.constant(2.38, dtype) / tf.sqrt(d)

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
    P = (-H + ridge * I) / (kappa * kappa)

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
