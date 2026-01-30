# market_shock_estimators/lu_shrinkage_kernels.py
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


def rw_mh_step(theta0, logp_fn, k, rng: tf.random.Generator):
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype
    k = tf.cast(tf.convert_to_tensor(k), dtype)

    z = rng.normal(tf.shape(theta0), dtype=dtype)
    theta_prop = theta0 + k * z

    logp_curr = tf.cast(logp_fn(theta0), dtype)
    logp_prop = tf.cast(logp_fn(theta_prop), dtype)
    log_alpha = logp_prop - logp_curr

    u = rng.uniform([], dtype=dtype)
    accepted = tf.math.log(u) < log_alpha
    theta_new = tf.where(accepted, theta_prop, theta0)

    return theta_new, accepted, log_alpha


def tmh_step(
    theta0,
    logp_fn,
    ridge,
    rng,
    k,
):

    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    k = tf.cast(tf.convert_to_tensor(k), dtype)
    ridge = tf.cast(tf.convert_to_tensor(ridge), dtype)

    d = tf.shape(theta0)[0]
    I = tf.eye(d, dtype=dtype)

    eps_floor = tf.cast(1e-8, dtype)
    false = tf.constant(False)

    # --- Autograph-safe initializers for variables referenced in later control-flow
    # These get overwritten on the normal path, but must exist for all branches.
    delta = tf.zeros_like(theta0)  # vector
    gp = tf.zeros_like(theta0)  # vector
    jitter0 = tf.cast(0.0, dtype)  # scalar
    jitterp = tf.cast(0.0, dtype)  # scalar
    min_eig0 = tf.cast(0.0, dtype)  # scalar
    min_eigp = tf.cast(0.0, dtype)  # scalar
    log_alpha = tf.cast(0.0, dtype)  # scalar
    logu = tf.cast(0.0, dtype)  # scalar

    def _fallback(reason: str):
        tf.print("[TMH] fallback:", reason)
        return theta0, false

    def _lp_grad_hess(theta):
        theta = tf.convert_to_tensor(theta, dtype=dtype)
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(theta)
            with tf.GradientTape() as t1:
                t1.watch(theta)
                lp = logp_fn(theta)
            g = t1.gradient(lp, theta)
        H = t2.jacobian(g, theta)
        del t2
        return lp, g, H

    def _sym(A):
        return 0.5 * (A + tf.transpose(A))

    def _chol_spd(A):
        """
        Always returns tensors: (L, jitter, min_eig, ok).
        If ok is False, L is I and jitter/min_eig are zeros.
        """
        A = _sym(A)
        evals = tf.linalg.eigvalsh(A)

        ok_evals = tf.reduce_all(tf.math.is_finite(evals))
        min_eig = tf.where(ok_evals, tf.reduce_min(evals), tf.cast(0.0, dtype))

        jitter = tf.where(
            ok_evals,
            tf.maximum(tf.cast(0.0, dtype), -min_eig + eps_floor),
            tf.cast(0.0, dtype),
        )
        A_spd = A + jitter * I

        # If evals were not finite, don't trust cholesky(A_spd); return safe defaults.
        def _do_chol():
            L = tf.linalg.cholesky(A_spd)
            ok_L = tf.reduce_all(tf.math.is_finite(L))
            L_safe = tf.where(ok_L, L, I)
            return L_safe, ok_L

        def _do_fail():
            return I, tf.constant(False)

        L, ok_L = tf.cond(ok_evals, _do_chol, _do_fail)
        ok = tf.logical_and(ok_evals, ok_L)

        return L, jitter, min_eig, ok

    def _invA_times_vec_from_chol(L, v):
        # A^{-1} v via Cholesky: solve A x = v
        x = tf.linalg.cholesky_solve(L, v[:, None])
        return x[:, 0]

    def _sample_from_cov_invA_from_chol(L):
        # delta ~ N(0, k^2 * A^{-1})
        # If A = L L^T, then A^{-1/2} z = L^{-T} z
        z = rng.normal([d], dtype=dtype)
        x = tf.linalg.triangular_solve(tf.transpose(L), z[:, None], lower=False)[:, 0]
        return k * x

    def _log_q_from_chol(x, mu_theta, L):
        # Using precision P = A / k^2:
        # log q(x) = 0.5 logdet(P) - 0.5 (x-mu)^T P (x-mu) + const
        diff = x - mu_theta

        # quad = diff^T (A/k^2) diff = ||L^T diff||^2 / k^2
        y = tf.linalg.matvec(tf.transpose(L), diff)
        quad = tf.reduce_sum(tf.square(y)) / (k * k)

        # logdet(P) = logdet(A) - d*log(k^2), and logdet(A)=2*sum(log(diag(L)))
        logdetA = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        logdetP = logdetA - tf.cast(d, dtype) * tf.math.log(k * k)

        return 0.5 * logdetP - 0.5 * quad

    # --- Params at theta0
    lp0, g0, H0 = _lp_grad_hess(theta0)
    if (g0 is None) or (H0 is None):
        return _fallback("gradient/Hessian is None at theta0")
    if not tf.reduce_all(tf.math.is_finite(lp0)):
        return _fallback("non-finite logp(theta0)")
    if not tf.reduce_all(tf.math.is_finite(g0)):
        return _fallback("non-finite grad(theta0)")
    if not tf.reduce_all(tf.math.is_finite(H0)):
        return _fallback("non-finite Hessian(theta0)")

    A0 = -H0 + ridge * I
    L0, jitter0, min_eig0, ok0 = _chol_spd(A0)
    if L0 is None:
        return _fallback("failed SPD Cholesky at theta0")

    # mu0 = theta0 + 0.5*k^2 * A0^{-1} g0
    invA0_g0 = _invA_times_vec_from_chol(L0, g0)
    mu0 = theta0 + tf.cast(0.5, dtype) * (k * k) * invA0_g0

    # propose
    delta = _sample_from_cov_invA_from_chol(L0)
    prop = mu0 + delta

    if not tf.reduce_all(tf.math.is_finite(prop)):
        return _fallback("non-finite proposal")

    # --- Params at prop (for reverse q)
    lpp, gp, Hp = _lp_grad_hess(prop)
    if (gp is None) or (Hp is None):
        return _fallback("gradient/Hessian is None at proposal")
    if not tf.reduce_all(tf.math.is_finite(lpp)):
        return _fallback("non-finite logp(prop)")
    if not tf.reduce_all(tf.math.is_finite(gp)):
        return _fallback("non-finite grad(prop)")
    if not tf.reduce_all(tf.math.is_finite(Hp)):
        return _fallback("non-finite Hessian(prop)")

    Ap = -Hp + ridge * I
    Lp, jitterp, min_eigp, okp = _chol_spd(Ap)
    if Lp is None:
        return _fallback("failed SPD Cholesky at proposal")

    invAp_gp = _invA_times_vec_from_chol(Lp, gp)
    mup = prop + tf.cast(0.5, dtype) * (k * k) * invAp_gp

    # MH ratio with asymmetric proposals

    logq_prop_given_0 = _log_q_from_chol(prop, mu0, L0)
    logq_0_given_prop = _log_q_from_chol(theta0, mup, Lp)

    log_alpha = lpp + logq_0_given_prop - lp0 - logq_prop_given_0
    if not tf.reduce_all(tf.math.is_finite(log_alpha)):
        return _fallback("non-finite log_alpha")

    u = rng.uniform([], dtype=dtype)
    logu = tf.math.log(u)
    accepted = logu < log_alpha

    def _print_reject():
        tf.print(
            "[TMH] reject:",
            "log_alpha=",
            log_alpha,
            "logu=",
            logu,
            "||g0||=",
            tf.norm(g0),
            "||gp||=",
            tf.norm(gp),
            "min_eig(A0)=",
            min_eig0,
            "min_eig(Ap)=",
            min_eigp,
            "jitter0=",
            jitter0,
            "jitterp=",
            jitterp,
            "||delta||=",
            tf.norm(delta),
            "k=",
            k,
            "ridge=",
            ridge,
        )
        return 0

    tf.cond(accepted, lambda: 0, _print_reject)

    theta_new = tf.where(accepted, prop, theta0)
    return theta_new, accepted


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
    Gibbs update for gamma_t given n_t and phi_t under the Lu spike-and-slab.
    Returns gamma_t as float64 0/1 (caller can cast).
    """
    njt_t = tf.convert_to_tensor(njt_t)
    dtype = njt_t.dtype

    phi_t = tf.cast(phi_t, dtype)
    eps = tf.cast(1e-30, dtype)

    logp0 = -0.5 * (njt_t * njt_t) / T0_sq - 0.5 * log_T0_sq
    logp1 = -0.5 * (njt_t * njt_t) / T1_sq - 0.5 * log_T1_sq

    log_a = tf.math.log(phi_t + eps) + logp1
    log_b = tf.math.log(1.0 - phi_t + eps) + logp0
    m = tf.maximum(log_a, log_b)
    prob1 = tf.exp(log_a - m) / (tf.exp(log_a - m) + tf.exp(log_b - m))

    u = rng.uniform(tf.shape(prob1), dtype=dtype)
    return tf.cast(u < prob1, dtype=dtype)


def gibbs_phi_market(
    *,
    gamma_t: tf.Tensor,
    a_phi: float | tf.Tensor,
    b_phi: float | tf.Tensor,
    J: int,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Gibbs update for phi_t | gamma_t under Beta-Bernoulli:
      phi_t ~ Beta(a_phi + sum_j gamma_jt, b_phi + J - sum_j gamma_jt)

    Uses stateless gamma with seeds from `rng` (Generator has no .gamma in TF 2.16).
    """
    gamma_t = tf.convert_to_tensor(gamma_t)
    dtype = gamma_t.dtype

    a_phi = tf.cast(tf.convert_to_tensor(a_phi), dtype)
    b_phi = tf.cast(tf.convert_to_tensor(b_phi), dtype)

    s = tf.cast(tf.reduce_sum(tf.cast(gamma_t, tf.int32)), dtype)
    J_t = tf.cast(tf.convert_to_tensor(J), dtype)

    a = a_phi + s
    b = b_phi + (J_t - s)

    seeds = rng.make_seeds(2)
    x = tf.random.stateless_gamma(shape=[], seed=seeds[0], alpha=a, dtype=dtype)
    y = tf.random.stateless_gamma(shape=[], seed=seeds[1], alpha=b, dtype=dtype)
    return x / (x + y)
