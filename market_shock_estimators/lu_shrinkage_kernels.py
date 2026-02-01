# market_shock_estimators/lu_shrinkage_kernels.py
from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf


@tf.function(reduce_retracing=True)
def rw_mh_step(
    theta0: tf.Tensor,
    logp_fn: Callable[[tf.Tensor], tf.Tensor],
    k: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Random-walk MH step:
      theta' = theta0 + k * z,  z ~ N(0, I)

    Assumes inputs are tf.float64 tensors. Keeps casting logp outputs for safety.
    """
    z = rng.normal(tf.shape(theta0), dtype=tf.float64)
    theta_prop = theta0 + k * z

    logp_curr = tf.cast(logp_fn(theta0), tf.float64)
    logp_prop = tf.cast(logp_fn(theta_prop), tf.float64)
    log_alpha = logp_prop - logp_curr

    u = rng.uniform([], dtype=tf.float64)
    accepted = tf.math.log(u) < log_alpha
    theta_new = tf.where(accepted, theta_prop, theta0)

    return theta_new, accepted, log_alpha


@tf.function(reduce_retracing=True)
def tmh_step(
    theta0: tf.Tensor,
    logp_fn: Callable[[tf.Tensor], tf.Tensor],
    ridge: tf.Tensor,
    rng: tf.random.Generator,
    k: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow-graph-safe TMH step (no Python branching on tensors).
    Core TMH mechanism is unchanged:
      - local quadratic drift using A = -H + ridge*I
      - proposal: N(mu0, k^2 A0^{-1})
      - asymmetric MH correction using forward/reverse q from local metrics

    Assumes inputs are tf.float64 tensors. Keeps casting logp/grad/Hess for safety.
    """
    d = tf.shape(theta0)[0]
    I = tf.eye(d, dtype=tf.float64)
    eps_floor = tf.constant(1e-8, dtype=tf.float64)
    false = tf.constant(False)

    # --- Helpers (all TF-safe) -------------------------------------------------

    def _lp_grad_hess(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(theta)
            with tf.GradientTape() as t1:
                t1.watch(theta)
                lp = logp_fn(theta)
            g = t1.gradient(
                lp, theta, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
        H = t2.jacobian(g, theta, experimental_use_pfor=False)
        del t2

        # Safety casts (do not change core logic)
        lp = tf.cast(lp, tf.float64)
        g = tf.cast(g, tf.float64)
        H = tf.cast(H, tf.float64)
        return lp, g, H

    def _sym(A: tf.Tensor) -> tf.Tensor:
        return 0.5 * (A + tf.transpose(A))

    def _chol_spd(A: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Always returns tensors: (L, jitter, min_eig, ok).
        If ok is False, L is I and jitter/min_eig are zeros.
        """
        A = _sym(A)
        evals = tf.linalg.eigvalsh(A)

        ok_evals = tf.reduce_all(tf.math.is_finite(evals))
        min_eig = tf.where(ok_evals, tf.reduce_min(evals), tf.constant(0.0, tf.float64))

        jitter = tf.where(
            ok_evals,
            tf.maximum(tf.constant(0.0, tf.float64), -min_eig + eps_floor),
            tf.constant(0.0, tf.float64),
        )
        A_spd = A + jitter * I

        def _do_chol() -> Tuple[tf.Tensor, tf.Tensor]:
            L = tf.linalg.cholesky(A_spd)
            ok_L = tf.reduce_all(tf.math.is_finite(L))
            L_safe = tf.where(ok_L, L, I)
            return L_safe, ok_L

        def _do_fail() -> Tuple[tf.Tensor, tf.Tensor]:
            return I, tf.constant(False)

        L, ok_L = tf.cond(ok_evals, _do_chol, _do_fail)
        ok = tf.logical_and(ok_evals, ok_L)
        return L, jitter, min_eig, ok

    def _invA_times_vec_from_chol(L: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        x = tf.linalg.cholesky_solve(L, v[:, None])
        return x[:, 0]

    def _sample_from_cov_invA_from_chol(L: tf.Tensor) -> tf.Tensor:
        # delta ~ N(0, k^2 * A^{-1}), with A = L L^T  =>  A^{-1/2} z = L^{-T} z
        z = rng.normal(tf.shape(theta0), dtype=tf.float64)
        x = tf.linalg.triangular_solve(tf.transpose(L), z[:, None], lower=False)[:, 0]
        return k * x

    def _log_q_from_chol(x: tf.Tensor, mu_theta: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
        # Precision P = A/k^2. Using chol(A)=L:
        # quad = (x-mu)^T P (x-mu) = ||L^T (x-mu)||^2 / k^2
        diff = x - mu_theta
        y = tf.linalg.matvec(tf.transpose(L), diff)
        quad = tf.reduce_sum(tf.square(y)) / (k * k)

        # logdet(P) = logdet(A) - d*log(k^2), logdet(A)=2*sum(log(diag(L)))
        logdetA = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        logdetP = logdetA - tf.cast(d, tf.float64) * tf.math.log(k * k)
        return 0.5 * logdetP - 0.5 * quad

    def _fallback(msg: tf.Tensor | str) -> Tuple[tf.Tensor, tf.Tensor]:
        tf.print("[TMH] fallback:", msg)
        return theta0, false

    # --- Precompute at theta0 --------------------------------------------------
    lp0, g0, H0 = _lp_grad_hess(theta0)

    ok_lp0 = tf.reduce_all(tf.math.is_finite(lp0))
    ok_g0 = tf.reduce_all(tf.math.is_finite(g0))
    ok_H0 = tf.reduce_all(tf.math.is_finite(H0))
    ok0_basic = tf.logical_and(ok_lp0, tf.logical_and(ok_g0, ok_H0))

    A0 = -H0 + ridge * I
    L0, jitter0, min_eig0, ok0_chol = _chol_spd(A0)
    ok0 = tf.logical_and(ok0_basic, ok0_chol)

    # --- Main step -------------------------------------------------------------
    def _do_step_from_theta0() -> Tuple[tf.Tensor, tf.Tensor]:
        # mu0 = theta0 + 0.5*k^2 * A0^{-1} g0
        invA0_g0 = _invA_times_vec_from_chol(L0, g0)
        mu0 = theta0 + tf.constant(0.5, tf.float64) * (k * k) * invA0_g0

        # propose
        delta = _sample_from_cov_invA_from_chol(L0)
        prop = mu0 + delta

        ok_prop = tf.reduce_all(tf.math.is_finite(prop))

        def _do_from_prop() -> Tuple[tf.Tensor, tf.Tensor]:
            lpp, gp, Hp = _lp_grad_hess(prop)

            ok_lpp = tf.reduce_all(tf.math.is_finite(lpp))
            ok_gp = tf.reduce_all(tf.math.is_finite(gp))
            ok_Hp = tf.reduce_all(tf.math.is_finite(Hp))
            okp_basic = tf.logical_and(ok_lpp, tf.logical_and(ok_gp, ok_Hp))

            Ap = -Hp + ridge * I
            Lp, jitterp, min_eigp, okp_chol = _chol_spd(Ap)
            okp = tf.logical_and(okp_basic, okp_chol)

            def _do_accept_reject() -> Tuple[tf.Tensor, tf.Tensor]:
                invAp_gp = _invA_times_vec_from_chol(Lp, gp)
                mup = prop + tf.constant(0.5, tf.float64) * (k * k) * invAp_gp

                logq_prop_given_0 = _log_q_from_chol(prop, mu0, L0)
                logq_0_given_prop = _log_q_from_chol(theta0, mup, Lp)

                log_alpha = lpp + logq_0_given_prop - lp0 - logq_prop_given_0
                ok_alpha = tf.reduce_all(tf.math.is_finite(log_alpha))

                def _do_mh() -> Tuple[tf.Tensor, tf.Tensor]:
                    u = rng.uniform([], dtype=tf.float64)
                    logu = tf.math.log(u)
                    accepted = logu < log_alpha

                    def _print_reject() -> tf.Tensor:
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
                        return tf.constant(0)

                    # tf.cond(accepted, lambda: tf.constant(0), _print_reject)

                    theta_new = tf.where(accepted, prop, theta0)
                    return theta_new, accepted

                return tf.cond(
                    ok_alpha, _do_mh, lambda: _fallback("non-finite log_alpha")
                )

            return tf.cond(
                okp,
                _do_accept_reject,
                lambda: _fallback("non-finite / non-SPD at proposal"),
            )

        return tf.cond(ok_prop, _do_from_prop, lambda: _fallback("non-finite proposal"))

    return tf.cond(
        ok0, _do_step_from_theta0, lambda: _fallback("non-finite / non-SPD at theta0")
    )


@tf.function(reduce_retracing=True)
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
    Returns gamma_t as tf.float64 0/1.
    """
    eps = tf.constant(1e-30, dtype=tf.float64)

    logp0 = -0.5 * (njt_t * njt_t) / T0_sq - 0.5 * log_T0_sq
    logp1 = -0.5 * (njt_t * njt_t) / T1_sq - 0.5 * log_T1_sq

    log_a = tf.math.log(phi_t + eps) + logp1
    log_b = tf.math.log(1.0 - phi_t + eps) + logp0
    m = tf.maximum(log_a, log_b)
    prob1 = tf.exp(log_a - m) / (tf.exp(log_a - m) + tf.exp(log_b - m))

    u = rng.uniform(tf.shape(prob1), dtype=tf.float64)
    return tf.cast(u < prob1, dtype=tf.float64)


@tf.function(reduce_retracing=True)
def gibbs_phi_market(
    *,
    gamma_t: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Gibbs update for phi_t | gamma_t under Beta-Bernoulli:
      phi_t ~ Beta(a_phi + sum_j gamma_jt, b_phi + J - sum_j gamma_jt)

    Uses stateless gamma with seeds from `rng` (Generator has no .gamma in TF 2.16).
    Assumes gamma_t is 0/1 in tf.float64, and a_phi/b_phi are scalar tf.float64 tensors.
    """
    s = tf.reduce_sum(gamma_t)
    J = tf.cast(tf.shape(gamma_t)[0], tf.float64)

    a = a_phi + s
    b = b_phi + (J - s)

    seeds = rng.make_seeds(2)
    x = tf.random.stateless_gamma(shape=[], seed=seeds[0], alpha=a, dtype=tf.float64)
    y = tf.random.stateless_gamma(shape=[], seed=seeds[1], alpha=b, dtype=tf.float64)
    return x / (x + y)
