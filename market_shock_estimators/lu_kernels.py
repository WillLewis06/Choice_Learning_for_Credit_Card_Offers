"""
MCMC transition kernels used by the Lu shrinkage estimator.

This module provides:
  - Random-walk Metropolis–Hastings (RW-MH) for scalar/vector blocks.
  - Manifold Metropolis–Hastings (TMH) for dense parameter blocks using local curvature.
  - Gibbs updates for sparsity indicators (gamma) and inclusion rates (phi).

All kernels are TensorFlow graph compatible and operate in tf.float64.
"""

from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf

from market_shock_estimators.lu_validate_input import tmh_step_validate_input_tf


@tf.function(reduce_retracing=True)
def rw_mh_step(
    theta0: tf.Tensor,
    logp_fn: Callable[[tf.Tensor], tf.Tensor],
    k: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run a random-walk Metropolis–Hastings update.

    Proposal:
        theta' = theta0 + k z,  z ~ Normal(0, I).

    This supports elementwise/batched updates. If theta0 has shape (B,),
    then `logp_fn` must return a tensor of shape (B,) giving per-element log
    densities, and acceptance is applied elementwise.

    Args:
        theta0: Current parameter (scalar or batch), tf.float64.
        logp_fn: Callable returning log density at theta (scalar or elementwise).
        k: Proposal scale, scalar tf.float64.
        rng: TensorFlow RNG for Normal and Uniform draws.

    Returns:
        theta_new: Updated theta with same shape as theta0.
        accepted: Boolean tensor with same shape as theta0.
    """
    # Symmetric Gaussian random-walk proposal.
    z = rng.normal(tf.shape(theta0), dtype=tf.float64)
    theta_prop = theta0 + k * z

    # Symmetric MH ratio uses only the log density difference.
    logp_curr = logp_fn(theta0)
    logp_prop = logp_fn(theta_prop)
    log_alpha = logp_prop - logp_curr

    # Accept/reject (elementwise if theta0 is batched).
    u = rng.uniform(tf.shape(theta0), dtype=tf.float64)
    accepted = tf.math.log(u) < log_alpha
    theta_new = tf.where(accepted, theta_prop, theta0)

    return theta_new, accepted


@tf.function(reduce_retracing=True)
def tmh_step(
    theta0: tf.Tensor,
    logp_fn: Callable[[tf.Tensor], tf.Tensor],
    ridge: tf.Tensor,
    rng: tf.random.Generator,
    k: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run a TensorFlow-safe manifold Metropolis–Hastings (TMH) update.

    TMH builds a local Gaussian proposal using the gradient and Hessian of the
    log density lp(theta) = logp_fn(theta).

    Definitions:
      g(theta) = ∇ lp(theta)
      H(theta) = ∇² lp(theta)
      A(theta) = -H(theta) + ridge I        (local precision; must be SPD)

    Proposal:
      mu0 = theta0 + 0.5 k^2 A0^{-1} g0
      theta' ~ Normal(mu0, k^2 A0^{-1})

    Because the proposal depends on theta, the MH correction uses both forward
    and reverse proposal densities.

    Args:
        theta0: Current parameter, shape (d,), tf.float64.
        logp_fn: Callable returning scalar tf.float64 log density.
        ridge: Nonnegative scalar added to A(theta) for stability.
        rng: TensorFlow RNG for proposal and accept/reject draws.
        k: Proposal scale, scalar tf.float64.

    Returns:
        theta_new: Updated theta, shape (d,).
        accepted: Boolean scalar indicating acceptance.
    """
    tmh_step_validate_input_tf(theta0=theta0, k=k, ridge=ridge)

    d = tf.shape(theta0)[0]
    I = tf.eye(d, dtype=tf.float64)

    # Minimum eigenvalue floor after jitter to enforce SPD.
    eps_floor = tf.constant(1e-8, dtype=tf.float64)
    false = tf.constant(False)

    def _all_finite(*xs: tf.Tensor) -> tf.Tensor:
        """Return True if all values in all tensors are finite."""
        ok = tf.constant(True)
        for x in xs:
            ok = tf.logical_and(ok, tf.reduce_all(tf.math.is_finite(x)))
        return ok

    def _lp_grad_hess(theta: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute log density, gradient, and Hessian at theta."""
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
        return lp, g, H

    def _sym(A: tf.Tensor) -> tf.Tensor:
        """Symmetrize a square matrix."""
        return 0.5 * (A + tf.transpose(A))

    def _chol_spd(A: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute a Cholesky factor after jittering A to be SPD.

        Strategy:
          1) Symmetrize A.
          2) Compute eigenvalues and shift by jitter so min eigenvalue >= eps_floor.
          3) Cholesky factorize the shifted matrix.

        Returns:
            L: Cholesky factor (lower-triangular), or identity on failure.
            ok: Boolean scalar indicating success.
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
            """Attempt Cholesky factorization of the jittered matrix."""
            L = tf.linalg.cholesky(A_spd)
            ok_L = tf.reduce_all(tf.math.is_finite(L))
            return tf.where(ok_L, L, I), ok_L

        def _fail_chol() -> Tuple[tf.Tensor, tf.Tensor]:
            """Return a safe fallback factorization result on failure."""
            return I, tf.constant(False)

        L, ok_L = tf.cond(ok_evals, _do_chol, _fail_chol)
        ok = tf.logical_and(ok_evals, ok_L)
        return L, ok

    def _solve_Ainv(L: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        """Compute A^{-1} v given chol(A)=L."""
        x = tf.linalg.cholesky_solve(L, v[:, None])
        return x[:, 0]

    def _sample_kAinv(L: tf.Tensor) -> tf.Tensor:
        """Sample delta ~ Normal(0, k^2 A^{-1}) given chol(A)=L."""
        # If A = L L^T then A^{-1/2} z = L^{-T} z.
        z = rng.normal(tf.shape(theta0), dtype=tf.float64)
        x = tf.linalg.triangular_solve(tf.transpose(L), z[:, None], lower=False)[:, 0]
        return k * x

    def _log_q_gaussian_precision(
        x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor
    ) -> tf.Tensor:
        """Compute log q(x | mu) for precision A/k^2, up to a constant."""
        # Quadratic form: (x-mu)^T (A/k^2) (x-mu) = ||L^T (x-mu)||^2 / k^2.
        diff = x - mu
        y = tf.linalg.matvec(tf.transpose(L), diff)
        quad = tf.reduce_sum(tf.square(y)) / (k * k)

        # logdet(A/k^2) = logdet(A) - d*log(k^2), logdet(A)=2*sum(log(diag(L))).
        logdetA = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        logdetP = logdetA - tf.cast(d, tf.float64) * tf.math.log(k * k)

        return 0.5 * logdetP - 0.5 * quad

    def _geometry(
        theta: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute local geometry (lp, grad, chol(A), ok) at theta."""
        lp, g, H = _lp_grad_hess(theta)
        A = -H + ridge * I
        L, ok_chol = _chol_spd(A)
        ok = tf.logical_and(_all_finite(lp, g, H), ok_chol)
        return lp, g, L, ok

    def _fallback(msg: tf.Tensor | str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return current state and reject, emitting a short message."""
        tf.print("[TMH] fallback:", msg)
        return theta0, false

    # -------------------------------------------------------------------------
    # Step 1: compute local geometry at the current point theta0
    # -------------------------------------------------------------------------
    lp0, g0, L0, ok0 = _geometry(theta0)

    def _do_step() -> Tuple[tf.Tensor, tf.Tensor]:
        """Propose theta' from local Gaussian and apply MH correction."""
        # ---------------------------------------------------------------------
        # Step 2: build proposal mean mu0 = theta0 + 0.5*k^2*A0^{-1} g0
        # ---------------------------------------------------------------------
        invA0_g0 = _solve_Ainv(L0, g0)
        mu0 = theta0 + tf.constant(0.5, tf.float64) * (k * k) * invA0_g0

        # ---------------------------------------------------------------------
        # Step 3: draw proposal theta' ~ Normal(mu0, k^2 A0^{-1})
        # ---------------------------------------------------------------------
        prop = mu0 + _sample_kAinv(L0)
        ok_prop = tf.reduce_all(tf.math.is_finite(prop))

        def _do_from_prop() -> Tuple[tf.Tensor, tf.Tensor]:
            """Compute reverse proposal terms at theta' and accept/reject."""
            # Reverse density requires geometry at the proposal.
            lpp, gp, Lp, okp = _geometry(prop)

            def _do_accept_reject() -> Tuple[tf.Tensor, tf.Tensor]:
                """Compute MH ratio with asymmetry correction and accept/reject."""
                # Reverse mean: mup = prop + 0.5*k^2*Ap^{-1} gp.
                invAp_gp = _solve_Ainv(Lp, gp)
                mup = prop + tf.constant(0.5, tf.float64) * (k * k) * invAp_gp

                # Forward and reverse proposal log densities (up to constants).
                logq_prop_given_0 = _log_q_gaussian_precision(prop, mu0, L0)
                logq_0_given_prop = _log_q_gaussian_precision(theta0, mup, Lp)

                # Full MH ratio for asymmetric proposals.
                log_alpha = lpp + logq_0_given_prop - lp0 - logq_prop_given_0
                ok_alpha = tf.reduce_all(tf.math.is_finite(log_alpha))

                def _mh_accept_reject() -> Tuple[tf.Tensor, tf.Tensor]:
                    """Apply MH accept/reject using log_alpha."""
                    u = rng.uniform([], dtype=tf.float64)
                    accepted = tf.math.log(u) < log_alpha
                    theta_new = tf.where(accepted, prop, theta0)
                    return theta_new, accepted

                def _alpha_invalid() -> Tuple[tf.Tensor, tf.Tensor]:
                    """Reject when the acceptance ratio is not finite."""
                    return _fallback("non-finite log_alpha")

                return tf.cond(ok_alpha, _mh_accept_reject, _alpha_invalid)

            ok_all = tf.logical_and(okp, ok_prop)

            def _accept_path() -> Tuple[tf.Tensor, tf.Tensor]:
                """Proceed to MH correction when proposal geometry is valid."""
                return _do_accept_reject()

            def _reject_path() -> Tuple[tf.Tensor, tf.Tensor]:
                """Reject when proposal geometry is invalid."""
                return _fallback("non-finite / non-SPD at proposal")

            return tf.cond(ok_all, _accept_path, _reject_path)

        def _proposal_invalid() -> Tuple[tf.Tensor, tf.Tensor]:
            """Reject when the proposed theta' is not finite."""
            return _fallback("non-finite proposal")

        return tf.cond(ok_prop, _do_from_prop, _proposal_invalid)

    def _theta0_invalid() -> Tuple[tf.Tensor, tf.Tensor]:
        """Reject when the local geometry at theta0 is invalid."""
        return _fallback("non-finite / non-SPD at theta0")

    return tf.cond(ok0, _do_step, _theta0_invalid)


@tf.function(reduce_retracing=True)
def gibbs_gamma(
    njt_t: tf.Tensor,
    phi_t: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    log_T0_sq: tf.Tensor,
    log_T1_sq: tf.Tensor,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """Sample gamma_t under a spike-and-slab normal prior.

    Model for a single market t:
      n_{j,t} | gamma_{j,t}=0 ~ Normal(0, T0_sq)   (spike)
      n_{j,t} | gamma_{j,t}=1 ~ Normal(0, T1_sq)   (slab)
      gamma_{j,t} | phi_t    ~ Bernoulli(phi_t)

    This computes P(gamma_{j,t}=1 | n_{j,t}, phi_t) for each j and samples.

    Args:
        njt_t: Market-product shocks n_t, shape (J,).
        phi_t: Inclusion rate for market t, scalar.
        T0_sq: Spike variance, scalar.
        T1_sq: Slab variance, scalar.
        log_T0_sq: Precomputed log(T0_sq).
        log_T1_sq: Precomputed log(T1_sq).
        rng: TensorFlow RNG for Uniform draws.

    Returns:
        gamma_t: 0/1 indicators, shape (J,), dtype tf.float64.
    """
    # eps prevents log(0) when phi_t is extremely close to 0 or 1.
    eps = tf.constant(1e-30, dtype=tf.float64)

    # Component log densities for n_{j,t}, dropping constants that cancel.
    logp0 = -0.5 * (njt_t * njt_t) / T0_sq - 0.5 * log_T0_sq
    logp1 = -0.5 * (njt_t * njt_t) / T1_sq - 0.5 * log_T1_sq

    # Posterior log weights for gamma=1 and gamma=0.
    log_a = tf.math.log(phi_t + eps) + logp1
    log_b = tf.math.log(1.0 - phi_t + eps) + logp0

    # Stable conversion to probabilities via log-sum-exp.
    m = tf.maximum(log_a, log_b)
    prob1 = tf.exp(log_a - m) / (tf.exp(log_a - m) + tf.exp(log_b - m))

    # Bernoulli draw per product.
    u = rng.uniform(tf.shape(prob1), dtype=tf.float64)
    return tf.cast(u < prob1, dtype=tf.float64)


@tf.function(reduce_retracing=True)
def gibbs_phi(
    gamma: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """Sample phi under Beta–Bernoulli conjugacy.

    For each market t:
      phi_t ~ Beta(a_phi, b_phi)
      gamma_{t,j} | phi_t ~ Bernoulli(phi_t)

    Posterior:
      phi_t | gamma_t ~ Beta(a_phi + sum_j gamma_{t,j},
                             b_phi + J - sum_j gamma_{t,j})

    Args:
        gamma: Indicators, shape (T, J) or (J,), values in {0,1}.
        a_phi: Prior Beta parameter a, scalar tf.float64.
        b_phi: Prior Beta parameter b, scalar tf.float64.
        rng: TensorFlow RNG used to generate stateless seeds.

    Returns:
        phi: Shape (T,) if gamma is (T, J), else scalar if gamma is (J,).
    """
    # Count active indicators per market.
    s = tf.reduce_sum(gamma, axis=-1)  # (T,) or ()

    # J as float64 to match a_phi/b_phi dtype.
    J = tf.cast(tf.shape(gamma)[-1], tf.float64)

    # Posterior Beta parameters.
    a = a_phi + s
    b = b_phi + (J - s)

    # Sample Beta(a,b) via Gamma(a,1) / (Gamma(a,1) + Gamma(b,1)).
    seeds = rng.make_seeds(2)
    x = tf.random.stateless_gamma(
        shape=tf.shape(s), seed=seeds[0], alpha=a, dtype=tf.float64
    )
    y = tf.random.stateless_gamma(
        shape=tf.shape(s), seed=seeds[1], alpha=b, dtype=tf.float64
    )
    return x / (x + y)
