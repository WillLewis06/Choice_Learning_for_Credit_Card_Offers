"""
tmh.py

Tailored Metropolis–Hastings (TMH) aligned with Lu (2025), Section 4.

Lu-style TMH differs from a state-dependent Newton/MALA proposal. It:
1) Finds the conditional posterior mode for a parameter block via Newton steps.
2) Forms a Laplace (Gaussian) approximation at the mode.
3) Proposes from an independence Gaussian centered at the mode:
       theta' ~ N(theta_hat, kappa^2 * V_hat),
   where V_hat^{-1} = -∇^2 log p(theta | rest) evaluated at theta_hat.
4) Accept/reject with an independence-MH ratio using the same proposal density.

This module is algorithmic only. It does not know the model. It requires:
- log_posterior(theta): float
- grad_log_posterior(theta): (d,) array
- hess_log_posterior(theta): (d,d) array, the Hessian of log posterior

Numerical differentiation is provided as a fallback but is not Lu-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ----------------------------------------------------------------------
# State container
# ----------------------------------------------------------------------


@dataclass
class TMHState:
    """Container for a TMH block state."""

    theta: np.ndarray
    logp: float


# ----------------------------------------------------------------------
# Numerical differentiation (fallback only; not Lu-aligned)
# ----------------------------------------------------------------------


def compute_gradient(theta: np.ndarray, log_posterior, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of log_posterior (fallback)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be 1D.")
    d = theta.size
    grad = np.zeros(d, dtype=float)
    for i in range(d):
        step = np.zeros(d, dtype=float)
        step[i] = eps
        grad[i] = (log_posterior(theta + step) - log_posterior(theta - step)) / (
            2.0 * eps
        )
    return grad


def compute_hessian(theta: np.ndarray, log_posterior, eps: float = 1e-4) -> np.ndarray:
    """
    Central-difference Hessian of log_posterior (fallback).

    Returns Hessian ∇² log p(theta).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be 1D.")
    d = theta.size
    H = np.zeros((d, d), dtype=float)
    f0 = log_posterior(theta)

    for i in range(d):
        ei = np.zeros(d, dtype=float)
        ei[i] = eps
        f_ip = log_posterior(theta + ei)
        f_im = log_posterior(theta - ei)

        # diagonal second derivative
        H[i, i] = (f_ip - 2.0 * f0 + f_im) / (eps**2)

        for j in range(i + 1, d):
            ej = np.zeros(d, dtype=float)
            ej[j] = eps
            f_pp = log_posterior(theta + ei + ej)
            f_pm = log_posterior(theta + ei - ej)
            f_mp = log_posterior(theta - ei + ej)
            f_mm = log_posterior(theta - ei - ej)

            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            H[i, j] = val
            H[j, i] = val

    return H


# ----------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------


def _regularize_neg_hessian(neg_hess: np.ndarray, ridge: float) -> np.ndarray:
    """
    Regularize a negative Hessian matrix to ensure positive definiteness:
        H_reg = H + ridge * I
    where H is intended to be -∇² log p at some point.
    """
    H = np.asarray(neg_hess, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("neg_hess must be square.")
    d = H.shape[0]
    return H + float(ridge) * np.eye(d, dtype=float)


def _mvnorm_logpdf(
    x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray, logdet_cov: float
) -> float:
    """
    Log density of N(mean, cov), up to the additive constant -d/2*log(2π),
    which cancels in MH ratios if dimensions are fixed.

    Includes the log-determinant term (required for correctness).
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    diff = x - mean
    quad = float(diff @ cov_inv @ diff)
    return -0.5 * quad - 0.5 * float(logdet_cov)


def _cov_inv_and_logdet(cov: np.ndarray) -> tuple[np.ndarray, float]:
    cov = np.asarray(cov, dtype=float)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance is not positive definite.")
    cov_inv = np.linalg.inv(cov)
    return cov_inv, float(logdet)


# ----------------------------------------------------------------------
# Lu-style mode finding (Newton)
# ----------------------------------------------------------------------


def newton_find_mode(
    theta_init: np.ndarray,
    log_posterior,
    grad_log_posterior,
    hess_log_posterior,
    *,
    ridge: float = 1e-6,
    max_iter: int = 25,
    tol: float = 1e-8,
    backtrack: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Find the mode of log_posterior via Newton ascent.

    Updates:
        theta <- theta + step
    where step solves:
        (-∇² log p(theta) + ridge I) step = ∇ log p(theta)

    Parameters
    ----------
    theta_init : (d,) array
    log_posterior : callable(theta)->float
    grad_log_posterior : callable(theta)->(d,)
    hess_log_posterior : callable(theta)->(d,d)  (Hessian of log posterior)
    ridge : float
        Regularization added to the negative Hessian.
    max_iter : int
    tol : float
        Stop when ||grad||_inf <= tol.
    backtrack : bool
        If True, do simple backtracking if log posterior decreases.

    Returns
    -------
    theta_hat : (d,) array
    logp_hat : float
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    if theta.ndim != 1:
        raise ValueError("theta_init must be 1D.")

    logp = float(log_posterior(theta))

    for _ in range(int(max_iter)):
        g = np.asarray(grad_log_posterior(theta), dtype=float)
        if g.shape != theta.shape:
            raise ValueError("grad_log_posterior returned wrong shape.")
        if np.max(np.abs(g)) <= tol:
            break

        hess = np.asarray(hess_log_posterior(theta), dtype=float)
        if hess.shape != (theta.size, theta.size):
            raise ValueError("hess_log_posterior returned wrong shape.")

        neg_hess = -hess
        H_reg = _regularize_neg_hessian(neg_hess, ridge=ridge)

        # Newton step: H_reg * step = g
        step = np.linalg.solve(H_reg, g)

        if not backtrack:
            theta = theta + step
            logp = float(log_posterior(theta))
            continue

        # Minimal backtracking to enforce ascent (stability, not a change in target)
        alpha = 1.0
        for _bt in range(12):
            cand = theta + alpha * step
            cand_logp = float(log_posterior(cand))
            if np.isfinite(cand_logp) and cand_logp >= logp:
                theta = cand
                logp = cand_logp
                break
            alpha *= 0.5
        else:
            # If we cannot find an improving step, stop.
            break

    return theta, logp


# ----------------------------------------------------------------------
# Lu-style TMH independence proposal
# ----------------------------------------------------------------------


def tmh_step(
    state: TMHState,
    log_posterior,
    grad_log_posterior=None,
    hess_log_posterior=None,
    rng: np.random.Generator | None = None,
    *,
    kappa: float | None = None,
    ridge: float = 1e-6,
    newton_max_iter: int = 25,
    newton_tol: float = 1e-8,
) -> tuple[TMHState, bool]:
    """
    One Lu-style TMH update for a block.

    1) Find mode theta_hat given current "rest" (via Newton).
    2) Build Laplace proposal q = N(theta_hat, kappa^2 * V_hat),
       where V_hat^{-1} = -∇² log p(theta_hat | rest).
    3) Propose theta' ~ q and accept with independence MH.

    Parameters
    ----------
    state : TMHState
    log_posterior : callable(theta)->float
    grad_log_posterior : callable(theta)->(d,), optional
    hess_log_posterior : callable(theta)->(d,d), optional
    rng : np.random.Generator, optional
    kappa : float, optional
        If None, uses Lu default: 2.38 / sqrt(d).
    ridge : float
        Regularization for -Hessian at mode (and in Newton steps).
    newton_max_iter, newton_tol : Newton controls.

    Returns
    -------
    new_state : TMHState
    accepted : bool
    """
    theta = np.asarray(state.theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("state.theta must be 1D.")
    d = theta.size

    if rng is None:
        rng = np.random.default_rng()

    # Lu default scaling
    if kappa is None:
        kappa = 2.38 / np.sqrt(d)
    kappa = float(kappa)

    # Derivative providers: Lu expects analytic/structured derivatives.
    if grad_log_posterior is None or hess_log_posterior is None:
        # Fallback (not Lu-aligned)
        def grad_log_posterior(th):
            return compute_gradient(th, log_posterior, eps=1e-6)

        def hess_log_posterior(th):
            return compute_hessian(th, log_posterior, eps=1e-4)

    # 1) Mode for current conditional posterior
    theta_hat, _ = newton_find_mode(
        theta,
        log_posterior,
        grad_log_posterior,
        hess_log_posterior,
        ridge=ridge,
        max_iter=newton_max_iter,
        tol=newton_tol,
        backtrack=True,
    )

    # 2) Curvature at the mode (Laplace)
    hess_hat = np.asarray(hess_log_posterior(theta_hat), dtype=float)
    if hess_hat.shape != (d, d):
        raise ValueError("hess_log_posterior returned wrong shape at mode.")
    neg_hess_hat = -hess_hat
    H_reg = _regularize_neg_hessian(neg_hess_hat, ridge=ridge)
    V_hat = np.linalg.inv(H_reg)

    cov = (kappa**2) * V_hat

    cov_inv, logdet_cov = _cov_inv_and_logdet(cov)

    # Independence proposal from q(.)
    theta_prop = rng.multivariate_normal(theta_hat, cov)
    logp_prop = float(log_posterior(theta_prop))

    if not np.isfinite(logp_prop):
        return state, False

    # 3) Independence MH acceptance
    log_q_curr = _mvnorm_logpdf(theta, theta_hat, cov_inv, logdet_cov)
    log_q_prop = _mvnorm_logpdf(theta_prop, theta_hat, cov_inv, logdet_cov)

    log_alpha = (logp_prop - state.logp) + (log_q_curr - log_q_prop)

    if np.log(rng.uniform()) < log_alpha:
        return TMHState(theta=theta_prop, logp=logp_prop), True

    return state, False


# ----------------------------------------------------------------------
# Block wrapper for use in estimator code
# ----------------------------------------------------------------------


def tmh_update_block(
    get_block,
    set_block,
    full_log_posterior,
    rng: np.random.Generator,
    *,
    block_grad_log_posterior=None,
    block_hess_log_posterior=None,
    kappa: float | None = None,
    ridge: float = 1e-6,
    newton_max_iter: int = 25,
    newton_tol: float = 1e-8,
) -> bool:
    """
    Apply Lu-style TMH to a parameter block inside a larger parameter state.

    Expected calling pattern:
    - get_block() -> (theta_block, logp_current_full)
    - set_block(theta_block_new) mutates the full parameter state

    full_log_posterior() returns the full log posterior of the current full state.

    To be Lu-aligned, provide:
    - block_grad_log_posterior(theta_block): gradient of FULL log posterior wrt the block,
      evaluated at a full state where the block equals theta_block.
    - block_hess_log_posterior(theta_block): Hessian of FULL log posterior wrt the block.

    If derivatives are not provided, numerical differentiation will be used (fallback).

    Returns
    -------
    accepted : bool
    """
    theta_block, logp_current = get_block()
    theta_block = np.asarray(theta_block, dtype=float)
    if theta_block.ndim != 1:
        raise ValueError("get_block must return a 1D theta_block.")
    logp_current = float(logp_current)

    # Block-level log posterior wrapper: evaluates full posterior at a temporary block value.
    def block_logp(theta_block_new):
        old = theta_block.copy()
        set_block(theta_block_new)
        val = float(full_log_posterior())
        set_block(old)
        return val

    # Derivative wrappers (if provided, they must follow the same temporary-set pattern)
    if block_grad_log_posterior is not None:

        def block_grad(theta_block_new):
            old = theta_block.copy()
            set_block(theta_block_new)
            g = np.asarray(block_grad_log_posterior(theta_block_new), dtype=float)
            set_block(old)
            return g

    else:
        block_grad = None

    if block_hess_log_posterior is not None:

        def block_hess(theta_block_new):
            old = theta_block.copy()
            set_block(theta_block_new)
            H = np.asarray(block_hess_log_posterior(theta_block_new), dtype=float)
            set_block(old)
            return H

    else:
        block_hess = None

    state = TMHState(theta=theta_block, logp=logp_current)

    new_state, accepted = tmh_step(
        state,
        block_logp,
        grad_log_posterior=block_grad,
        hess_log_posterior=block_hess,
        rng=rng,
        kappa=kappa,
        ridge=ridge,
        newton_max_iter=newton_max_iter,
        newton_tol=newton_tol,
    )

    if accepted:
        set_block(new_state.theta)

    return accepted
