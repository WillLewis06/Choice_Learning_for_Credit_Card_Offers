# tmh.py
"""
Tailored Metropolis–Hastings (TMH) aligned with Lu (2025), Section 4.

Lu-style TMH:
1) Find the conditional posterior mode for a parameter block via Newton steps.
2) Form a Laplace (Gaussian) approximation at the mode.
3) Propose from an independence Gaussian centered at the mode:
       theta' ~ N(theta_hat, kappa^2 * V_hat),
   where V_hat^{-1} = -∇^2 log p(theta | rest) evaluated at theta_hat.
4) Accept/reject with an independence-MH ratio using the same proposal density.

This module is algorithmic only. It does not know the model.

To be Lu-aligned and efficient, provide analytic derivatives:
- grad_log_posterior(theta): (d,) array
- hess_log_posterior(theta): (d,d) array

Numerical differentiation exists as a fallback but is not Lu-aligned.
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
    """Return (neg_hess + ridge * I), where neg_hess is intended as -∇² log p."""
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
# Derivative selection (analytic preferred; fallback optional)
# ----------------------------------------------------------------------


def _select_derivatives(
    log_posterior,
    grad_log_posterior,
    hess_log_posterior,
    *,
    allow_fallback: bool,
) -> tuple[callable, callable]:
    """
    Return (grad_fn, hess_fn). Prefer supplied derivatives.
    If allow_fallback=True, fill missing pieces via numerical differentiation.
    """
    grad_fn = grad_log_posterior
    hess_fn = hess_log_posterior

    if grad_fn is None:
        if not allow_fallback:
            raise ValueError(
                "grad_log_posterior is required when allow_fallback=False."
            )

        def grad_fn(th):
            return compute_gradient(th, log_posterior, eps=1e-6)

    if hess_fn is None:
        if not allow_fallback:
            raise ValueError(
                "hess_log_posterior is required when allow_fallback=False."
            )

        def hess_fn(th):
            return compute_hessian(th, log_posterior, eps=1e-4)

    return grad_fn, hess_fn


# ----------------------------------------------------------------------
# Lu-style mode finding (Newton)
# ----------------------------------------------------------------------


def newton_find_mode(
    theta_init: np.ndarray,
    log_posterior,
    grad_log_posterior,
    hess_log_posterior,
    *,
    ridge: float = 1e-2,
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
    """
    theta = np.asarray(theta_init, dtype=float).copy()
    if theta.ndim != 1:
        raise ValueError("theta_init must be 1D.")

    logp = float(log_posterior(theta))

    for _ in range(int(max_iter)):
        # print("[tmh] newton iteration {_}")
        g = np.asarray(grad_log_posterior(theta), dtype=float)
        if g.shape != theta.shape:
            raise ValueError("grad_log_posterior returned wrong shape.")
        if np.max(np.abs(g)) <= tol:
            break

        hess = np.asarray(hess_log_posterior(theta), dtype=float)
        if hess.shape != (theta.size, theta.size):
            raise ValueError("hess_log_posterior returned wrong shape.")

        H_reg = _regularize_neg_hessian(-hess, ridge=ridge)
        step = np.linalg.solve(H_reg, g)

        if not backtrack:
            theta = theta + step
            logp = float(log_posterior(theta))
            continue

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
    allow_fallback: bool = True,
) -> tuple[TMHState, bool]:
    """
    One Lu-style TMH update for a block.

    If analytic derivatives are supplied they are used directly.
    If allow_fallback=True, missing derivatives are filled using numerical
    differentiation (not Lu-aligned, but useful for debugging).
    """
    theta = np.asarray(state.theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("state.theta must be 1D.")
    d = theta.size

    if rng is None:
        rng = np.random.default_rng()

    if kappa is None:
        kappa = 2.38 / np.sqrt(d)
    kappa = float(kappa)

    grad_fn, hess_fn = _select_derivatives(
        log_posterior,
        grad_log_posterior,
        hess_log_posterior,
        allow_fallback=allow_fallback,
    )

    # 1) Mode for current conditional posterior
    theta_hat, _ = newton_find_mode(
        theta,
        log_posterior,
        grad_fn,
        hess_fn,
        ridge=ridge,
        max_iter=newton_max_iter,
        tol=newton_tol,
        backtrack=True,
    )

    # 2) Curvature at the mode (Laplace)
    hess_hat = np.asarray(hess_fn(theta_hat), dtype=float)
    if hess_hat.shape != (d, d):
        raise ValueError("hess_log_posterior returned wrong shape at mode.")
    H_reg = _regularize_neg_hessian(-hess_hat, ridge=ridge)
    V_hat = np.linalg.inv(H_reg)

    cov = (kappa**2) * V_hat
    cov_inv, logdet_cov = _cov_inv_and_logdet(cov)

    # Independence proposal
    theta_prop = rng.multivariate_normal(theta_hat, cov)
    logp_prop = float(log_posterior(theta_prop))

    if not np.isfinite(logp_prop):
        print("[TMH] reject: non-finite logp_prop")
        return state, False

    # 3) Independence MH acceptance
    log_q_curr = _mvnorm_logpdf(theta, theta_hat, cov_inv, logdet_cov)
    log_q_prop = _mvnorm_logpdf(theta_prop, theta_hat, cov_inv, logdet_cov)

    log_alpha = (logp_prop - state.logp) + (log_q_curr - log_q_prop)

    if np.log(rng.uniform()) < log_alpha:
        return TMHState(theta=theta_prop, logp=logp_prop), True

    print(f"[TMH] reject: log_alpha={log_alpha:.3e}")
    return state, False
