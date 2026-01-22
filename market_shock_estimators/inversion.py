"""
Berry (1994) market-share inversion (contraction mapping) for random-coefficients logit.

Purpose
-------
Recover mean utilities δ_jt from observed market shares, conditional on a fixed
heterogeneity parameter σ.

This file implements ONLY the inversion step used inside:
- standard BLP
- Lu–Shimizu

It does not estimate σ, β, or apply shrinkage.
"""

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------
# 1. Sanity check
# ---------------------------------------------------------------------
def check_market_inputs(s_obs: Array, s0: float, p: Array) -> None:
    """
    Validate inputs for a single market.

    Requirements:
      - s_obs and p are 1D arrays of equal length J
      - s_obs[j] > 0 for all j
      - s0 > 0
      - sum_j s_obs[j] + s0 == 1 (up to numerical tolerance)
      - all values finite

    Raises ValueError on failure.
    """
    s_obs = np.asarray(s_obs, dtype=float)
    p = np.asarray(p, dtype=float)

    if s_obs.ndim != 1 or p.ndim != 1:
        raise ValueError("s_obs and p must be 1D arrays.")
    if s_obs.shape[0] != p.shape[0]:
        raise ValueError("s_obs and p must have the same length.")

    if not np.all(np.isfinite(s_obs)):
        raise ValueError("s_obs contains NaN or inf.")
    if not np.all(np.isfinite(p)):
        raise ValueError("p contains NaN or inf.")
    if not np.isfinite(s0):
        raise ValueError("s0 contains NaN or inf.")

    if np.any(s_obs <= 0.0):
        raise ValueError("All inside-good shares must be strictly positive.")
    if s0 <= 0.0:
        raise ValueError("Outside share s0 must be strictly positive.")

    if not np.isclose(float(s_obs.sum()) + float(s0), 1.0, atol=1e-10):
        raise ValueError("Inside shares plus outside share must sum to 1.")


# ---------------------------------------------------------------------
# 2. Share simulation
# ---------------------------------------------------------------------
def simulate_shares(delta: Array, p: Array, sigma: float, v_draws: Array) -> Array:
    """
    Simulate market shares under random coefficients logit for one market.

    Utility for draw r:
        u_j(r) = δ_j + σ * v_r * p_j

    Outside option utility normalized to 0.

    Returns mean shares across draws.
    """
    delta = np.asarray(delta, dtype=float)
    p = np.asarray(p, dtype=float)
    v = np.asarray(v_draws, dtype=float)

    U = delta[None, :] + (sigma * v)[:, None] * p[None, :]

    m = U.max(axis=1, keepdims=True)  # (R,1)
    U = U - m  # stabilize inside utilities

    expU = np.exp(U)
    denom = np.exp(-m[:, 0]) + expU.sum(axis=1)  # outside term must be exp(0 - m)

    return (expU / denom[:, None]).mean(axis=0)


# ---------------------------------------------------------------------
# 3. Single-market inversion
# ---------------------------------------------------------------------
def invert_market(
    s_obs: Array,
    s0: float,
    p: Array,
    sigma: float,
    v_draws: Array,
    delta_init: Array | None = None,
    *,
    damping: float = 1.0,
    tol: float = 1e-8,
    share_tol=1e-10,
    max_iter: int = 10_000,
) -> tuple[Array, int]:
    """
    Recover δ for one market using Berry contraction.

    Initialization:
        δ^(0) = log(s_obs) - log(s0)
        (or use delta_init if provided)

    Update:
        δ_new = δ + damping * (log(s_obs) - log(s_hat))

    Returns:
      - (delta, iterations)

    Raises RuntimeError if convergence fails.
    """
    check_market_inputs(s_obs, s0, p)

    s_obs = np.asarray(s_obs, dtype=float)
    p = np.asarray(p, dtype=float)
    s0 = float(s0)
    eps = np.finfo(float).tiny

    if not (0.0 < float(damping) <= 1.0):
        raise ValueError("damping must be in (0, 1].")
    damping = float(damping)

    if delta_init is not None:
        delta = np.asarray(delta_init, dtype=float)
        if delta.ndim != 1 or delta.shape[0] != s_obs.shape[0]:
            raise ValueError("delta_init must be a 1D array of length J.")
        if not np.all(np.isfinite(delta)):
            raise ValueError("delta_init contains NaN or inf.")
    else:
        delta = np.log(s_obs) - np.log(s0)

    for it in range(1, max_iter + 1):
        s_hat = simulate_shares(delta, p, sigma, v_draws)

        if np.any(s_hat <= 0.0) or not np.all(np.isfinite(s_hat)):
            raise RuntimeError("Predicted shares became invalid during inversion.")

        if np.max(np.abs(s_hat - s_obs)) < share_tol:
            return delta, it

        s_hat = np.maximum(s_hat, eps)
        delta_new = delta + damping * (np.log(s_obs) - np.log(s_hat))

        if np.max(np.abs(delta_new - delta)) < tol:
            return delta_new, it

        delta = delta_new

    raise RuntimeError("Berry inversion failed to converge.")


# ---------------------------------------------------------------------
# 4. Multi-market inversion
# ---------------------------------------------------------------------
def invert_all_markets(
    sjt: Array,
    s0t: Array,
    pjt: Array,
    sigma: float,
    v_draws: Array,
    delta_init: Array | None = None,
    *,
    damping: float = 1.0,
    tol: float = 1e-8,
    share_tol: float = 1e-10,
    max_iter: int = 10_000,
) -> Array:
    """
    Invert all markets sequentially.

    Inputs:
      - sjt, pjt: arrays of shape (T, J)
      - s0t: outside shares, shape (T,)
      - sigma: scalar
      - v_draws: simulation draws
      - delta_init: optional warm start array of shape (T, J)
      - damping: contraction damping in (0, 1]

    Returns:
      - delta array of shape (T, J)
    """
    sjt = np.asarray(sjt, dtype=float)
    pjt = np.asarray(pjt, dtype=float)
    s0t = np.asarray(s0t, dtype=float)

    if sjt.shape != pjt.shape:
        raise ValueError("sjt and pjt must have the same shape.")
    if sjt.ndim != 2:
        raise ValueError("sjt and pjt must be 2D arrays of shape (T, J).")
    if s0t.ndim != 1:
        raise ValueError("s0t must be a 1D array of length T.")
    if sjt.shape[0] != s0t.shape[0]:
        raise ValueError("s0t must have length T equal to sjt.shape[0].")

    T, J = sjt.shape
    delta = np.empty((T, J), dtype=float)

    if delta_init is not None:
        delta_init = np.asarray(delta_init, dtype=float)
        if delta_init.shape != (T, J):
            raise ValueError("delta_init must have shape (T, J) matching sjt.")
        if not np.all(np.isfinite(delta_init)):
            raise ValueError("delta_init contains NaN or inf.")

    for t in range(T):

        delta_t, iters = invert_market(
            s_obs=sjt[t],
            s0=float(s0t[t]),
            p=pjt[t],
            sigma=sigma,
            v_draws=v_draws,
            delta_init=None if delta_init is None else delta_init[t],
            damping=damping,
            tol=tol,
            share_tol=share_tol,
            max_iter=max_iter,
        )

        delta[t] = delta_t

    return delta
