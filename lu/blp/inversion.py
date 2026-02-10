"""
Berry (1994) market-share inversion (contraction mapping) for RC logit.

This module implements the single-purpose inversion step used in BLP-style
estimators: recover mean utilities `delta` that rationalize observed market
shares given a fixed heterogeneity parameter `sigma`.

Scope:
  - Invert shares -> delta for one market or many markets.
  - Simulate predicted shares under a price-only random coefficient.
  - Validate basic per-market inputs.

Not in scope:
  - Estimating sigma or linear parameters (e.g. beta).
  - GMM moments, IV, or any shrinkage logic.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


# -----------------------------------------------------------------------------
# 1) Validation (single market)
# -----------------------------------------------------------------------------


def check_market_inputs(s_obs: Array, s0: float, p: Array) -> None:
    """Validate inputs for a single-market inversion.

    Required structure:
      - s_obs and p are 1D arrays of equal length J.
      - s_obs[j] > 0 for all j and s0 > 0.
      - sum(s_obs) + s0 == 1 (up to tolerance).
      - all values are finite.

    Raises:
        ValueError: If any requirement is violated.
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

    # A tight tolerance is appropriate here because these are computed shares.
    if not np.isclose(float(s_obs.sum()) + float(s0), 1.0, atol=1e-10):
        raise ValueError("Inside shares plus outside share must sum to 1.")


# -----------------------------------------------------------------------------
# 2) Predicted shares (single market)
# -----------------------------------------------------------------------------


def simulate_shares(delta: Array, p: Array, sigma: float, v_draws: Array) -> Array:
    """Simulate inside-good shares for one market under RC logit.

    Model (price-only random coefficient):
      utility for draw r:
        u_j(r) = delta[j] + sigma * v_draws[r] * p[j]
      outside option utility is 0.

    Computation:
      - compute draw-specific logit shares
      - return the mean share across simulation draws

    Args:
        delta: Mean utilities, shape (J,).
        p: Prices, shape (J,).
        sigma: Heterogeneity scale on price, scalar.
        v_draws: Simulation draws, shape (R,).

    Returns:
        s_hat: Predicted inside-good shares, shape (J,).
    """
    delta = np.asarray(delta, dtype=float)
    p = np.asarray(p, dtype=float)
    v = np.asarray(v_draws, dtype=float)

    # Utilities per draw: U[r, j] = delta[j] + (sigma * v[r]) * p[j].
    U = delta[None, :] + (sigma * v)[:, None] * p[None, :]

    # Numerically-stable logit: shift utilities by per-draw max.
    m = U.max(axis=1, keepdims=True)  # (R, 1)
    U = U - m

    expU = np.exp(U)  # (R, J)

    # Outside option has utility 0, so its term is exp(0 - m[r]).
    denom = np.exp(-m[:, 0]) + expU.sum(axis=1)  # (R,)

    # Average shares across draws.
    return (expU / denom[:, None]).mean(axis=0)


# -----------------------------------------------------------------------------
# 3) Berry contraction (single market)
# -----------------------------------------------------------------------------


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
    share_tol: float = 1e-10,
    max_iter: int = 10_000,
) -> tuple[Array, int]:
    """Invert one market's observed shares to recover `delta`.

    This implements the Berry contraction mapping. For a fixed (sigma, v_draws),
    define s_hat(delta) as the predicted shares from `simulate_shares`.

    Initialization:
      - default: delta = log(s_obs) - log(s0)
      - or use delta_init if provided (warm start)

    Update:
      delta_new = delta + damping * (log(s_obs) - log(s_hat))

    Stopping criteria (either condition):
      - max |s_hat - s_obs| < share_tol
      - max |delta_new - delta| < tol

    Args:
        s_obs: Observed inside-good shares, shape (J,).
        s0: Observed outside share, scalar.
        p: Prices, shape (J,).
        sigma: Heterogeneity scale, scalar.
        v_draws: Simulation draws, shape (R,).
        delta_init: Optional initial delta, shape (J,).
        damping: Step size in (0, 1], used for stability when needed.
        tol: Stopping tolerance on delta changes.
        share_tol: Stopping tolerance on share differences.
        max_iter: Maximum contraction iterations.

    Returns:
        delta: Recovered mean utilities, shape (J,).
        iterations: Number of contraction iterations used.

    Raises:
        ValueError: On invalid inputs.
        RuntimeError: If predicted shares become invalid or convergence fails.
    """
    check_market_inputs(s_obs, s0, p)

    s_obs = np.asarray(s_obs, dtype=float)
    p = np.asarray(p, dtype=float)
    s0 = float(s0)
    eps = np.finfo(float).tiny

    if not (0.0 < float(damping) <= 1.0):
        raise ValueError("damping must be in (0, 1].")
    damping = float(damping)

    # Choose the initial delta.
    if delta_init is not None:
        delta = np.asarray(delta_init, dtype=float)
        if delta.ndim != 1 or delta.shape[0] != s_obs.shape[0]:
            raise ValueError("delta_init must be a 1D array of length J.")
        if not np.all(np.isfinite(delta)):
            raise ValueError("delta_init contains NaN or inf.")
    else:
        # Standard logit inversion starting point.
        delta = np.log(s_obs) - np.log(s0)

    # Contraction iterations.
    for it in range(1, max_iter + 1):
        s_hat = simulate_shares(delta, p, sigma, v_draws)

        # Shares must remain positive and finite for logs below.
        if np.any(s_hat <= 0.0) or not np.all(np.isfinite(s_hat)):
            raise RuntimeError("Predicted shares became invalid during inversion.")

        # Optional early stop on share-space difference.
        if np.max(np.abs(s_hat - s_obs)) < share_tol:
            return delta, it

        # Guard against log(0) in the update.
        s_hat = np.maximum(s_hat, eps)
        delta_new = delta + damping * (np.log(s_obs) - np.log(s_hat))

        # Standard stop on delta-space difference.
        if np.max(np.abs(delta_new - delta)) < tol:
            return delta_new, it

        delta = delta_new

    raise RuntimeError("Berry inversion failed to converge.")


# -----------------------------------------------------------------------------
# 4) Multi-market wrapper
# -----------------------------------------------------------------------------


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
    """Invert all markets sequentially.

    This is a convenience wrapper around `invert_market`. It loops over markets
    t=0..T-1 and returns a (T, J) delta array.

    Args:
        sjt: Observed inside shares, shape (T, J).
        s0t: Observed outside shares, shape (T,).
        pjt: Prices, shape (T, J).
        sigma: Heterogeneity scale, scalar.
        v_draws: Simulation draws, shape (R,).
        delta_init: Optional warm start, shape (T, J).
        damping: Contraction damping in (0, 1].
        tol: Stopping tolerance on delta changes (per market).
        share_tol: Stopping tolerance on share differences (per market).
        max_iter: Max contraction iterations (per market).

    Returns:
        delta: Mean utilities, shape (T, J).

    Raises:
        ValueError: If array shapes are inconsistent.
        RuntimeError: If any market inversion fails to converge.
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
        # Invert each market independently. Warm starts can be passed market-by-market.
        delta_t, _ = invert_market(
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
