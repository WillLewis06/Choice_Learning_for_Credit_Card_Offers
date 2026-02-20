"""
Berry (1994) contraction mapping for random-coefficients logit (price-only RC).

Numerical routines only. Configuration validation and data integrity checks are
handled upstream (e.g., config/input validation modules).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def logit_delta_init(s_obs: Array, s0: float) -> Array:
    """Compute the standard logit starting value: log(s_obs) - log(s0).

    Args:
        s_obs: Inside-good shares, shape (J,).
        s0: Outside share, scalar.

    Returns:
        delta_init: Initial mean utilities, shape (J,).

    Raises:
        ValueError: If logs are undefined (non-positive shares).
    """
    s_obs = np.asarray(s_obs)
    s0 = float(s0)

    delta_init = np.log(s_obs) - np.log(s0)
    if not np.all(np.isfinite(delta_init)):
        raise ValueError("logit_delta_init requires strictly positive s_obs and s0.")
    return delta_init


def simulate_shares(delta: Array, p: Array, sigma: float, v_draws: Array) -> Array:
    """Simulate inside-good shares for one market under price-only RC logit.

    Utility for simulation draw r:
        u_j(r) = delta[j] + sigma * v_draws[r] * p[j]
    Outside option utility is normalized to 0.

    Args:
        delta: Mean utilities, shape (J,).
        p: Prices, shape (J,).
        sigma: Heterogeneity scale on price, scalar.
        v_draws: Simulation draws, shape (R,).

    Returns:
        s_hat: Predicted inside-good shares, shape (J,).
    """
    delta = np.asarray(delta)
    p = np.asarray(p)
    v = np.asarray(v_draws).reshape(-1)

    # Draw-specific utilities: U[r, j] = delta[j] + (sigma * v[r]) * p[j]
    U = delta[None, :] + (sigma * v)[:, None] * p[None, :]

    # Numerically-stable softmax with explicit outside option:
    # shift by max(0, max_j U[r, j]) so both inside and outside terms are stable.
    m = np.maximum(0.0, U.max(axis=1, keepdims=True))  # shape (R, 1)
    expU = np.exp(U - m)  # shape (R, J)

    # Outside option term is exp(0 - m[r]).
    denom = np.exp(-m[:, 0]) + expU.sum(axis=1)  # shape (R,)

    # Mean share across simulation draws.
    return (expU / denom[:, None]).mean(axis=0)


def invert_market(
    s_obs: Array,
    p: Array,
    sigma: float,
    v_draws: Array,
    delta_init: Array,
    damping: float,
    tol: float,
    share_tol: float,
    max_iter: int,
) -> tuple[Array, int]:
    """Invert one market's observed shares to recover mean utilities `delta`.

    Berry contraction mapping for fixed (sigma, v_draws):
        delta_{k+1} = delta_k + damping * (log(s_obs) - log(s_hat(delta_k)))

    Args:
        s_obs: Observed inside-good shares, shape (J,).
        p: Prices, shape (J,).
        sigma: Heterogeneity scale on price, scalar.
        v_draws: Simulation draws, shape (R,).
        delta_init: Initial delta (provided explicitly), shape (J,).
        damping: Contraction step size (validated upstream).
        tol: Stopping tolerance on delta changes (validated upstream).
        share_tol: Stopping tolerance on share differences (validated upstream).
        max_iter: Maximum contraction iterations (validated upstream).

    Returns:
        delta: Recovered mean utilities, shape (J,).
        iterations: Number of iterations performed.

    Raises:
        ValueError: If shapes are inconsistent or logs are undefined.
        RuntimeError: If the contraction does not converge or produces invalid shares.
    """
    s_obs = np.asarray(s_obs)
    p = np.asarray(p)
    delta = np.asarray(delta_init)

    if s_obs.ndim != 1 or p.ndim != 1 or delta.ndim != 1:
        raise ValueError("s_obs, p, and delta_init must be 1D arrays.")
    if not (s_obs.shape[0] == p.shape[0] == delta.shape[0]):
        raise ValueError("s_obs, p, and delta_init must have the same length J.")

    # Precompute log(s_obs) once for the fixed-point updates.
    log_s_obs = np.log(s_obs)
    if not np.all(np.isfinite(log_s_obs)):
        raise ValueError("invert_market requires strictly positive s_obs.")

    for it in range(1, max_iter + 1):
        # Predicted shares under the current delta.
        s_hat = simulate_shares(delta, p, sigma, v_draws)

        # Contraction update uses logs; shares must be strictly positive and finite.
        if np.any(s_hat <= 0.0) or not np.all(np.isfinite(s_hat)):
            raise RuntimeError("Predicted shares became invalid during inversion.")

        # Early stopping in share space.
        if np.max(np.abs(s_hat - s_obs)) < share_tol:
            return delta, it

        # Fixed-point update in delta space.
        delta_new = delta + damping * (log_s_obs - np.log(s_hat))

        # Stopping criterion in delta space.
        if np.max(np.abs(delta_new - delta)) < tol:
            return delta_new, it

        delta = delta_new

    raise RuntimeError("Berry inversion failed to converge within max_iter.")


def invert_all_markets(
    sjt: Array,
    pjt: Array,
    sigma: float,
    v_draws: Array,
    delta_init: Array,
    damping: float,
    tol: float,
    share_tol: float,
    max_iter: int,
) -> Array:
    """Invert a panel of markets sequentially.

    Args:
        sjt: Observed inside shares, shape (T, J).
        pjt: Prices, shape (T, J).
        sigma: Heterogeneity scale on price, scalar.
        v_draws: Simulation draws, shape (R,).
        delta_init: Initial deltas for each market, shape (T, J).
        damping: Contraction step size (validated upstream).
        tol: Stopping tolerance on delta changes (validated upstream).
        share_tol: Stopping tolerance on share differences (validated upstream).
        max_iter: Maximum contraction iterations per market (validated upstream).

    Returns:
        delta: Recovered mean utilities, shape (T, J).

    Raises:
        ValueError: If shapes are inconsistent.
        RuntimeError: If any market inversion fails to converge.
    """
    sjt = np.asarray(sjt)
    pjt = np.asarray(pjt)
    delta_init = np.asarray(delta_init)

    if sjt.ndim != 2 or pjt.ndim != 2 or delta_init.ndim != 2:
        raise ValueError("sjt, pjt, and delta_init must be 2D arrays of shape (T, J).")
    if sjt.shape != pjt.shape or sjt.shape != delta_init.shape:
        raise ValueError(
            "sjt, pjt, and delta_init must all have the same shape (T, J)."
        )

    T, J = sjt.shape
    delta = np.empty((T, J), dtype=delta_init.dtype)

    for t in range(T):
        # Each market is inverted independently using its own warm start.
        delta_t, _ = invert_market(
            sjt[t],
            pjt[t],
            sigma,
            v_draws,
            delta_init[t],
            damping,
            tol,
            share_tol,
            max_iter,
        )
        delta[t] = delta_t

    return delta
