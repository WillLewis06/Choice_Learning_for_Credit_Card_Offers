# ching_conftest.py
"""
Deterministic test builders for Phase-3 stockpiling (Ching-style).

This module is imported directly by unit tests (not as pytest fixtures), so it
provides plain Python functions that construct small, valid inputs consistent
with the current core implementation:

- Uses s_mjt (price-state path), not p_state_mjt.
- Provides lambda_mn as a fixed input (passed, not estimated).
- No eps anywhere.
- sigmas schema: {z_beta,z_alpha,z_v,z_fc,z_u_scale}.
- init_theta schema: {beta,alpha,v,fc,u_scale} with shapes:
    beta: scalar, alpha/v/fc: (J,), u_scale: (M,)
- k schema: {beta,alpha,v,fc,u_scale}
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Reduce TF C++ logging (tests import this before importing TensorFlow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# =============================================================================
# Core tiny dimensions / hyperparameters
# =============================================================================


def tiny_dims() -> dict[str, int]:
    """
    Return a small deterministic dimension dict used across all tests.

    Chosen to satisfy common test assumptions:
      - M=2, J=2, S=2, I_max=2
    """
    return {
        "M": 2,
        "N": 3,
        "J": 2,
        "T": 6,
        "S": 2,
        "I_max": 2,
    }


def tiny_dp_config() -> dict[str, Any]:
    """
    DP / filter controls used by solve_ccp_buy and the likelihood.

    Note:
      - No 'eps' (removed in the updated core).
      - tol is a python float, max_iter a python int (posterior expects these types).
    """
    return {
        "waste_cost": 0.25,
        "tol": 1e-10,
        "max_iter": 200,
    }


# =============================================================================
# Price process builders
# =============================================================================


def price_process(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Build a valid (row-stochastic) Markov chain and strictly-positive prices.

    Returns:
      P_price_mj:    (M,J,S,S) float64
      price_vals_mj: (M,J,S)   float64
    """
    M = int(dims["M"])
    J = int(dims["J"])
    S = int(dims["S"])

    p_stay = 0.85
    P = np.zeros((M, J, S, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            for s in range(S):
                P[m, j, s, :] = (1.0 - p_stay) / float(S - 1)
                P[m, j, s, s] = p_stay

    # Positive prices with a "high" and "low" state (for S>=2)
    price_vals = np.zeros((M, J, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            base = 2.0 + 0.15 * float(m) + 0.25 * float(j)
            for s in range(S):
                # decreasing in s gives variation while keeping positive
                price_vals[m, j, s] = base * (0.85 ** float(s))

    return {"P_price_mj": P, "price_vals_mj": price_vals}


def simulate_s_mjt_from_P(
    P_price_mj: np.ndarray,
    T: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate s_mjt given P_price_mj.

    Returns:
      s_mjt: (M,J,T) int32 with states in {0,...,S-1}
    """
    P = np.asarray(P_price_mj, dtype=np.float64)
    M, J, S, _ = P.shape
    T = int(T)

    rng = np.random.default_rng(int(seed))
    s_mjt = np.zeros((M, J, T), dtype=np.int32)

    # Initialize at 0 for determinism
    s_mjt[:, :, 0] = 0
    for t in range(1, T):
        for m in range(M):
            for j in range(J):
                s_prev = int(s_mjt[m, j, t - 1])
                s_next = rng.choice(S, p=P[m, j, s_prev, :])
                s_mjt[m, j, t] = int(s_next)

    return s_mjt


# =============================================================================
# Panel + fixed inputs
# =============================================================================


def panel_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Build observed panel components needed by estimator/posterior.

    Returns:
      a_mnjt: (M,N,J,T) int32 in {0,1}
      s_mjt:  (M,J,T)   int32 in {0,...,S-1}
    """
    M = int(dims["M"])
    N = int(dims["N"])
    J = int(dims["J"])
    T = int(dims["T"])
    S = int(dims["S"])

    # Deterministic but non-degenerate actions
    rng = np.random.default_rng(123)
    a = (rng.random((M, N, J, T)) < 0.25).astype(np.int32)

    # Deterministic state pattern (guaranteed in-range)
    s = np.zeros((M, J, T), dtype=np.int32)
    for m in range(M):
        for j in range(J):
            for t in range(T):
                s[m, j, t] = int((m + j + t) % S)

    return {"a_mnjt": a, "s_mjt": s}


def u_mj_np(dims: dict[str, int]) -> np.ndarray:
    """Phase-1/2 intercept passed into Phase-3. Shape (M,J), float64."""
    M = int(dims["M"])
    J = int(dims["J"])
    u = np.zeros((M, J), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            u[m, j] = 0.4 + 0.1 * float(m) + 0.05 * float(j)
    return u


def lambda_mn_np(dims: dict[str, int]) -> np.ndarray:
    """Consumption probabilities. Shape (M,N), float64, entries strictly in (0,1)."""
    M = int(dims["M"])
    N = int(dims["N"])
    lam = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            lam[m, n] = 0.2 + 0.1 * float(m) + 0.05 * float(n)
    # Clip away from 0/1 to satisfy strict (0,1) requirement
    lam = np.clip(lam, 1e-3, 1.0 - 1e-3)
    return lam


def pi_I0_uniform(dims: dict[str, int]) -> np.ndarray:
    """Uniform initial inventory distribution. Shape (I_max+1,), float64, sums to 1."""
    I_max = int(dims["I_max"])
    I = I_max + 1
    pi = np.ones((I,), dtype=np.float64) / float(I)
    return pi


# =============================================================================
# Estimation-related builders (sigmas, init_theta, k, z-blocks)
# =============================================================================


def sigmas() -> dict[str, float]:
    """
    Prior scales for z-blocks (estimator-only).

    Strict schema (no extras):
      {z_beta,z_alpha,z_v,z_fc,z_u_scale}
    """
    return {
        "z_alpha": 0.5,
        "z_beta": 0.5,
        "z_fc": 0.5,
        "z_u_scale": 0.5,
        "z_v": 0.5,
    }


def init_theta_np(dims: dict[str, int]) -> dict[str, Any]:
    """
    Constrained initial theta for StockpilingEstimator.fit(init_theta=...).

    Strict schema (no extras):
      beta: scalar in (0,1)
      alpha/v/fc: (J,) > 0
      u_scale: (M,) > 0
    """
    M = int(dims["M"])
    J = int(dims["J"])

    return {
        "beta": 0.85,
        "alpha": np.full((J,), 1.2, dtype=np.float64),
        "v": np.full((J,), 0.9, dtype=np.float64),
        "fc": np.full((J,), 0.4, dtype=np.float64),
        "u_scale": np.full((M,), 1.0, dtype=np.float64),
    }


def k_proposals() -> dict[str, float]:
    """
    Random-walk proposal step sizes for StockpilingEstimator.fit(n_iter, k).

    Strict schema (no extras): {beta,alpha,v,fc,u_scale}
    """
    return {
        "alpha": 0.08,
        "beta": 0.05,
        "fc": 0.08,
        "u_scale": 0.05,
        "v": 0.08,
    }


def z_blocks_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Unconstrained z-blocks at the prior mode (all zeros), matching the current model
    transform in stockpiling_model.unconstrained_to_theta.

    Shapes:
      z_beta:    scalar ()
      z_alpha:   (J,)
      z_v:       (J,)
      z_fc:      (J,)
      z_u_scale: (M,)
    """
    M = int(dims["M"])
    J = int(dims["J"])
    return {
        "z_beta": np.asarray(0.0, dtype=np.float64),
        "z_alpha": np.zeros((J,), dtype=np.float64),
        "z_v": np.zeros((J,), dtype=np.float64),
        "z_fc": np.zeros((J,), dtype=np.float64),
        "z_u_scale": np.zeros((M,), dtype=np.float64),
    }


# =============================================================================
# Optional convenience helpers used by updated tests
# =============================================================================


def estimator_init_inputs_np(rng_seed: int) -> dict[str, Any]:
    """
    Build a complete, validation-ready input bundle for StockpilingEstimator.__init__.

    Returns a dict with keys matching the constructor signature.
    """
    dims = tiny_dims()
    dp = tiny_dp_config()
    panel = panel_np(dims)
    price = price_process(dims)

    return {
        "a_mnjt": panel["a_mnjt"],
        "s_mjt": panel["s_mjt"],
        "u_mj": u_mj_np(dims),
        "price_vals_mj": price["price_vals_mj"],
        "P_price_mj": price["P_price_mj"],
        "lambda_mn": lambda_mn_np(dims),
        "I_max": int(dims["I_max"]),
        "pi_I0": pi_I0_uniform(dims),
        "waste_cost": float(dp["waste_cost"]),
        "tol": float(dp["tol"]),
        "max_iter": int(dp["max_iter"]),
        "sigmas": sigmas(),
        "rng_seed": int(rng_seed),
    }


def inventory_maps_tf(I_max: int):
    """
    Build inventory maps as TF tensors using the core implementation.

    This is only used by posterior tests that require `inventory_maps` in inputs.
    """
    import tensorflow as tf  # local import (keeps module import light)
    from ching.stockpiling_model import build_inventory_maps

    return build_inventory_maps(tf.convert_to_tensor(int(I_max), dtype=tf.int32))
