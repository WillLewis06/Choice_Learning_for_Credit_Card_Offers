# ching_conftest.py
"""
Shared Phase-3 (Ching-style stockpiling) test builders.

This file is intentionally not a pytest conftest module and does not define pytest
fixtures. Tests should import and call these helpers directly.

Key conventions (multi-product):
  a_mnjt:      (M, N, J, T) int32 actions in {0,1}
  p_state_mjt: (M, J, T)    int32 price-state indices in {0,...,S-1}
  ccp_buy:     (M, N, J, S, I) where I = I_max + 1
  u_mj:        (M, J) float64 fixed intercepts from Phase 1–2
  price_vals_mj: (M, J, S) float64 price levels by state
  P_price_mj:    (M, J, S, S) float64 row-stochastic transitions
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Must be set before importing TensorFlow to suppress most C++ logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402


def _quiet_tf_logger() -> None:
    """Reduce TensorFlow python logging (separate from TF_CPP_MIN_LOG_LEVEL)."""
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        return


_quiet_tf_logger()


def tiny_dims() -> dict[str, int]:
    """
    Small dimensions used across Phase-3 tests.

    Returns:
      dict with keys: M, N, J, T, S, I_max
    """
    return {"M": 2, "N": 3, "J": 2, "T": 5, "S": 2, "I_max": 2}


def tiny_dp_config() -> dict[str, Any]:
    """DP / filtering configuration used in posterior and estimator tests."""
    return {
        "waste_cost": 0.1,
        "eps": 1.0e-12,
        "tol": 1.0e-8,
        "max_iter": 200,
    }


def price_process(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Prices by state and Markov transition matrix for price states (market×product specific).

    For S=2:
      base price levels: [1.0, 0.8]
      base transition:   [[0.9,0.1],[0.2,0.8]]

    Returns:
      {
        "price_vals_mj": (M,J,S),
        "P_price_mj":    (M,J,S,S),
      }
    """
    M, J, S = int(dims["M"]), int(dims["J"]), int(dims["S"])
    if S != 2:
        raise ValueError(
            "price_process currently assumes S=2; update if S>2 is needed."
        )

    base_price = np.asarray([1.0, 0.8], dtype=np.float64)  # (S,)
    base_P = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)  # (S,S)

    price_vals_mj = np.empty((M, J, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            price_vals_mj[m, j, :] = base_price

    P_price_mj = np.empty((M, J, S, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            P_price_mj[m, j, :, :] = base_P

    return {"price_vals_mj": price_vals_mj, "P_price_mj": P_price_mj}


def pi_I0_uniform(dims: dict[str, int]) -> np.ndarray:
    """Uniform initial inventory belief pi_I0 over {0,...,I_max}."""
    i = int(dims["I_max"]) + 1
    return (np.ones(i, dtype=np.float64) / float(i)).astype(np.float64)


# Optional alias with no capitals in the name.
pi_i0_uniform = pi_I0_uniform


def panel_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Deterministic, non-degenerate panel (multi-product).

    - p_state_mjt cycles through states for each (m,j) (ensures both states appear).
    - a_mnjt uses a deterministic pattern across (m,n,j,t) containing both 0 and 1.

    Returns:
      {"a_mnjt": (M,N,J,T) int32, "p_state_mjt": (M,J,T) int32}
    """
    M, N, J, T, S = (
        int(dims["M"]),
        int(dims["N"]),
        int(dims["J"]),
        int(dims["T"]),
        int(dims["S"]),
    )

    p_state_mjt = np.empty((M, J, T), dtype=np.int32)
    for m in range(M):
        for j in range(J):
            p_state_mjt[m, j, :] = (np.arange(T, dtype=np.int32) + m + 3 * j) % S

    a_mnjt = np.empty((M, N, J, T), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            for j in range(J):
                a_mnjt[m, n, j, :] = (
                    (m * 31 + n * 17 + j * 13 + np.arange(T, dtype=np.int32)) % 2
                ).astype(np.int32)

    return {"a_mnjt": a_mnjt, "p_state_mjt": p_state_mjt}


def u_mj_np(dims: dict[str, int]) -> np.ndarray:
    """Fixed intercepts u_mj (M,J) used as known inputs in Phase-3."""
    M, J = int(dims["M"]), int(dims["J"])
    u_market = np.linspace(0.5, 1.5, M, dtype=np.float64)[:, None]  # (M,1)
    u_prod = np.linspace(-0.2, 0.2, J, dtype=np.float64)[None, :]  # (1,J)
    return (u_market + u_prod).astype(np.float64)


def sigmas() -> dict[str, float]:
    """
    Prior scales for z-blocks (must include all keys used by posterior/estimator).

    Note: input validation currently requires {z_beta,z_alpha,z_v,z_fc,z_lambda}.
    We also include z_u_scale because the estimator/posterior use it.
    """
    return {
        "z_beta": 1.0,
        "z_alpha": 1.0,
        "z_v": 1.0,
        "z_fc": 1.0,
        "z_lambda": 1.0,
        "z_u_scale": 1.0,
    }


def theta_constrained_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Reasonable constrained theta values (primarily for evaluation/predictive tests).

    Shapes:
      beta, alpha, v, fc: (M, J)
      lambda:            (M, N)
      u_scale:           (M,)
    """
    M, N, J = int(dims["M"]), int(dims["N"]), int(dims["J"])

    beta = np.full((M, J), 0.6, dtype=np.float64)
    alpha = np.full((M, J), 1.0, dtype=np.float64)
    v = np.full((M, J), 2.0, dtype=np.float64)
    fc = np.full((M, J), 0.2, dtype=np.float64)

    lambda_mn = np.full((M, N), 0.3, dtype=np.float64)

    # Used only in recovery/estimation (not in the DGP).
    u_scale = np.full((M,), 1.0, dtype=np.float64)

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "lambda": lambda_mn,
        "u_scale": u_scale,
    }


def z_blocks_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Prior-mode unconstrained blocks (all zeros).

    Shapes:
      z_beta, z_alpha, z_v, z_fc: (M, J)
      z_lambda:                  (M, N)
      z_u_scale:                 (M,)
    """
    M, N, J = int(dims["M"]), int(dims["N"]), int(dims["J"])
    z_mj = np.zeros((M, J), dtype=np.float64)
    z_mn = np.zeros((M, N), dtype=np.float64)
    z_m = np.zeros((M,), dtype=np.float64)

    return {
        "z_beta": z_mj.copy(),
        "z_alpha": z_mj.copy(),
        "z_v": z_mj.copy(),
        "z_fc": z_mj.copy(),
        "z_lambda": z_mn.copy(),
        "z_u_scale": z_m.copy(),
    }
