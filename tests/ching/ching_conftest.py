# ching_conftest.py
"""
Shared Phase-3 (Ching-style stockpiling) test builders.

This file is intentionally not a pytest conftest module and does not define pytest
fixtures. Tests should import and call these helpers directly.

Key conventions:
  a_imt:      (M, N, T) int32 actions in {0,1}
  p_state_mt: (M, T)    int32 price-state indices in {0,...,S-1}
  ccp_buy:    (M, N, S, I) where I = I_max + 1
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
      dict with keys: M, N, T, S, I_max
    """
    return {"M": 2, "N": 3, "T": 5, "S": 2, "I_max": 2}


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
    Prices by state and Markov transition matrix for price states.

    For S=2:
      price_vals = [1.0, 0.8]
      P_price    = [[0.9,0.1],[0.2,0.8]]
    """
    S = int(dims["S"])
    if S != 2:
        raise ValueError(
            "price_process currently assumes S=2; update if S>2 is needed."
        )

    price_vals = np.asarray([1.0, 0.8], dtype=np.float64)
    p_price = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    return {"price_vals": price_vals, "P_price": p_price}


def pi_I0_uniform(dims: dict[str, int]) -> np.ndarray:
    """Uniform initial inventory belief pi_I0 over {0,...,I_max}."""
    i = int(dims["I_max"]) + 1
    return (np.ones(i, dtype=np.float64) / float(i)).astype(np.float64)


# Optional alias with no capitals in the name.
pi_i0_uniform = pi_I0_uniform


def panel_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Deterministic, non-degenerate panel.

    - p_state_mt cycles through states (ensures both states appear).
    - a_imt uses a deterministic pattern across (m,n,t) containing both 0 and 1.
    """
    M, N, T, S = int(dims["M"]), int(dims["N"]), int(dims["T"]), int(dims["S"])

    p_state_mt = np.empty((M, T), dtype=np.int32)
    for m in range(M):
        p_state_mt[m, :] = (np.arange(T, dtype=np.int32) + m) % S

    a_imt = np.empty((M, N, T), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            a_imt[m, n, :] = (
                (m * 31 + n * 17 + np.arange(T, dtype=np.int32)) % 2
            ).astype(np.int32)

    return {"a_imt": a_imt, "p_state_mt": p_state_mt}


def u_m_np(dims: dict[str, int]) -> np.ndarray:
    """Market utilities u_m (M,) used as fixed inputs in Phase-3."""
    M = int(dims["M"])
    return np.linspace(0.5, 1.5, M, dtype=np.float64)


def sigmas() -> dict[str, float]:
    """Prior scales for z-blocks (must include all keys used by posterior/estimator)."""
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
      beta, alpha, v, fc, lambda_c: (M, N)
      u_scale: (M,)
    """
    M, N = int(dims["M"]), int(dims["N"])

    beta = np.full((M, N), 0.6, dtype=np.float64)
    alpha = np.full((M, N), 1.0, dtype=np.float64)
    v = np.full((M, N), 2.0, dtype=np.float64)
    fc = np.full((M, N), 0.2, dtype=np.float64)
    lambda_c = np.full((M, N), 0.3, dtype=np.float64)
    u_scale = np.full((M,), 1.0, dtype=np.float64)

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "lambda_c": lambda_c,
        "u_scale": u_scale,
    }


def z_blocks_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Prior-mode unconstrained blocks (all zeros).

    Shapes:
      z_beta, z_alpha, z_v, z_fc, z_lambda: (M, N)
      z_u_scale: (M,)
    """
    M, N = int(dims["M"]), int(dims["N"])
    z_mn = np.zeros((M, N), dtype=np.float64)
    z_m = np.zeros((M,), dtype=np.float64)

    return {
        "z_beta": z_mn.copy(),
        "z_alpha": z_mn.copy(),
        "z_v": z_mn.copy(),
        "z_fc": z_mn.copy(),
        "z_lambda": z_mn.copy(),
        "z_u_scale": z_m.copy(),
    }
