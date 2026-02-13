"""
ching/stockpiling_input_validation.py

Single validation module for Phase-3 stockpiling (DGP + estimator).

This file validates:
  1) DGP generation inputs: validate_stockpiling_dgp_inputs(...)
  2) Estimator initialization inputs: validate_stockpiling_estimator_init_inputs(...)
  3) Estimator fit inputs: validate_stockpiling_estimator_fit_inputs(n_iter, k)

Conventions:
  - All inputs are accepted as array-likes; they are converted with np.asarray.
  - This module raises ValueError with consistent "expected vs got" messages.
  - Input validation is centralized here; downstream modules should not re-validate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Small helpers
# =============================================================================


def _as_np(x: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: could not convert to numpy array ({e})") from e
    return a


def _require_ndim(a: np.ndarray, ndim: int, name: str) -> None:
    if a.ndim != ndim:
        raise ValueError(
            f"{name}: expected ndim={ndim}, got ndim={a.ndim} with shape={a.shape}"
        )


def _require_shape(a: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape={shape}, got shape={a.shape}")


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: expected all finite, got non-finite entries")


def _require_int_scalar(x: Any, name: str, min_value: int | None = None) -> int:
    try:
        v = int(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: expected int scalar, got {x} ({e})") from e
    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _require_float_scalar(x: Any, name: str, min_value: float | None = None) -> float:
    try:
        v = float(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: expected float scalar, got {x} ({e})") from e
    if not np.isfinite(v):
        raise ValueError(f"{name}: expected finite float, got {v}")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _panel_dims_from_a(a_mnjt: Any) -> tuple[np.ndarray, int, int, int, int]:
    a = _as_np(a_mnjt, "a_mnjt")
    _require_ndim(a, 4, "a_mnjt")
    M, N, J, T = a.shape
    return a, int(M), int(N), int(J), int(T)


def _validate_state_range(p_state_mjt: np.ndarray, S: int) -> None:
    if p_state_mjt.size == 0:
        return
    mn = int(p_state_mjt.min())
    mx = int(p_state_mjt.max())
    if mn < 0 or mx >= S:
        raise ValueError(
            f"p_state_mjt: expected integer states in [0,{S-1}], got min={mn}, max={mx}"
        )


def _validate_price_inputs(
    *,
    P_price_mj: Any,
    price_vals_mj: Any,
    M: int,
    J: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    P = _as_np(P_price_mj, "P_price_mj", dtype=np.float64)
    _require_ndim(P, 4, "P_price_mj")
    if P.shape[0] != M or P.shape[1] != J:
        raise ValueError(
            f"P_price_mj: expected first dims (M,J)=({M},{J}), got {P.shape[:2]}"
        )
    S = int(P.shape[2])
    if P.shape != (M, J, S, S):
        raise ValueError(
            f"P_price_mj: expected shape (M,J,S,S)=({M},{J},{S},{S}), got {P.shape}"
        )
    _require_finite(P, "P_price_mj")
    if S < 2:
        raise ValueError(f"P_price_mj: expected S>=2, got S={S}")
    # row-stochastic check (lightweight)
    row_sums = P.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        mx_err = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"P_price_mj: rows must sum to 1; max |sum-1|={mx_err}")

    pv = _as_np(price_vals_mj, "price_vals_mj", dtype=np.float64)
    _require_ndim(pv, 3, "price_vals_mj")
    if pv.shape != (M, J, S):
        raise ValueError(
            f"price_vals_mj: expected shape (M,J,S)=({M},{J},{S}), got {pv.shape}"
        )
    _require_finite(pv, "price_vals_mj")

    return P, pv, S


# =============================================================================
# Public validators
# =============================================================================


def validate_stockpiling_dgp_inputs(
    *,
    seed: int,
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    N: int,
    T: int,
    I_max: int,
    P_price_mj: Any,
    price_vals_mj: Any,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> None:
    """
    Validate inputs to datasets.ching_dgp.generate_dgp.

    Shapes:
      delta_true  (J,)
      E_bar_true  (M,)
      njt_true    (M,J)
      P_price_mj  (M,J,S,S)
      price_vals_mj (M,J,S)
    """
    _require_int_scalar(seed, "seed", min_value=0)
    N = _require_int_scalar(N, "N", min_value=1)
    T = _require_int_scalar(T, "T", min_value=1)
    I_max = _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_float_scalar(tol, "tol", min_value=0.0)
    _require_int_scalar(max_iter, "max_iter", min_value=1)

    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    _require_ndim(delta, 1, "delta_true")
    J = int(delta.shape[0])
    _require_finite(delta, "delta_true")

    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    _require_ndim(E_bar, 1, "E_bar_true")
    M = int(E_bar.shape[0])
    _require_finite(E_bar, "E_bar_true")

    njt = _as_np(njt_true, "njt_true", dtype=np.float64)
    _require_ndim(njt, 2, "njt_true")
    _require_shape(njt, (M, J), "njt_true")
    _require_finite(njt, "njt_true")

    _validate_price_inputs(P_price_mj=P_price_mj, price_vals_mj=price_vals_mj, M=M, J=J)
    _ = (N, T)  # used downstream; validated here


def validate_stockpiling_estimator_init_inputs(
    *,
    a_mnjt: Any,
    p_state_mjt: Any,
    u_mj: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    I_max: int,
    pi_I0: Any,
    waste_cost: float,
    eps: float,
    tol: float,
    max_iter: int,
    sigmas: dict[str, Any],
    seed: int,
) -> None:
    """
    Validate inputs to StockpilingEstimator.__init__.

    Validates internal consistency of shapes:
      a_mnjt (M,N,J,T)
      p_state_mjt (M,J,T)
      u_mj (M,J)
      price_vals_mj (M,J,S)
      P_price_mj (M,J,S,S)
      pi_I0 (I_max+1,)

    Also validates:
      eps >= 0, tol >= 0, max_iter >= 1, seed int
      sigmas contains required z_* keys with positive values (including z_u_scale)
    """
    I_max = _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_float_scalar(eps, "eps", min_value=0.0)
    _require_float_scalar(tol, "tol", min_value=0.0)
    _require_int_scalar(max_iter, "max_iter", min_value=1)
    _require_int_scalar(seed, "seed", min_value=0)

    a, M, N, J, T = _panel_dims_from_a(a_mnjt)

    p_state = _as_np(p_state_mjt, "p_state_mjt")
    _require_ndim(p_state, 3, "p_state_mjt")
    _require_shape(p_state, (M, J, T), "p_state_mjt")

    u = _as_np(u_mj, "u_mj", dtype=np.float64)
    _require_ndim(u, 2, "u_mj")
    _require_shape(u, (M, J), "u_mj")
    _require_finite(u, "u_mj")

    # Price inputs + S
    P, pv, S = _validate_price_inputs(
        P_price_mj=P_price_mj, price_vals_mj=price_vals_mj, M=M, J=J
    )
    _validate_state_range(p_state, S)
    _ = (P, pv, a)  # validated; caller uses original args

    # Initial inventory prior
    pi = _as_np(pi_I0, "pi_I0", dtype=np.float64)
    _require_ndim(pi, 1, "pi_I0")
    _require_shape(pi, (I_max + 1,), "pi_I0")
    _require_finite(pi, "pi_I0")
    if pi.size:
        if (pi < 0.0).any():
            raise ValueError(f"pi_I0: expected nonnegative entries, got min={pi.min()}")
        s = float(pi.sum())
        if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
            raise ValueError(f"pi_I0: expected to sum to 1, got sum={s}")

    # Prior scales for z
    required_sigma_keys = {"z_beta", "z_alpha", "z_v", "z_fc", "z_lambda", "z_u_scale"}
    missing = required_sigma_keys - set(sigmas.keys())
    if missing:
        raise ValueError(f"sigmas: missing keys {sorted(missing)}")

    for k in sorted(required_sigma_keys):
        v = _require_float_scalar(sigmas[k], f"sigmas['{k}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"sigmas['{k}']: expected > 0, got {v}")


def validate_stockpiling_estimator_fit_inputs(
    *,
    n_iter: int,
    k: dict[str, Any],
) -> None:
    """
    Validate inputs to StockpilingEstimator.fit(n_iter, k).

    Args:
      n_iter: positive int
      k: dict with required keys:
         {"beta","alpha","v","fc","lambda","u_scale"}
         each value must be a positive float.
    """
    n_iter = _require_int_scalar(n_iter, "n_iter", min_value=1)

    required = {"beta", "alpha", "v", "fc", "lambda", "u_scale"}
    if not isinstance(k, dict):
        raise ValueError(f"k: expected dict, got type={type(k)}")

    missing = required - set(k.keys())
    if missing:
        raise ValueError(f"k: missing keys {sorted(missing)}")

    extra = set(k.keys()) - required
    if extra:
        raise ValueError(f"k: unexpected keys {sorted(extra)}")

    for key in sorted(required):
        v = _require_float_scalar(k[key], f"k['{key}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"k['{key}']: expected > 0, got {v}")
