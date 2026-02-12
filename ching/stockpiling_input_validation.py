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


def _require_leading_dims(a: np.ndarray, leading: tuple[int, ...], name: str) -> None:
    if a.shape[: len(leading)] != leading:
        raise ValueError(
            f"{name}: expected leading dims={leading}, got shape={a.shape}"
        )


def _require_int_scalar(x: Any, name: str, *, min_value: int | None = None) -> int:
    # Reject bool (since bool is a subclass of int).
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{name}: expected int scalar, got bool={x}")

    if isinstance(x, (np.integer, int)):
        v = int(x)
    else:
        raise ValueError(f"{name}: expected int scalar, got type={type(x)}")

    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _require_float_scalar(
    x: Any, name: str, *, min_value: float | None = None
) -> float:
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{name}: expected float scalar, got bool={x}")

    try:
        v = float(x)
    except Exception as e:
        raise ValueError(
            f"{name}: expected float scalar, got type={type(x)} ({e})"
        ) from e

    if not np.isfinite(v):
        raise ValueError(f"{name}: expected finite float, got {v}")

    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")

    return v


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(
            f"{name}: expected all finite values, found non-finite entries"
        )


def _validate_binary_array(a: np.ndarray, name: str) -> None:
    """
    Validate a is a {0,1} array using cheap min/max checks.
    Requires integer/bool dtype.
    """
    if a.dtype.kind not in ("b", "i", "u"):
        raise ValueError(
            f"{name}: expected bool/int dtype for binary array, got dtype={a.dtype}"
        )
    if a.size == 0:
        return
    amin = int(a.min())
    amax = int(a.max())
    if amin < 0 or amax > 1:
        raise ValueError(
            f"{name}: expected binary values in {{0,1}}, got min={amin}, max={amax}"
        )


# =============================================================================
# Price inputs (shared)
# =============================================================================


def _validate_markov_tensor_mj(P_price_mj: np.ndarray, M: int, J: int) -> int:
    """
    Validate P_price_mj is (M,J,S,S), row-stochastic, entries in [0,1] (up to tolerance).
    Returns S.
    """
    _require_ndim(P_price_mj, 4, "P_price_mj")
    _require_leading_dims(P_price_mj, (M, J), "P_price_mj")

    S1 = int(P_price_mj.shape[2])
    S2 = int(P_price_mj.shape[3])
    if S1 != S2:
        raise ValueError(
            f"P_price_mj: expected last two dims (S,S), got {P_price_mj.shape[2:]}"
        )
    if S1 < 2:
        raise ValueError(f"P_price_mj: expected S>=2, got S={S1}")

    if P_price_mj.dtype.kind not in ("f",):
        raise ValueError(
            f"P_price_mj: expected float dtype, got dtype={P_price_mj.dtype}"
        )
    _require_finite(P_price_mj, "P_price_mj")

    if P_price_mj.size > 0:
        pmin = float(P_price_mj.min())
        pmax = float(P_price_mj.max())
        if pmin < -1e-12 or pmax > 1.0 + 1e-12:
            raise ValueError(
                f"P_price_mj: expected entries in [0,1], got min={pmin}, max={pmax}"
            )

    row_sums = P_price_mj.sum(axis=3)  # (M,J,S)
    max_err = float(np.max(np.abs(row_sums - 1.0)))
    if max_err > 1e-6:
        raise ValueError(f"P_price_mj: rows must sum to 1; max |row_sum-1|={max_err}")

    return S1


def _validate_price_vals_mj(price_vals_mj: np.ndarray, M: int, J: int, S: int) -> None:
    _require_ndim(price_vals_mj, 3, "price_vals_mj")
    _require_shape(price_vals_mj, (M, J, S), "price_vals_mj")

    if price_vals_mj.dtype.kind not in ("f",):
        raise ValueError(
            f"price_vals_mj: expected float dtype, got dtype={price_vals_mj.dtype}"
        )
    _require_finite(price_vals_mj, "price_vals_mj")


def _validate_price_inputs(
    P_price_mj: Any,
    price_vals_mj: Any,
    M: int,
    J: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Validate and return (P_price_mj_np, price_vals_mj_np, S).
    """
    P = _as_np(P_price_mj, "P_price_mj", dtype=np.float64)
    S = _validate_markov_tensor_mj(P, M=M, J=J)

    pv = _as_np(price_vals_mj, "price_vals_mj", dtype=np.float64)
    _validate_price_vals_mj(pv, M=M, J=J, S=S)

    return P, pv, S


def _validate_state_range(p_state_mjt: np.ndarray, S: int) -> None:
    if p_state_mjt.dtype.kind not in ("b", "i", "u"):
        raise ValueError(
            f"p_state_mjt: expected integer dtype, got dtype={p_state_mjt.dtype}"
        )
    if p_state_mjt.size == 0:
        return
    smin = int(p_state_mjt.min())
    smax = int(p_state_mjt.max())
    if smin < 0 or smax >= S:
        raise ValueError(
            f"p_state_mjt: expected state indices in [0,{S-1}], got min={smin}, max={smax}"
        )


# =============================================================================
# Panel dims helper
# =============================================================================


def _panel_dims_from_a(a_mnjt: Any) -> tuple[np.ndarray, int, int, int, int]:
    a = _as_np(a_mnjt, "a_mnjt")
    _require_ndim(a, 4, "a_mnjt")
    _validate_binary_array(a, "a_mnjt")
    M, N, J, T = (int(x) for x in a.shape)
    return a, M, N, J, T


# =============================================================================
# Public validators
# =============================================================================


def validate_stockpiling_dgp_inputs(
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    N: int,
    T: int,
    theta_true: dict[str, Any],
    I_max: int,
    P_price_mj: Any,
    price_vals_mj: Any,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> None:
    """
    Validate inputs to datasets.ching_dgp.generate_dgp(...).

    This validates:
      - Phase-1/2 utilities: delta_true, E_bar_true, njt_true
      - theta_true shapes and parameter domains
      - price process tensors
      - scalar hyperparameters (I_max, waste_cost, tol, max_iter, N, T)
    """
    N = _require_int_scalar(N, "N", min_value=1)
    T = _require_int_scalar(T, "T", min_value=1)
    I_max = _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_float_scalar(tol, "tol", min_value=0.0)
    _require_int_scalar(max_iter, "max_iter", min_value=1)

    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    njt = _as_np(njt_true, "njt_true", dtype=np.float64)

    _require_ndim(delta, 1, "delta_true")
    _require_ndim(E_bar, 1, "E_bar_true")
    _require_ndim(njt, 2, "njt_true")

    J = int(delta.shape[0])
    M = int(E_bar.shape[0])

    _require_shape(njt, (M, J), "njt_true")
    _require_finite(delta, "delta_true")
    _require_finite(E_bar, "E_bar_true")
    _require_finite(njt, "njt_true")

    # Price inputs
    P, pv, S = _validate_price_inputs(
        P_price_mj=P_price_mj, price_vals_mj=price_vals_mj, M=M, J=J
    )
    _ = (P, pv, S)  # validated; caller uses original args

    # Theta blocks
    required = {"beta", "alpha", "v", "fc", "lambda"}
    missing = required - set(theta_true.keys())
    if missing:
        raise ValueError(f"theta_true: missing keys {sorted(missing)}")

    beta = _as_np(theta_true["beta"], "theta_true['beta']", dtype=np.float64)
    alpha = _as_np(theta_true["alpha"], "theta_true['alpha']", dtype=np.float64)
    v = _as_np(theta_true["v"], "theta_true['v']", dtype=np.float64)
    fc = _as_np(theta_true["fc"], "theta_true['fc']", dtype=np.float64)
    lam = _as_np(theta_true["lambda"], "theta_true['lambda']", dtype=np.float64)

    _require_shape(beta, (M, J), "theta_true['beta']")
    _require_shape(alpha, (M, J), "theta_true['alpha']")
    _require_shape(v, (M, J), "theta_true['v']")
    _require_shape(fc, (M, J), "theta_true['fc']")
    _require_shape(lam, (M, N), "theta_true['lambda']")

    for name, arr in [
        ("theta_true['beta']", beta),
        ("theta_true['alpha']", alpha),
        ("theta_true['v']", v),
        ("theta_true['fc']", fc),
        ("theta_true['lambda']", lam),
    ]:
        _require_finite(arr, name)

    # Domain checks
    if beta.size and (beta.min() <= 0.0 or beta.max() >= 1.0):
        raise ValueError(
            f"theta_true['beta']: expected in (0,1), got min={beta.min()}, max={beta.max()}"
        )
    if lam.size and (lam.min() <= 0.0 or lam.max() >= 1.0):
        raise ValueError(
            f"theta_true['lambda']: expected in (0,1), got min={lam.min()}, max={lam.max()}"
        )
    for name, arr in [
        ("theta_true['alpha']", alpha),
        ("theta_true['v']", v),
        ("theta_true['fc']", fc),
    ]:
        if arr.size and arr.min() <= 0.0:
            raise ValueError(f"{name}: expected > 0, got min={arr.min()}")


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
      sigmas contains required z_* keys with positive values
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
    required_sigma_keys = {"z_beta", "z_alpha", "z_v", "z_fc", "z_lambda"}
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
         {"beta","alpha","v","fc","lambda"}
         each value must be a positive float.
    """
    n_iter = _require_int_scalar(n_iter, "n_iter", min_value=1)

    required = {"beta", "alpha", "v", "fc", "lambda"}
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
