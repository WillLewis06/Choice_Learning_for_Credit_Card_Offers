"""
bonus2_input_validation.py

Centralized input validation for Bonus Q2.

This module is the single source of truth for validating external inputs:
- raw arrays (panel inputs, Phase-1 outputs)
- configuration dictionaries (init values, prior scales, step sizes)
- DGP configuration (if used)

Other Bonus2 modules must not implement their own validation, fallback logic, or
shape/dtype coercions beyond straightforward conversion after validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Constants
# =============================================================================


Z_KEYS: tuple[str, ...] = (
    "z_beta_intercept_j",
    "z_beta_habit_j",
    "z_beta_peer_j",
    "z_beta_weekend_jw",
    "z_a_m",
    "z_b_m",
)


INIT_THETA_KEYS: tuple[str, ...] = (
    "beta_intercept",
    "beta_habit",
    "beta_peer",
    "beta_weekend_weekday",
    "beta_weekend_weekend",
    "a_m",
    "b_m",
)


# =============================================================================
# Helpers
# =============================================================================


def _as_np(x: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    """Convert to numpy array or raise with a clear message."""
    try:
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: could not convert to numpy array ({e})") from e


def _require_ndim(a: np.ndarray, ndim: int, name: str) -> None:
    if a.ndim != ndim:
        raise ValueError(
            f"{name}: expected ndim={ndim}, got ndim={a.ndim} shape={a.shape}"
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


def _require_float_scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: expected float scalar, got {x} ({e})") from e
    if not np.isfinite(v):
        raise ValueError(f"{name}: expected finite float, got {v}")
    return v


def _require_float_in_open_interval(x: Any, name: str, lo: float, hi: float) -> float:
    v = _require_float_scalar(x, name)
    if not (lo < v < hi):
        raise ValueError(f"{name}: expected in ({lo},{hi}), got {v}")
    return v


def _require_dict(x: Any, name: str) -> dict:
    if not isinstance(x, dict):
        raise ValueError(f"{name}: expected dict, got type={type(x)}")
    return x


def _validate_required_keys(d: dict, name: str, required: set[str]) -> None:
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")
    extra = set(d.keys()) - required
    if extra:
        raise ValueError(f"{name}: unexpected keys {sorted(extra)}")


def _validate_positive_float_dict(d: dict, name: str, required: set[str]) -> None:
    _validate_required_keys(d, name, required)
    for k in sorted(required):
        v = _require_float_scalar(d[k], f"{name}['{k}']")
        if v <= 0.0:
            raise ValueError(f"{name}['{k}']: expected > 0, got {v}")


def _panel_dims_from_y(y_mit: Any, name: str) -> tuple[np.ndarray, int, int, int]:
    y = _as_np(y_mit, name)
    _require_ndim(y, 3, name)
    if y.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name}: expected integer dtype, got dtype={y.dtype}")
    M, N, T = (int(x) for x in y.shape)
    return y, M, N, T


def _validate_is_weekend_t(is_weekend_t: Any, T: int, name: str) -> np.ndarray:
    w = _as_np(is_weekend_t, name)
    _require_ndim(w, 1, name)
    _require_shape(w, (T,), name)

    if w.dtype.kind not in ("i", "u", "b"):
        raise ValueError(f"{name}: expected integer/bool dtype, got dtype={w.dtype}")

    w_int = w.astype(np.int64, copy=False)
    if w_int.size:
        if not np.all((w_int == 0) | (w_int == 1)):
            mn = int(w_int.min())
            mx = int(w_int.max())
            raise ValueError(
                f"{name}: expected values in {{0,1}}, got min={mn}, max={mx}"
            )
    return w_int


def _validate_y_range(y: np.ndarray, J: int, name: str) -> None:
    y_int = y.astype(np.int64, copy=False)
    if y_int.size:
        mn = int(y_int.min())
        mx = int(y_int.max())
        if mn < 0 or mx > J:
            raise ValueError(
                f"{name}: expected values in [0,{J}] (0=outside, 1..J=inside encoded as j+1), "
                f"got min={mn}, max={mx}"
            )


def _validate_neighbors_m(neighbors_m: Any, M: int, N: int, name: str) -> None:
    """Validate neighbors_m[m][i] -> 1D integer list with indices in [0,N-1], no self, no dups."""
    if not isinstance(neighbors_m, (list, tuple)):
        raise ValueError(f"{name}: expected list/tuple, got type={type(neighbors_m)}")
    if len(neighbors_m) != M:
        raise ValueError(
            f"{name}: expected length M={M}, got length={len(neighbors_m)}"
        )

    for m in range(M):
        nm = neighbors_m[m]
        if not isinstance(nm, (list, tuple)):
            raise ValueError(
                f"{name}[{m}]: expected list/tuple (length N={N}), got type={type(nm)}"
            )
        if len(nm) != N:
            raise ValueError(
                f"{name}[{m}]: expected length N={N}, got length={len(nm)}"
            )

        for i in range(N):
            ni = nm[i]
            arr = _as_np(ni, f"{name}[{m}][{i}]")
            if arr.dtype.kind not in ("i", "u"):
                raise ValueError(
                    f"{name}[{m}][{i}]: expected integer dtype, got dtype={arr.dtype}"
                )
            _require_ndim(arr, 1, f"{name}[{m}][{i}]")

            if arr.size == 0:
                continue

            mn = int(arr.min())
            mx = int(arr.max())
            if mn < 0 or mx >= N:
                raise ValueError(
                    f"{name}[{m}][{i}]: expected indices in [0,{N-1}], got min={mn}, max={mx}"
                )
            if (arr == i).any():
                raise ValueError(f"{name}[{m}][{i}]: self-edge detected (i={i})")
            if np.unique(arr).size != arr.size:
                raise ValueError(
                    f"{name}[{m}][{i}]: duplicate neighbor indices detected"
                )


def _validate_seasonal_features(
    season_sin_kt: Any, season_cos_kt: Any, T: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate seasonal basis matrices with canonical shape (K,T)."""
    sin = _as_np(season_sin_kt, "season_sin_kt", dtype=np.float64)
    cos = _as_np(season_cos_kt, "season_cos_kt", dtype=np.float64)

    _require_ndim(sin, 2, "season_sin_kt")
    _require_ndim(cos, 2, "season_cos_kt")
    _require_finite(sin, "season_sin_kt")
    _require_finite(cos, "season_cos_kt")

    if sin.shape != cos.shape:
        raise ValueError(
            "season_sin_kt/season_cos_kt: expected same shape, "
            f"got {sin.shape} vs {cos.shape}"
        )

    if int(sin.shape[1]) != int(T):
        raise ValueError(
            f"season_sin_kt/season_cos_kt: expected shape (K,T) with T={T}, got shape={sin.shape}"
        )

    K = int(sin.shape[0])
    if K < 0:
        raise ValueError(f"season_sin_kt/season_cos_kt: expected K>=0, got K={K}")

    return sin, cos, K


# =============================================================================
# Public validators
# =============================================================================


def validate_phase1_delta_hat(delta_hat: Any, num_products: Any) -> None:
    """Validate Phase-1 delta_hat vector used to build delta_mj."""
    J = _require_int_scalar(num_products, "num_products", min_value=1)
    a = _as_np(delta_hat, "delta_hat", dtype=np.float64)
    _require_ndim(a, 1, "delta_hat")
    _require_shape(a, (J,), "delta_hat")
    _require_finite(a, "delta_hat")


def validate_bonus2_panel(panel: Any) -> None:
    """Validate an estimator-ready Bonus2 panel dict (canonical schema)."""
    p = _require_dict(panel, "panel")
    required = {
        "y_mit",
        "delta_mj",
        "is_weekend_t",
        "season_sin_kt",
        "season_cos_kt",
        "neighbors_m",
        "lookback",
        "decay",
    }
    _validate_required_keys(p, "panel", required)

    y, M, N, T = _panel_dims_from_y(p["y_mit"], "panel['y_mit']")

    delta = _as_np(p["delta_mj"], "panel['delta_mj']", dtype=np.float64)
    _require_ndim(delta, 2, "panel['delta_mj']")
    _require_finite(delta, "panel['delta_mj']")
    if int(delta.shape[0]) != M:
        raise ValueError(
            f"panel['delta_mj']: expected first axis M={M} to match panel['y_mit'], got shape={delta.shape}"
        )
    J = int(delta.shape[1])
    if J < 1:
        raise ValueError(f"panel['delta_mj']: expected J>=1, got J={J}")

    _validate_y_range(y, J, "panel['y_mit']")
    _validate_is_weekend_t(p["is_weekend_t"], T, "panel['is_weekend_t']")
    _validate_seasonal_features(p["season_sin_kt"], p["season_cos_kt"], T)
    _validate_neighbors_m(p["neighbors_m"], M=M, N=N, name="panel['neighbors_m']")

    _require_int_scalar(p["lookback"], "panel['lookback']", min_value=1)
    _require_float_in_open_interval(p["decay"], "panel['decay']", 0.0, 1.0)


def validate_bonus2_estimator_init_inputs(
    y_mit: Any,
    delta_mj: Any,
    is_weekend_t: Any,
    season_sin_kt: Any,
    season_cos_kt: Any,
    neighbors_m: Any,
    lookback: Any,
    decay: Any,
    init_theta: Any,
    sigmas: Any,
    step_size_z: Any,
    seed: Any,
) -> None:
    """Validate inputs to Bonus2Estimator.__init__ (canonical schema)."""
    _require_int_scalar(seed, "seed", min_value=0)

    y, M, N, T = _panel_dims_from_y(y_mit, "y_mit")

    d = _as_np(delta_mj, "delta_mj", dtype=np.float64)
    _require_ndim(d, 2, "delta_mj")
    _require_finite(d, "delta_mj")
    if int(d.shape[0]) != M:
        raise ValueError(f"delta_mj: expected first axis M={M}, got shape={d.shape}")
    J = int(d.shape[1])
    if J < 1:
        raise ValueError(f"delta_mj: expected J>=1, got J={J}")

    _validate_y_range(y, J, "y_mit")
    _validate_is_weekend_t(is_weekend_t, T, "is_weekend_t")
    _validate_seasonal_features(season_sin_kt, season_cos_kt, T)
    _validate_neighbors_m(neighbors_m, M=M, N=N, name="neighbors_m")

    _require_int_scalar(lookback, "lookback", min_value=1)
    _require_float_in_open_interval(decay, "decay", 0.0, 1.0)

    init_theta_d = _require_dict(init_theta, "init_theta")
    _validate_required_keys(init_theta_d, "init_theta", set(INIT_THETA_KEYS))
    for k in INIT_THETA_KEYS:
        _require_float_scalar(init_theta_d[k], f"init_theta['{k}']")

    sigmas_d = _require_dict(sigmas, "sigmas")
    _validate_positive_float_dict(sigmas_d, "sigmas", set(Z_KEYS))

    step_d = _require_dict(step_size_z, "step_size_z")
    _validate_positive_float_dict(step_d, "step_size_z", set(Z_KEYS))


def validate_bonus2_estimator_fit_inputs(n_iter: Any) -> None:
    """Validate inputs to Bonus2Estimator.fit(n_iter)."""
    _require_int_scalar(n_iter, "n_iter", min_value=1)


def validate_bonus2_dgp_inputs(
    delta_mj: Any,
    N: Any,
    T: Any,
    avg_friends: Any,
    params_true: Any,
    decay: Any,
    seed: Any,
    season_period: Any,
    friends_sd: Any,
    K: Any,
    lookback: Any,
) -> None:
    """Validate inputs to the Bonus2 DGP simulation (canonical naming)."""
    d = _as_np(delta_mj, "delta_mj", dtype=np.float64)
    _require_ndim(d, 2, "delta_mj")
    _require_finite(d, "delta_mj")
    M, J = int(d.shape[0]), int(d.shape[1])
    if M < 1 or J < 1:
        raise ValueError(
            f"delta_mj: expected shape (M,J) with M>=1,J>=1, got {d.shape}"
        )

    _require_int_scalar(N, "N", min_value=1)
    _require_int_scalar(T, "T", min_value=1)
    _require_int_scalar(season_period, "season_period", min_value=1)
    _require_int_scalar(K, "K", min_value=0)
    _require_int_scalar(lookback, "lookback", min_value=1)

    _require_float_scalar(avg_friends, "avg_friends")
    _require_float_scalar(friends_sd, "friends_sd")
    _require_float_in_open_interval(decay, "decay", 0.0, 1.0)
    _require_int_scalar(seed, "seed", min_value=0)

    pt = _require_dict(params_true, "params_true")
    required_keys = {
        "habit_mean",
        "habit_sd",
        "peer_mean",
        "peer_sd",
        "mktprod_sd",
        "weekend_prod_sd",
        "season_mkt_sd",
    }
    _validate_required_keys(pt, "params_true", required_keys)

    _require_float_scalar(pt["habit_mean"], "params_true['habit_mean']")
    _require_float_scalar(pt["peer_mean"], "params_true['peer_mean']")

    habit_sd = _require_float_scalar(pt["habit_sd"], "params_true['habit_sd']")
    peer_sd = _require_float_scalar(pt["peer_sd"], "params_true['peer_sd']")
    if habit_sd <= 0.0:
        raise ValueError(f"params_true['habit_sd']: expected > 0, got {habit_sd}")
    if peer_sd <= 0.0:
        raise ValueError(f"params_true['peer_sd']: expected > 0, got {peer_sd}")

    _require_float_scalar(pt["mktprod_sd"], "params_true['mktprod_sd']")
    _require_float_scalar(pt["weekend_prod_sd"], "params_true['weekend_prod_sd']")
    _require_float_scalar(pt["season_mkt_sd"], "params_true['season_mkt_sd']")
