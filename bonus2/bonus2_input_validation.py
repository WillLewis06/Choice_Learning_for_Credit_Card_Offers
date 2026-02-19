"""bonus2/bonus2_input_validation.py

Centralized input validation for Bonus Q2 (updated spec).

This module is intended to be the single source of truth for validation. Other
modules (DGP, estimator, runner) should call these functions and avoid their
own input checks.

Model-contract alignment (updated spec)
--------------------------------------
Known / observed by the estimator:
  - y_mit          (M,N,T) int, choices with outside option 0 and inside choices
                   encoded as j+1 (so values are in {0..J}).
  - delta_mj       (M,J) float, phase-1 baseline utilities.
  - weekend_t      (T,)  int in {0,1}.
  - season_sin_kt  (K,T) or (T,K) float.
  - season_cos_kt  (K,T) or (T,K) float.
  - neighbors      nested list-like: neighbors[m][i] -> list[int] within market.
  - L              int >= 1, peer lookback window length.
  - decay          float in (0,1), known scalar habit decay.

Estimator parameter blocks (updated spec):
  - init_theta keys:
      {"beta_market","beta_habit","beta_peer","beta_dow_j","a_m","b_m"}
    Each value is a scalar float used to fill the corresponding z-block.
  - sigmas keys (prior scales over z-blocks):
      {"z_beta_market_j","z_beta_habit_j","z_beta_peer_j","z_beta_dow_j","z_a_m","z_b_m"}
    Each value must be strictly positive.
  - step sizes k passed to fit():
      {"beta_market","beta_habit","beta_peer","beta_dow_j","a_m","b_m"}
    Each value must be strictly positive.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Helpers
# =============================================================================


def _as_np(x: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: could not convert to numpy array ({e})") from e


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


def _require_optional_int_scalar(
    x: Any, name: str, min_value: int | None = None
) -> int | None:
    if x is None:
        return None
    return _require_int_scalar(x, name, min_value=min_value)


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


def _require_float_in_open_interval(x: Any, name: str, lo: float, hi: float) -> float:
    v = _require_float_scalar(x, name)
    if not (lo < v < hi):
        raise ValueError(f"{name}: expected in ({lo},{hi}), got {v}")
    return v


def _require_dict(x: Any, name: str) -> dict:
    if not isinstance(x, dict):
        raise ValueError(f"{name}: expected dict, got type={type(x)}")
    return x


def _panel_dims_from_y(y_mit: Any) -> tuple[np.ndarray, int, int, int]:
    y = _as_np(y_mit, "y_mit")
    _require_ndim(y, 3, "y_mit")
    M, N, T = y.shape
    return y, int(M), int(N), int(T)


def _validate_weekend_t(weekend_t: Any, T: int) -> np.ndarray:
    w = _as_np(weekend_t, "weekend_t")
    _require_ndim(w, 1, "weekend_t")
    _require_shape(w, (T,), "weekend_t")
    try:
        w_int = w.astype(np.int64, copy=False)
    except Exception as e:  # pragma: no cover
        raise ValueError(
            f"weekend_t: expected integer-like, conversion failed ({e})"
        ) from e
    if w_int.size:
        if not np.all((w_int == 0) | (w_int == 1)):
            mn = int(w_int.min())
            mx = int(w_int.max())
            raise ValueError(
                f"weekend_t: expected values in {{0,1}}, got min={mn}, max={mx}"
            )
    return w_int


def _validate_y_range(y: np.ndarray, J: int) -> None:
    try:
        y_int = y.astype(np.int64, copy=False)
    except Exception as e:  # pragma: no cover
        raise ValueError(
            f"y_mit: expected integer-like, conversion failed ({e})"
        ) from e
    if y_int.size:
        mn = int(y_int.min())
        mx = int(y_int.max())
        if mn < 0 or mx > J:
            raise ValueError(
                f"y_mit: expected values in [0,{J}] (0=outside, 1..J=inside encoded as j+1), "
                f"got min={mn}, max={mx}"
            )


def _validate_neighbors_structure(neighbors: Any, M: int, N: int) -> None:
    """Validate neighbors[m][i] -> list[int] with indices in [0,N-1], no self, no dups."""
    if not isinstance(neighbors, (list, tuple)):
        raise ValueError(f"neighbors: expected list/tuple, got type={type(neighbors)}")
    if len(neighbors) != M:
        raise ValueError(
            f"neighbors: expected length M={M}, got length={len(neighbors)}"
        )

    for m in range(M):
        nm = neighbors[m]
        if not isinstance(nm, (list, tuple)):
            raise ValueError(
                f"neighbors[{m}]: expected list/tuple (length N={N}), got type={type(nm)}"
            )
        if len(nm) != N:
            raise ValueError(
                f"neighbors[{m}]: expected length N={N}, got length={len(nm)}"
            )

        for i in range(N):
            ni = nm[i]
            try:
                arr = np.asarray(ni, dtype=np.int64)
            except Exception as e:  # pragma: no cover
                raise ValueError(
                    f"neighbors[{m}][{i}]: could not convert to int array ({e})"
                ) from e

            if arr.ndim != 1:
                raise ValueError(
                    f"neighbors[{m}][{i}]: expected 1D, got ndim={arr.ndim} shape={arr.shape}"
                )

            if arr.size == 0:
                continue

            mn = int(arr.min())
            mx = int(arr.max())
            if mn < 0 or mx >= N:
                raise ValueError(
                    f"neighbors[{m}][{i}]: expected indices in [0,{N-1}], got min={mn}, max={mx}"
                )
            if (arr == i).any():
                raise ValueError(f"neighbors[{m}][{i}]: self-edge detected (i={i})")
            if np.unique(arr).size != arr.size:
                raise ValueError(
                    f"neighbors[{m}][{i}]: duplicate neighbor indices detected"
                )


def _validate_seasonal_features(
    season_sin_kt: Any, season_cos_kt: Any, T: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate seasonal feature matrices.

    Accepted shapes:
      - (K,T)
      - (T,K)

    Requirements:
      - season_sin_kt and season_cos_kt same shape
      - one axis equals T
      - finite float64 values
    """
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

    if int(sin.shape[1]) == T:
        K = int(sin.shape[0])
    elif int(sin.shape[0]) == T:
        K = int(sin.shape[1])
    else:
        raise ValueError(
            f"season_sin_kt/season_cos_kt: expected one axis equal to T={T}, got shape={sin.shape}"
        )

    if K < 0:
        raise ValueError(f"season_sin_kt/season_cos_kt: expected K>=0, got K={K}")

    return sin, cos, K


def _validate_required_keys(
    d: dict, name: str, required: set[str], allow_extra: bool
) -> None:
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")
    if not allow_extra:
        extra = set(d.keys()) - required
        if extra:
            raise ValueError(f"{name}: unexpected keys {sorted(extra)}")


def _validate_positive_float_dict(d: dict, name: str, required: set[str]) -> None:
    _validate_required_keys(d, name, required, allow_extra=False)
    for k in sorted(required):
        v = _require_float_scalar(d[k], f"{name}['{k}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"{name}['{k}']: expected > 0, got {v}")


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


def validate_bonus2_dgp_inputs(
    delta: Any,
    N: Any,
    T: Any,
    avg_friends: Any,
    params_true: Any,
    decay: Any,
    seed: Any,
    season_period: Any,
    friends_sd: Any,
    K: Any,
    peer_lookback_L: Any,
) -> None:
    """Validate inputs to datasets.bonus2_dgp.simulate_bonus2_dgp (updated spec)."""
    d = _as_np(delta, "delta", dtype=np.float64)
    _require_ndim(d, 2, "delta")
    _require_finite(d, "delta")
    M, J = int(d.shape[0]), int(d.shape[1])
    if M < 1 or J < 1:
        raise ValueError(f"delta: expected shape (M,J) with M>=1,J>=1, got {d.shape}")

    _require_int_scalar(N, "N", min_value=1)
    _require_int_scalar(T, "T", min_value=1)
    _require_int_scalar(season_period, "season_period", min_value=1)
    _require_int_scalar(K, "K", min_value=0)
    _require_int_scalar(peer_lookback_L, "peer_lookback_L", min_value=1)

    _require_float_scalar(avg_friends, "avg_friends", min_value=0.0)
    _require_float_scalar(friends_sd, "friends_sd", min_value=0.0)

    _require_float_in_open_interval(decay, "decay", 0.0, 1.0)

    _require_optional_int_scalar(seed, "seed", min_value=0)

    pt = _require_dict(params_true, "params_true")
    required_keys = {
        "habit_mean",
        "habit_sd",
        "peer_mean",
        "peer_sd",
        "mktprod_sd",
        "dow_prod_sd",
        "season_mkt_sd",
    }
    _validate_required_keys(pt, "params_true", required_keys, allow_extra=False)

    habit_sd = _require_float_scalar(
        pt["habit_sd"], "params_true['habit_sd']", min_value=0.0
    )
    peer_sd = _require_float_scalar(
        pt["peer_sd"], "params_true['peer_sd']", min_value=0.0
    )
    if habit_sd <= 0.0:
        raise ValueError(f"params_true['habit_sd']: expected > 0, got {habit_sd}")
    if peer_sd <= 0.0:
        raise ValueError(f"params_true['peer_sd']: expected > 0, got {peer_sd}")

    _require_float_scalar(pt["habit_mean"], "params_true['habit_mean']")
    _require_float_scalar(pt["peer_mean"], "params_true['peer_mean']")
    _require_float_scalar(pt["mktprod_sd"], "params_true['mktprod_sd']", min_value=0.0)
    _require_float_scalar(
        pt["dow_prod_sd"], "params_true['dow_prod_sd']", min_value=0.0
    )
    _require_float_scalar(
        pt["season_mkt_sd"], "params_true['season_mkt_sd']", min_value=0.0
    )


def validate_bonus2_panel(panel: Any) -> None:
    """Validate the DGP output panel dict used by the runner/estimator."""
    p = _require_dict(panel, "panel")
    required = {
        "y",
        "delta",
        "w",
        "season_sin_kt",
        "season_cos_kt",
        "nbrs",
        "decay",
        "peer_lookback_L",
    }
    _validate_required_keys(p, "panel", required, allow_extra=True)

    y, M, N, T = _panel_dims_from_y(p["y"])

    delta = _as_np(p["delta"], "panel['delta']", dtype=np.float64)
    _require_ndim(delta, 2, "panel['delta']")
    _require_finite(delta, "panel['delta']")
    if int(delta.shape[0]) != M:
        raise ValueError(
            f"panel['delta']: expected first axis M={M} to match panel['y'], got shape={delta.shape}"
        )
    J = int(delta.shape[1])
    if J < 1:
        raise ValueError(f"panel['delta']: expected J>=1, got J={J}")

    _validate_y_range(y, J)
    _validate_weekend_t(p["w"], T)
    _validate_seasonal_features(p["season_sin_kt"], p["season_cos_kt"], T)
    _validate_neighbors_structure(p["nbrs"], M=M, N=N)

    _require_int_scalar(p["peer_lookback_L"], "panel['peer_lookback_L']", min_value=1)
    _require_float_in_open_interval(p["decay"], "panel['decay']", 0.0, 1.0)


def validate_bonus2_estimator_init_inputs(
    y_mit: Any,
    delta_mj: Any,
    weekend_t: Any,
    season_sin_kt: Any,
    season_cos_kt: Any,
    neighbors: Any,
    L: Any,
    decay: Any,
    init_theta: Any,
    sigmas: Any,
    seed: Any,
) -> None:
    """Validate inputs to bonus2.bonus2_estimator.Bonus2Estimator.__init__."""
    _require_int_scalar(seed, "seed", min_value=0)

    y, M, N, T = _panel_dims_from_y(y_mit)

    d = _as_np(delta_mj, "delta_mj", dtype=np.float64)
    _require_ndim(d, 2, "delta_mj")
    _require_finite(d, "delta_mj")
    if int(d.shape[0]) != M:
        raise ValueError(f"delta_mj: expected first axis M={M}, got shape={d.shape}")
    J = int(d.shape[1])
    if J < 1:
        raise ValueError(f"delta_mj: expected J>=1, got J={J}")

    _validate_y_range(y, J)
    _validate_weekend_t(weekend_t, T)
    _validate_seasonal_features(season_sin_kt, season_cos_kt, T)
    _validate_neighbors_structure(neighbors, M=M, N=N)

    _require_int_scalar(L, "L", min_value=1)
    _require_float_in_open_interval(decay, "decay", 0.0, 1.0)

    init_theta_d = _require_dict(init_theta, "init_theta")
    required_init = {
        "beta_market",
        "beta_habit",
        "beta_peer",
        "beta_dow_j",
        "a_m",
        "b_m",
    }
    _validate_required_keys(
        init_theta_d, "init_theta", required_init, allow_extra=False
    )
    for k in sorted(required_init):
        _require_float_scalar(init_theta_d[k], f"init_theta['{k}']")

    sigmas_d = _require_dict(sigmas, "sigmas")
    required_sigma = {
        "z_beta_market_j",
        "z_beta_habit_j",
        "z_beta_peer_j",
        "z_beta_dow_j",
        "z_a_m",
        "z_b_m",
    }
    _validate_positive_float_dict(sigmas_d, "sigmas", required_sigma)


def validate_bonus2_estimator_fit_inputs(n_iter: Any, k: Any) -> None:
    """Validate inputs to bonus2.bonus2_estimator.Bonus2Estimator.fit(n_iter, k)."""
    _require_int_scalar(n_iter, "n_iter", min_value=1)

    if not isinstance(k, dict):
        raise ValueError(f"k: expected dict, got type={type(k)}")

    required = {
        "beta_market",
        "beta_habit",
        "beta_peer",
        "beta_dow_j",
        "a_m",
        "b_m",
    }
    _validate_required_keys(k, "k", required, allow_extra=False)
    for key in sorted(required):
        v = _require_float_scalar(k[key], f"k['{key}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"k['{key}']: expected > 0, got {v}")
