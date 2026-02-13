"""
bonus2/bonus2_input_validation.py

Single validation module for Bonus Q2 (DGP + estimator).

This file validates:
  1) DGP generation inputs: validate_bonus2_dgp_inputs(...)
  2) Estimator initialization inputs: validate_bonus2_estimator_init_inputs(...)
  3) Estimator fit inputs: validate_bonus2_estimator_fit_inputs(n_iter, k)

Conventions:
  - All array-like inputs are converted with np.asarray where feasible.
  - This module raises ValueError with consistent "expected vs got" messages.
  - Validation is centralized here; downstream modules should not re-validate.

Known / observed by the estimator (per model spec):
  - Time features: dow_t (day-of-week), and seasonal Fourier features sin_k_theta/cos_k_theta.
    Seasonal features are accepted as either (K,T) or (T,K); the estimator may transpose internally.
  - Social network structure (neighbors) is known and validated here.
  - Decay prior hyperparameter kappa_decay is known (scalar > 0) and validated here.
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


def _validate_neighbors_structure(neighbors: Any, M: int, N: int) -> None:
    """
    neighbors is expected to be list-like of length M,
    with neighbors[m] list-like of length N,
    and neighbors[m][i] list-like of ints in [0, N-1].
    """
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

            # Require uniqueness to avoid overweighting peers in exposure counts
            if np.unique(arr).size != arr.size:
                raise ValueError(
                    f"neighbors[{m}][{i}]: duplicate neighbor indices detected"
                )


def _validate_seasonal_features(
    sin_k_theta: Any, cos_k_theta: Any, T: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Validate seasonal feature matrices and return (sin, cos, K).

    Accepted shapes:
      - (K,T)
      - (T,K)

    Requirement:
      - sin and cos same shape
      - one axis equals T
      - finite float64 values
    """
    sin = _as_np(sin_k_theta, "sin_k_theta", dtype=np.float64)
    cos = _as_np(cos_k_theta, "cos_k_theta", dtype=np.float64)
    _require_ndim(sin, 2, "sin_k_theta")
    _require_ndim(cos, 2, "cos_k_theta")
    _require_finite(sin, "sin_k_theta")
    _require_finite(cos, "cos_k_theta")

    if sin.shape != cos.shape:
        raise ValueError(
            f"sin_k_theta/cos_k_theta: expected same shape, got {sin.shape} vs {cos.shape}"
        )

    if int(sin.shape[1]) == T:
        # (K,T)
        K = int(sin.shape[0])
    elif int(sin.shape[0]) == T:
        # (T,K)
        K = int(sin.shape[1])
    else:
        raise ValueError(
            f"sin_k_theta/cos_k_theta: expected one axis equal to T={T}, got shape={sin.shape}"
        )

    if K < 0:
        raise ValueError(f"sin_k_theta/cos_k_theta: expected K>=0, got K={K}")

    return sin, cos, K


# =============================================================================
# Public validators
# =============================================================================


def validate_bonus2_dgp_inputs(
    delta: Any,
    N: int,
    T: int,
    avg_friends: float,
    params_true: dict,
    average_decay_rate: float,
    seed: Any,
    season_period: int,
    friends_sd: float,
    K: int,
    peer_lookback_L: int,
) -> None:
    """
    Validate inputs to datasets.bonus2_dgp.simulate_bonus2_dgp.

    Shapes:
      delta: (M,J) float64 finite
    """
    d = _as_np(delta, "delta", dtype=np.float64)
    _require_ndim(d, 2, "delta")
    _require_finite(d, "delta")
    M, J = (int(d.shape[0]), int(d.shape[1]))
    if M < 1 or J < 1:
        raise ValueError(f"delta: expected shape (M,J) with M>=1,J>=1, got {d.shape}")

    _require_int_scalar(N, "N", min_value=1)
    _require_int_scalar(T, "T", min_value=1)
    _require_int_scalar(season_period, "season_period", min_value=1)
    _require_int_scalar(K, "K", min_value=0)
    _require_int_scalar(peer_lookback_L, "peer_lookback_L", min_value=1)

    _require_float_scalar(avg_friends, "avg_friends", min_value=0.0)
    _require_float_scalar(friends_sd, "friends_sd", min_value=0.0)

    _require_float_in_open_interval(average_decay_rate, "average_decay_rate", 0.0, 1.0)

    _require_optional_int_scalar(seed, "seed", min_value=0)

    params_true = _require_dict(params_true, "params_true")
    required_keys = {
        "habit_mean",
        "habit_sd",
        "peer_mean",
        "peer_sd",
        "mktprod_sd",
        "dow_mkt_sd",
        "dow_prod_sd",
        "season_mkt_sd",
        "season_prod_sd",
        "decay_rate_eps",
    }
    missing = required_keys - set(params_true.keys())
    if missing:
        raise ValueError(f"params_true: missing keys {sorted(missing)}")

    habit_sd = _require_float_scalar(
        params_true["habit_sd"], "params_true['habit_sd']", min_value=0.0
    )
    peer_sd = _require_float_scalar(
        params_true["peer_sd"], "params_true['peer_sd']", min_value=0.0
    )
    if habit_sd <= 0.0:
        raise ValueError(f"params_true['habit_sd']: expected > 0, got {habit_sd}")
    if peer_sd <= 0.0:
        raise ValueError(f"params_true['peer_sd']: expected > 0, got {peer_sd}")

    _require_float_scalar(params_true["habit_mean"], "params_true['habit_mean']")
    _require_float_scalar(params_true["peer_mean"], "params_true['peer_mean']")

    _require_float_scalar(
        params_true["mktprod_sd"], "params_true['mktprod_sd']", min_value=0.0
    )
    _require_float_scalar(
        params_true["dow_mkt_sd"], "params_true['dow_mkt_sd']", min_value=0.0
    )
    _require_float_scalar(
        params_true["dow_prod_sd"], "params_true['dow_prod_sd']", min_value=0.0
    )
    _require_float_scalar(
        params_true["season_mkt_sd"], "params_true['season_mkt_sd']", min_value=0.0
    )
    _require_float_scalar(
        params_true["season_prod_sd"], "params_true['season_prod_sd']", min_value=0.0
    )

    decay_eps = _require_float_scalar(
        params_true["decay_rate_eps"], "params_true['decay_rate_eps']", min_value=0.0
    )
    if decay_eps >= 1.0:
        raise ValueError(
            f"params_true['decay_rate_eps']: expected < 1, got {decay_eps}"
        )


def validate_bonus2_estimator_init_inputs(
    y_mit: Any,
    delta_mj: Any,
    dow_t: Any,
    sin_k_theta: Any,
    cos_k_theta: Any,
    neighbors: Any,
    L: int,
    init_theta: dict[str, Any],
    sigmas: dict[str, Any],
    seed: int,
    kappa_decay: Any,
    eps_decay: Any | None = None,
) -> None:
    """
    Validate inputs to bonus2.bonus2_estimator.Bonus2Estimator.__init__.

    Shapes / conventions:
      y_mit        (M,N,T) int in {0..J}
      delta_mj     (M,J) float64 finite
      dow_t        (T,)  int in {0..6}
      sin_k_theta  (K,T) or (T,K) float64 finite
      cos_k_theta  (K,T) or (T,K) float64 finite
      neighbors    list-like length M of list-like length N of int arrays
      L            int >= 1
      kappa_decay  float > 0 (known hyperparameter for decay prior Beta(kappa,1))
      eps_decay    optional float in [0,1)
    """
    _require_int_scalar(seed, "seed", min_value=0)

    # kappa_decay known hyperparameter for decay prior
    kd = _require_float_scalar(kappa_decay, "kappa_decay", min_value=0.0)
    if kd <= 0.0:
        raise ValueError(f"kappa_decay: expected > 0, got {kd}")

    if eps_decay is not None:
        ed = _require_float_scalar(eps_decay, "eps_decay", min_value=0.0)
        if ed >= 1.0:
            raise ValueError(f"eps_decay: expected < 1, got {ed}")

    y, M, N, T = _panel_dims_from_y(y_mit)

    d = _as_np(delta_mj, "delta_mj", dtype=np.float64)
    _require_ndim(d, 2, "delta_mj")
    _require_finite(d, "delta_mj")
    if int(d.shape[0]) != M:
        raise ValueError(f"delta_mj: expected first axis M={M}, got shape={d.shape}")
    J = int(d.shape[1])
    if J < 1:
        raise ValueError(f"delta_mj: expected J>=1, got J={J}")

    # y range check (0..J)
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
                f"y_mit: expected values in [0,{J}] (0=outside, 1..J=inside), got min={mn}, max={mx}"
            )

    dow = _as_np(dow_t, "dow_t")
    _require_ndim(dow, 1, "dow_t")
    _require_shape(dow, (T,), "dow_t")
    try:
        dow_int = dow.astype(np.int64, copy=False)
    except Exception as e:  # pragma: no cover
        raise ValueError(
            f"dow_t: expected integer-like, conversion failed ({e})"
        ) from e
    if dow_int.size:
        mn = int(dow_int.min())
        mx = int(dow_int.max())
        if mn < 0 or mx > 6:
            raise ValueError(f"dow_t: expected values in [0,6], got min={mn}, max={mx}")

    # Seasonal features: accept (K,T) or (T,K)
    _validate_seasonal_features(sin_k_theta=sin_k_theta, cos_k_theta=cos_k_theta, T=T)

    _validate_neighbors_structure(neighbors=neighbors, M=M, N=N)

    _require_int_scalar(L, "L", min_value=1)

    init_theta = _require_dict(init_theta, "init_theta")
    required_init = {
        "beta_market",
        "beta_habit",
        "beta_peer",
        "decay_rate",
        "beta_dow_m",
        "beta_dow_j",
        "a_m",
        "b_m",
        "a_j",
        "b_j",
    }
    missing = required_init - set(init_theta.keys())
    if missing:
        raise ValueError(f"init_theta: missing keys {sorted(missing)}")

    for k in sorted(required_init):
        if k == "decay_rate":
            _require_float_in_open_interval(
                init_theta[k], "init_theta['decay_rate']", 0.0, 1.0
            )
        else:
            _require_float_scalar(init_theta[k], f"init_theta['{k}']")

    sigmas = _require_dict(sigmas, "sigmas")
    required_sigma_keys = {
        "z_beta_market_mj",
        "z_beta_habit_j",
        "z_beta_peer_j",
        "z_decay_rate_j",
        "z_beta_dow_m",
        "z_beta_dow_j",
        "z_a_m",
        "z_b_m",
        "z_a_j",
        "z_b_j",
    }
    missing = required_sigma_keys - set(sigmas.keys())
    if missing:
        raise ValueError(f"sigmas: missing keys {sorted(missing)}")

    extra = set(sigmas.keys()) - required_sigma_keys
    if extra:
        raise ValueError(f"sigmas: unexpected keys {sorted(extra)}")

    for k in sorted(required_sigma_keys):
        v = _require_float_scalar(sigmas[k], f"sigmas['{k}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"sigmas['{k}']: expected > 0, got {v}")


def validate_bonus2_estimator_fit_inputs(n_iter: int, k: dict[str, Any]) -> None:
    """
    Validate inputs to bonus2.bonus2_estimator.Bonus2Estimator.fit(n_iter, k).

    Args:
      n_iter: positive int
      k: dict with required keys:
         {"beta_market","beta_habit","beta_peer","decay_rate",
          "beta_dow_m","beta_dow_j","a_m","b_m","a_j","b_j"}
         each value must be a positive float.
    """
    _require_int_scalar(n_iter, "n_iter", min_value=1)

    if not isinstance(k, dict):
        raise ValueError(f"k: expected dict, got type={type(k)}")

    required = {
        "beta_market",
        "beta_habit",
        "beta_peer",
        "decay_rate",
        "beta_dow_m",
        "beta_dow_j",
        "a_m",
        "b_m",
        "a_j",
        "b_j",
    }

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
