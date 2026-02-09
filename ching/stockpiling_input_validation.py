# ching/stockpiling_input_validation.py
#
# Minimal Phase-3 input validation for the Ching-style stockpiling model.
# Intended use: fail fast in Python/NumPy before TF conversion or tf.function tracing.
#
# Design choice: keep checks minimal (shapes + constraints needed to avoid log/logit inf/nan
# and obvious indexing errors). Avoid deep/duplicate checks.

from __future__ import annotations

from typing import Any

import numpy as np


_REQUIRED_SIGMA_KEYS = (
    "z_beta",
    "z_alpha",
    "z_v",
    "z_fc",
    "z_lambda",
    "z_u_scale",
)

_REQUIRED_THETA_CONSUMER_KEYS = (
    "beta",
    "alpha",
    "v",
    "fc",
    "lambda_c",
)

_REQUIRED_THETA_INIT_KEYS = _REQUIRED_THETA_CONSUMER_KEYS + ("u_scale",)


def _as_np(x: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(x)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"{name}: could not convert to numpy array: {e}") from e


def _require_keys(d: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{name}: missing keys {missing}. Required keys: {list(keys)}")


def _require_ndim(a: np.ndarray, ndim: int, name: str) -> None:
    if a.ndim != ndim:
        raise ValueError(f"{name}: expected ndim={ndim}, got shape={a.shape}")


def _require_shape(a: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if tuple(a.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape={shape}, got shape={a.shape}")


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: contains non-finite values (nan/inf)")


def _require_positive_scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"{name}: expected a scalar float, got {type(x)}: {e}") from e
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name}: expected finite > 0, got {v}")
    return v


def _require_nonnegative_scalar(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:  # pragma: no cover
        raise TypeError(f"{name}: expected a scalar float, got {type(x)}: {e}") from e
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name}: expected finite >= 0, got {v}")
    return v


def _require_int_scalar(x: Any, name: str, *, min_value: int | None = None) -> int:
    try:
        v = int(x)
    except Exception as e:  # pragma: no cover
        raise TypeError(
            f"{name}: expected an int-like scalar, got {type(x)}: {e}"
        ) from e
    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _validate_markov_matrix(P: np.ndarray, name: str) -> None:
    _require_finite(P, name)
    if (P < 0).any():
        raise ValueError(f"{name}: expected all entries >= 0")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(
            f"{name}: rows must sum to 1. Row sums range [{row_sums.min()}, {row_sums.max()}]"
        )


def validate_stockpiling_dgp_inputs(
    *,
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    product_index: Any,
    N: Any,
    T: Any,
    theta_true: dict[str, Any],
    I_max: Any,
    P_price: Any,
    price_vals: Any,
    waste_cost: Any,
    tol: Any,
    max_iter: Any,
) -> tuple[int, int, int, int]:
    """
    Minimal validation for ching_dgp.generate_dgp inputs.

    Returns (M, N, T, S).
    """
    delta_true = _as_np(delta_true, "delta_true")
    E_bar_true = _as_np(E_bar_true, "E_bar_true")
    njt_true = _as_np(njt_true, "njt_true")

    _require_ndim(delta_true, 1, "delta_true")
    _require_ndim(E_bar_true, 1, "E_bar_true")
    _require_ndim(njt_true, 2, "njt_true")

    M = int(E_bar_true.shape[0])
    if njt_true.shape[0] != M:
        raise ValueError(
            f"njt_true: expected first dim M={M}, got shape={njt_true.shape}"
        )

    J = int(delta_true.shape[0])
    if njt_true.shape[1] != J:
        raise ValueError(
            f"njt_true: expected second dim J={J}, got shape={njt_true.shape}"
        )

    product_index = _require_int_scalar(product_index, "product_index", min_value=0)
    if product_index >= J:
        raise ValueError(
            f"product_index: expected 0 <= product_index < J={J}, got {product_index}"
        )

    N = _require_int_scalar(N, "N", min_value=1)
    T = _require_int_scalar(T, "T", min_value=1)

    _require_int_scalar(I_max, "I_max", min_value=0)

    P_price = _as_np(P_price, "P_price")
    price_vals = _as_np(price_vals, "price_vals")
    _require_ndim(P_price, 2, "P_price")
    _require_ndim(price_vals, 1, "price_vals")

    S = int(price_vals.shape[0])
    if P_price.shape != (S, S):
        raise ValueError(
            f"P_price: expected shape (S,S)=({S},{S}), got shape={P_price.shape}"
        )

    _require_finite(price_vals, "price_vals")
    _validate_markov_matrix(P_price, "P_price")

    _require_keys(theta_true, _REQUIRED_THETA_CONSUMER_KEYS, "theta_true")

    theta_arrays: dict[str, np.ndarray] = {}
    for k in _REQUIRED_THETA_CONSUMER_KEYS:
        a = _as_np(theta_true[k], f"theta_true[{k}]")
        _require_shape(a, (M, N), f"theta_true[{k}]")
        _require_finite(a, f"theta_true[{k}]")
        theta_arrays[k] = a

    beta = theta_arrays["beta"]
    lam = theta_arrays["lambda_c"]
    if not ((beta > 0.0) & (beta < 1.0)).all():
        raise ValueError("theta_true[beta]: expected all entries in (0,1)")
    if not ((lam > 0.0) & (lam < 1.0)).all():
        raise ValueError("theta_true[lambda_c]: expected all entries in (0,1)")

    for k in ("alpha", "v", "fc"):
        if not (theta_arrays[k] > 0.0).all():
            raise ValueError(f"theta_true[{k}]: expected all entries > 0")

    _require_nonnegative_scalar(waste_cost, "waste_cost")
    _require_positive_scalar(tol, "tol")
    _require_int_scalar(max_iter, "max_iter", min_value=1)

    return M, N, T, S


def validate_stockpiling_estimator_init_inputs(
    *,
    a_imt: Any,
    p_state_mt: Any,
    u_m: Any,
    price_vals: Any,
    P_price: Any,
    I_max: Any,
    pi_I0: Any,
    waste_cost: Any,
    eps: Any,
    tol: Any,
    max_iter: Any,
    sigmas: dict[str, Any],
    theta_init: dict[str, Any],
) -> tuple[int, int, int, int]:
    """
    Minimal validation for StockpilingEstimator.__init__ inputs.

    Returns (M, N, T, S).
    """
    a_imt = _as_np(a_imt, "a_imt")
    p_state_mt = _as_np(p_state_mt, "p_state_mt")
    u_m = _as_np(u_m, "u_m")
    price_vals = _as_np(price_vals, "price_vals")
    P_price = _as_np(P_price, "P_price")
    pi_I0 = _as_np(pi_I0, "pi_I0")

    _require_ndim(a_imt, 3, "a_imt")
    _require_ndim(p_state_mt, 2, "p_state_mt")
    _require_ndim(u_m, 1, "u_m")
    _require_ndim(price_vals, 1, "price_vals")
    _require_ndim(P_price, 2, "P_price")
    _require_ndim(pi_I0, 1, "pi_I0")

    M, N, T = map(int, a_imt.shape)
    if p_state_mt.shape[0] != M or p_state_mt.shape[1] != T:
        raise ValueError(
            f"p_state_mt: expected shape (M,T)=({M},{T}), got shape={p_state_mt.shape}"
        )
    if u_m.shape[0] != M:
        raise ValueError(f"u_m: expected shape (M,)=({M},), got shape={u_m.shape}")

    S = int(price_vals.shape[0])
    if P_price.shape != (S, S):
        raise ValueError(
            f"P_price: expected shape (S,S)=({S},{S}), got shape={P_price.shape}"
        )

    _require_finite(u_m, "u_m")
    _require_finite(price_vals, "price_vals")
    _validate_markov_matrix(P_price, "P_price")

    # Minimal value checks needed to avoid indexing / invalid log-likelihood.
    if np.issubdtype(a_imt.dtype, np.bool_):
        pass
    else:
        if not np.issubdtype(a_imt.dtype, np.integer):
            raise ValueError(
                f"a_imt: expected integer/bool values in {{0,1}}, got dtype={a_imt.dtype}"
            )
        amin = int(a_imt.min())
        amax = int(a_imt.max())
        if amin < 0 or amax > 1:
            raise ValueError(
                f"a_imt: expected values in {{0,1}}, got min={amin}, max={amax}"
            )

    if not np.issubdtype(p_state_mt.dtype, np.integer):
        raise ValueError(
            f"p_state_mt: expected integer states, got dtype={p_state_mt.dtype}"
        )
    smin = int(p_state_mt.min())
    smax = int(p_state_mt.max())
    if smin < 0 or smax >= S:
        raise ValueError(
            f"p_state_mt: expected states in [0,{S-1}], got min={smin}, max={smax}"
        )

    I_max = _require_int_scalar(I_max, "I_max", min_value=0)
    expected_I = I_max + 1
    if pi_I0.shape != (expected_I,):
        raise ValueError(
            f"pi_I0: expected shape (I_max+1,)=({expected_I},), got shape={pi_I0.shape}"
        )
    _require_finite(pi_I0, "pi_I0")
    if (pi_I0 < 0).any():
        raise ValueError("pi_I0: expected all entries >= 0")
    s = float(pi_I0.sum())
    if not np.isfinite(s) or abs(s - 1.0) > 1e-10:
        raise ValueError(f"pi_I0: expected sum == 1, got sum={s}")

    _require_nonnegative_scalar(waste_cost, "waste_cost")
    _require_positive_scalar(eps, "eps")
    _require_positive_scalar(tol, "tol")
    _require_int_scalar(max_iter, "max_iter", min_value=1)

    _require_keys(sigmas, _REQUIRED_SIGMA_KEYS, "sigmas")
    for k in _REQUIRED_SIGMA_KEYS:
        _require_positive_scalar(sigmas[k], f"sigmas[{k}]")

    _require_keys(theta_init, _REQUIRED_THETA_INIT_KEYS, "theta_init")

    # Shapes and constraints needed to avoid log/logit producing inf/nan in estimator init.
    theta0_arrays: dict[str, np.ndarray] = {}
    for k in _REQUIRED_THETA_CONSUMER_KEYS:
        a = _as_np(theta_init[k], f"theta_init[{k}]")
        _require_shape(a, (M, N), f"theta_init[{k}]")
        _require_finite(a, f"theta_init[{k}]")
        theta0_arrays[k] = a

    u_scale = _as_np(theta_init["u_scale"], "theta_init[u_scale]")
    _require_shape(u_scale, (M,), "theta_init[u_scale]")
    _require_finite(u_scale, "theta_init[u_scale]")

    beta0 = theta0_arrays["beta"]
    lam0 = theta0_arrays["lambda_c"]
    if not ((beta0 > 0.0) & (beta0 < 1.0)).all():
        raise ValueError("theta_init[beta]: expected all entries in (0,1)")
    if not ((lam0 > 0.0) & (lam0 < 1.0)).all():
        raise ValueError("theta_init[lambda_c]: expected all entries in (0,1)")

    for k in ("alpha", "v", "fc"):
        if not (theta0_arrays[k] > 0.0).all():
            raise ValueError(f"theta_init[{k}]: expected all entries > 0")

    if not (u_scale > 0.0).all():
        raise ValueError("theta_init[u_scale]: expected all entries > 0")

    return M, N, T, S


def validate_stockpiling_estimator_fit_inputs(
    *,
    n_iter: Any,
    k_beta: Any,
    k_alpha: Any,
    k_v: Any,
    k_fc: Any,
    k_lambda: Any,
    k_u_scale: Any,
) -> None:
    """Minimal validation for StockpilingEstimator.fit inputs."""
    _require_int_scalar(n_iter, "n_iter", min_value=1)

    _require_positive_scalar(k_beta, "k_beta")
    _require_positive_scalar(k_alpha, "k_alpha")
    _require_positive_scalar(k_v, "k_v")
    _require_positive_scalar(k_fc, "k_fc")
    _require_positive_scalar(k_lambda, "k_lambda")
    _require_positive_scalar(k_u_scale, "k_u_scale")
