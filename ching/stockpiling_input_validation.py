"""Validate external inputs for the Ching stockpiling DGP and chain.

This module now contains two separate validation layers:

1. DGP input validation / normalization
   - validate_stockpiling_dgp_inputs
   - normalize_stockpiling_dgp_inputs

2. Refactored chain input validation
   - observed_data_validate_input
   - posterior_validate_input
   - sampler_validate_input
   - init_state_validate_input
   - seed_validate_input
   - run_chain_validate_input

The DGP validators work on NumPy-style external inputs and return canonical numpy
arrays / Python scalars. The chain validators work on already-normalized
TensorFlow tensors and config objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

ATOL_PROB = 1e-8

__all__ = [
    "validate_stockpiling_dgp_inputs",
    "normalize_stockpiling_dgp_inputs",
    "observed_data_validate_input",
    "posterior_validate_input",
    "sampler_validate_input",
    "init_state_validate_input",
    "seed_validate_input",
    "run_chain_validate_input",
]


def _require(cond: bool, msg: str) -> None:
    """Raise ValueError when a validation condition fails."""
    if not cond:
        raise ValueError(msg)


def _require_type(x, t, msg: str) -> None:
    """Raise TypeError when a value is not of the required type."""
    if not isinstance(x, t):
        raise TypeError(msg)


# =============================================================================
# DGP validation helpers (NumPy / Python boundary)
# =============================================================================


def _as_np(x: Any, name: str, dtype) -> np.ndarray:
    """Convert an external input to a numpy array of the requested dtype."""
    try:
        arr = np.asarray(x, dtype=dtype)
    except Exception as exc:
        raise TypeError(f"{name} could not be converted to {dtype}.") from exc
    return arr


def _require_ndim(x: np.ndarray, ndim: int, name: str) -> None:
    """Validate numpy array rank."""
    if x.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {x.shape}.")


def _require_shape(x: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    """Validate exact numpy array shape."""
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {x.shape}.")


def _require_finite_np(x: np.ndarray, name: str) -> None:
    """Validate that a numpy array contains only finite entries."""
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite (no NaN/inf).")


def _require_int_scalar(x: Any, name: str, min_value: int | None = None) -> int:
    """Validate a Python / NumPy integer scalar."""
    if isinstance(x, bool) or not np.isscalar(x):
        raise TypeError(f"{name} must be an integer scalar; got {type(x)}.")
    if isinstance(x, (float, np.floating)) and not float(x).is_integer():
        raise TypeError(f"{name} must be an integer scalar; got {x}.")
    try:
        value = int(x)
    except Exception as exc:
        raise TypeError(f"{name} must be an integer scalar; got {x}.") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {value}.")
    return value


def _require_float_scalar(x: Any, name: str, min_value: float | None = None) -> float:
    """Validate a Python / NumPy real scalar."""
    if isinstance(x, bool) or not np.isscalar(x):
        raise TypeError(f"{name} must be a real scalar; got {type(x)}.")
    try:
        value = float(x)
    except Exception as exc:
        raise TypeError(f"{name} must be a real scalar; got {x}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {value}.")
    return value


def _validate_price_inputs(
    P_price_mj: Any,
    price_vals_mj: Any,
    M: int,
    J: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate price transition matrices and price levels for the DGP."""
    P = _as_np(P_price_mj, "P_price_mj", dtype=np.float64)
    pv = _as_np(price_vals_mj, "price_vals_mj", dtype=np.float64)

    _require_ndim(P, 4, "P_price_mj")
    _require_ndim(pv, 3, "price_vals_mj")

    if P.shape[0] != M or P.shape[1] != J:
        raise ValueError(
            f"P_price_mj must have leading shape (M,J)=({M},{J}); got {P.shape[:2]}."
        )
    if pv.shape[0] != M or pv.shape[1] != J:
        raise ValueError(
            f"price_vals_mj must have leading shape (M,J)=({M},{J}); got {pv.shape[:2]}."
        )

    S = int(P.shape[2])
    if P.shape[3] != S:
        raise ValueError(f"P_price_mj must have shape (M,J,S,S); got {P.shape}.")
    if S < 2:
        raise ValueError(f"P_price_mj must have S>=2; got S={S}.")
    if pv.shape[2] != S:
        raise ValueError(
            f"price_vals_mj must have shape (M,J,S)=({M},{J},{S}); got {pv.shape}."
        )

    _require_finite_np(P, "P_price_mj")
    _require_finite_np(pv, "price_vals_mj")

    if np.any(P < 0.0):
        raise ValueError("P_price_mj must be nonnegative.")
    row_sums = P.sum(axis=-1)
    if not np.all(np.abs(row_sums - 1.0) <= ATOL_PROB):
        raise ValueError("P_price_mj must be row-stochastic along the last axis.")

    if np.any(pv <= 0.0):
        raise ValueError("price_vals_mj must be strictly positive.")

    return P, pv, S


def validate_stockpiling_dgp_inputs(
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    N: int,
    T: int,
    I_max: int,
    waste_cost: float,
    seed: int,
    tol: float,
    max_iter: int,
) -> None:
    """Validate inputs to the Phase-3 stockpiling DGP generator."""
    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    _require_ndim(delta, 1, "delta_true")
    _require_finite_np(delta, "delta_true")
    J = int(delta.shape[0])
    if J < 1:
        raise ValueError(f"delta_true must have length J>=1; got J={J}.")

    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    _require_ndim(E_bar, 1, "E_bar_true")
    _require_finite_np(E_bar, "E_bar_true")
    M = int(E_bar.shape[0])
    if M < 1:
        raise ValueError(f"E_bar_true must have length M>=1; got M={M}.")

    njt = _as_np(njt_true, "njt_true", dtype=np.float64)
    _require_ndim(njt, 2, "njt_true")
    _require_shape(njt, (M, J), "njt_true")
    _require_finite_np(njt, "njt_true")

    _require_int_scalar(N, "N", min_value=1)
    _require_int_scalar(T, "T", min_value=1)
    _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_int_scalar(seed, "seed", min_value=0)
    _require_float_scalar(tol, "tol", min_value=0.0)
    _require_int_scalar(max_iter, "max_iter", min_value=1)

    _validate_price_inputs(P_price_mj, price_vals_mj, M=M, J=J)


def normalize_stockpiling_dgp_inputs(
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    N: int,
    T: int,
    I_max: int,
    waste_cost: float,
    seed: int,
    tol: float,
    max_iter: int,
) -> dict[str, Any]:
    """Validate and normalize DGP inputs into canonical numpy dtypes/shapes."""
    validate_stockpiling_dgp_inputs(
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        N=N,
        T=T,
        I_max=I_max,
        waste_cost=waste_cost,
        seed=seed,
        tol=tol,
        max_iter=max_iter,
    )

    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    njt = _as_np(njt_true, "njt_true", dtype=np.float64)

    M = int(E_bar.shape[0])
    J = int(delta.shape[0])
    P, pv, S = _validate_price_inputs(P_price_mj, price_vals_mj, M=M, J=J)

    return {
        "seed": int(seed),
        "M": M,
        "N": int(N),
        "J": J,
        "T": int(T),
        "I_max": int(I_max),
        "delta_true": delta,
        "E_bar_true": E_bar,
        "njt_true": njt,
        "P_price_mj": P,
        "price_vals_mj": pv,
        "waste_cost": float(waste_cost),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "S": S,
    }


# =============================================================================
# Refactored chain validation helpers (TensorFlow boundary)
# =============================================================================


def _require_int(x, name: str) -> None:
    """Validate that a value is a non-bool Python int."""
    _require_type(x, int, f"{name} must be an int; got {type(x)}.")
    _require(not isinstance(x, bool), f"{name} must be an int (not bool).")


def _require_positive_int(x, name: str) -> None:
    """Validate that an int is strictly positive."""
    _require_int(x, name)
    _require(x > 0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_int(x, name: str) -> None:
    """Validate that an int is non-negative."""
    _require_int(x, name)
    _require(x >= 0, f"{name} must be >= 0; got {x}.")


def _require_float(x, name: str) -> None:
    """Validate that a value is a non-bool Python float."""
    _require_type(x, float, f"{name} must be a float; got {type(x)}.")
    _require(not isinstance(x, bool), f"{name} must be a float (not bool).")


def _require_finite_float(x, name: str) -> None:
    """Validate that a float is finite."""
    _require_float(x, name)
    _require(
        x == x and x not in (float("inf"), float("-inf")),
        f"{name} must be finite; got {x}.",
    )


def _require_positive_float(x, name: str) -> None:
    """Validate that a float is finite and strictly positive."""
    _require_finite_float(x, name)
    _require(x > 0.0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_float(x, name: str) -> None:
    """Validate that a float is finite and non-negative."""
    _require_finite_float(x, name)
    _require(x >= 0.0, f"{name} must be >= 0; got {x}.")


def _require_open_unit_float(x, name: str) -> None:
    """Validate that a float lies in the open unit interval."""
    _require_finite_float(x, name)
    _require(0.0 < x < 1.0, f"{name} must satisfy 0 < {name} < 1; got {x}.")


def _require_bool(x, name: str) -> None:
    """Validate that a value is a Python bool."""
    _require_type(x, bool, f"{name} must be a bool; got {type(x)}.")


def _require_tensor_rank(x: tf.Tensor, rank: int, name: str) -> None:
    """Validate the rank of a tensor input."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got rank {x.shape.rank}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that an input is a tf.float64 tensor."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype == tf.float64, f"{name} must be tf.float64; got {x.dtype}.")


def _require_integer_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that an input is an integer tensor."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype.is_integer, f"{name} must have integer dtype; got {x.dtype}.")


def _require_scalar_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that an input is a scalar tf.float64 tensor."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 0, name)


def _require_vector_float64_tensor(x: tf.Tensor, length: int, name: str) -> None:
    """Validate a length-fixed 1D tf.float64 tensor."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 1, name)
    _require(
        x.shape[0] is not None,
        f"{name} must have static length {length}; got shape {x.shape}.",
    )
    _require(
        x.shape[0] == length,
        f"{name} must have shape ({length},); got {x.shape}.",
    )


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are finite."""
    ok = bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
    _require(ok, f"{name} must be finite (no NaN/inf).")


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are non-negative."""
    ok = bool(tf.reduce_all(x >= tf.zeros((), dtype=x.dtype)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_all_positive(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are strictly positive."""
    ok = bool(tf.reduce_all(x > tf.zeros((), dtype=x.dtype)).numpy())
    _require(ok, f"{name} must be strictly positive.")


def _require_all_binary(x: tf.Tensor, name: str) -> None:
    """Validate that an integer tensor contains only 0/1 entries."""
    ok = bool(tf.reduce_all(tf.logical_or(tf.equal(x, 0), tf.equal(x, 1))).numpy())
    _require(ok, f"{name} must contain only 0/1 values.")


def _require_all_in_open_unit(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries lie in the open unit interval."""
    zero = tf.zeros((), dtype=x.dtype)
    one = tf.ones((), dtype=x.dtype)
    ok = bool(tf.reduce_all(tf.logical_and(x > zero, x < one)).numpy())
    _require(ok, f"{name} must lie in the open unit interval entrywise.")


def _require_prob_vector(x: tf.Tensor, name: str) -> None:
    """Validate a probability vector."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 1, name)
    _require(
        x.shape[0] is not None and x.shape[0] > 0,
        f"{name} must have static length > 0; got shape {x.shape}.",
    )
    _require_all_finite(x, name)
    _require_all_nonnegative(x, name)
    total = float(tf.reduce_sum(x).numpy())
    _require(
        abs(total - 1.0) <= ATOL_PROB,
        f"{name} must sum to 1; got sum={total}.",
    )


def _require_row_stochastic_matrix(x: tf.Tensor, name: str) -> None:
    """Validate that the last axis forms row-stochastic probabilities."""
    _require_float64_tensor(x, name)
    _require_all_finite(x, name)
    _require_all_nonnegative(x, name)
    row_sums = tf.reduce_sum(x, axis=-1)
    ok = bool(tf.reduce_all(tf.abs(row_sums - 1.0) <= ATOL_PROB).numpy())
    _require(ok, f"{name} must be row-stochastic along the last axis.")


def _require_seed_input(seed: tf.Tensor, name: str) -> None:
    """Validate a stateless RNG seed tensor of shape (2,)."""
    _require(tf.is_tensor(seed), f"{name} must be a tf.Tensor.")
    _require(
        seed.dtype.is_integer, f"{name} must have integer dtype; got {seed.dtype}."
    )
    _require(
        seed.shape.rank == 1, f"{name} must have rank 1; got rank {seed.shape.rank}."
    )
    _require(seed.shape[0] == 2, f"{name} must have shape (2,); got {seed.shape}.")
    ok = bool(tf.reduce_all(seed >= tf.constant(0, dtype=seed.dtype)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_inventory_maps(inventory_maps, I: int) -> None:
    """Validate the precomputed inventory maps tuple."""
    _require_type(
        inventory_maps,
        tuple,
        f"inventory_maps must be a tuple; got {type(inventory_maps)}.",
    )
    _require(len(inventory_maps) == 5, "inventory_maps must contain 5 tensors.")

    I_vals, stockout_mask, at_cap_mask, idx_down, idx_up = inventory_maps

    _require_integer_tensor(I_vals, "inventory_maps[0] (I_vals)")
    _require_tensor_rank(I_vals, 1, "inventory_maps[0] (I_vals)")
    _require(
        I_vals.shape[0] == I, f"I_vals must have shape ({I},); got {I_vals.shape}."
    )

    _require_float64_tensor(stockout_mask, "inventory_maps[1] (stockout_mask)")
    _require_tensor_rank(stockout_mask, 1, "inventory_maps[1] (stockout_mask)")
    _require(
        stockout_mask.shape[0] == I,
        f"stockout_mask must have shape ({I},); got {stockout_mask.shape}.",
    )

    _require_float64_tensor(at_cap_mask, "inventory_maps[2] (at_cap_mask)")
    _require_tensor_rank(at_cap_mask, 1, "inventory_maps[2] (at_cap_mask)")
    _require(
        at_cap_mask.shape[0] == I,
        f"at_cap_mask must have shape ({I},); got {at_cap_mask.shape}.",
    )

    _require_integer_tensor(idx_down, "inventory_maps[3] (idx_down)")
    _require_tensor_rank(idx_down, 1, "inventory_maps[3] (idx_down)")
    _require(
        idx_down.shape[0] == I,
        f"idx_down must have shape ({I},); got {idx_down.shape}.",
    )

    _require_integer_tensor(idx_up, "inventory_maps[4] (idx_up)")
    _require_tensor_rank(idx_up, 1, "inventory_maps[4] (idx_up)")
    _require(
        idx_up.shape[0] == I, f"idx_up must have shape ({I},); got {idx_up.shape}."
    )


def observed_data_validate_input(
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    inventory_maps,
    pi_I0: tf.Tensor,
) -> None:
    """Validate the observed data and fixed tensors for the stockpiling chain."""
    _require_integer_tensor(a_mnjt, "a_mnjt")
    _require_integer_tensor(s_mjt, "s_mjt")
    _require_float64_tensor(u_mj, "u_mj")
    _require_float64_tensor(P_price_mj, "P_price_mj")
    _require_float64_tensor(price_vals_mj, "price_vals_mj")
    _require_float64_tensor(lambda_mn, "lambda_mn")
    _require_scalar_float64_tensor(waste_cost, "waste_cost")
    _require_prob_vector(pi_I0, "pi_I0")

    _require_tensor_rank(a_mnjt, 4, "a_mnjt")
    _require_tensor_rank(s_mjt, 3, "s_mjt")
    _require_tensor_rank(u_mj, 2, "u_mj")
    _require_tensor_rank(P_price_mj, 4, "P_price_mj")
    _require_tensor_rank(price_vals_mj, 3, "price_vals_mj")
    _require_tensor_rank(lambda_mn, 2, "lambda_mn")

    _require(
        all(dim is not None for dim in a_mnjt.shape),
        f"a_mnjt must have fully static shape; got {a_mnjt.shape}.",
    )
    _require(
        all(dim is not None for dim in s_mjt.shape),
        f"s_mjt must have fully static shape; got {s_mjt.shape}.",
    )
    _require(
        all(dim is not None for dim in u_mj.shape),
        f"u_mj must have fully static shape; got {u_mj.shape}.",
    )
    _require(
        all(dim is not None for dim in P_price_mj.shape),
        f"P_price_mj must have fully static shape; got {P_price_mj.shape}.",
    )
    _require(
        all(dim is not None for dim in price_vals_mj.shape),
        f"price_vals_mj must have fully static shape; got {price_vals_mj.shape}.",
    )
    _require(
        all(dim is not None for dim in lambda_mn.shape),
        f"lambda_mn must have fully static shape; got {lambda_mn.shape}.",
    )

    M, N, J, T = a_mnjt.shape
    M_s, J_s, T_s = s_mjt.shape
    M_u, J_u = u_mj.shape
    M_p, J_p, S_p, S_p2 = P_price_mj.shape
    M_v, J_v, S_v = price_vals_mj.shape
    M_l, N_l = lambda_mn.shape
    I = pi_I0.shape[0]

    _require(
        M > 0 and N > 0 and J > 0 and T > 0,
        f"a_mnjt must have positive dimensions; got {a_mnjt.shape}.",
    )
    _require(
        M_s == M and J_s == J and T_s == T,
        f"s_mjt must have shape (M,J,T)=({M},{J},{T}); got {s_mjt.shape}.",
    )
    _require(
        M_u == M and J_u == J,
        f"u_mj must have shape (M,J)=({M},{J}); got {u_mj.shape}.",
    )
    _require(
        M_p == M and J_p == J,
        f"P_price_mj must have leading shape (M,J)=({M},{J}); got {P_price_mj.shape[:2]}.",
    )
    _require(
        S_p == S_p2, f"P_price_mj must have shape (M,J,S,S); got {P_price_mj.shape}."
    )
    _require(S_p >= 2, f"P_price_mj must have S >= 2; got S={S_p}.")
    _require(
        M_v == M and J_v == J and S_v == S_p,
        f"price_vals_mj must have shape (M,J,S)=({M},{J},{S_p}); got {price_vals_mj.shape}.",
    )
    _require(
        M_l == M and N_l == N,
        f"lambda_mn must have shape (M,N)=({M},{N}); got {lambda_mn.shape}.",
    )
    _require(I > 0, f"pi_I0 must have length > 0; got length {I}.")

    _require_inventory_maps(inventory_maps, I)

    _require_all_binary(a_mnjt, "a_mnjt")
    _require_all_finite(u_mj, "u_mj")
    _require_all_finite(P_price_mj, "P_price_mj")
    _require_all_finite(price_vals_mj, "price_vals_mj")
    _require_all_finite(lambda_mn, "lambda_mn")
    _require_all_finite(pi_I0, "pi_I0")
    _require_all_finite(waste_cost, "waste_cost")

    _require_row_stochastic_matrix(P_price_mj, "P_price_mj")
    _require_all_positive(price_vals_mj, "price_vals_mj")
    _require_all_in_open_unit(lambda_mn, "lambda_mn")

    s_ok = bool(
        tf.reduce_all(
            tf.logical_and(
                s_mjt >= tf.constant(0, dtype=s_mjt.dtype),
                s_mjt < tf.constant(S_p, dtype=s_mjt.dtype),
            )
        ).numpy()
    )
    _require(s_ok, f"s_mjt must take integer values in [0, {S_p - 1}].")


def posterior_validate_input(posterior_config) -> None:
    """Validate the posterior config required to construct StockpilingPosteriorTF."""
    required = [
        "tol",
        "max_iter",
        "eps",
        "sigma_z_beta",
        "sigma_z_alpha",
        "sigma_z_v",
        "sigma_z_fc",
        "sigma_z_u_scale",
        "fix_u_scale",
        "fixed_z_u_scale",
    ]
    missing = [name for name in required if not hasattr(posterior_config, name)]
    _require(not missing, "posterior_config missing fields: " + ", ".join(missing))

    _require_nonnegative_float(posterior_config.tol, "posterior_config.tol")
    _require_positive_int(posterior_config.max_iter, "posterior_config.max_iter")

    _require_positive_float(posterior_config.eps, "posterior_config.eps")
    _require(
        posterior_config.eps < 0.5,
        f"posterior_config.eps must satisfy 0 < eps < 0.5; got {posterior_config.eps}.",
    )

    _require_positive_float(
        posterior_config.sigma_z_beta, "posterior_config.sigma_z_beta"
    )
    _require_positive_float(
        posterior_config.sigma_z_alpha, "posterior_config.sigma_z_alpha"
    )
    _require_positive_float(posterior_config.sigma_z_v, "posterior_config.sigma_z_v")
    _require_positive_float(posterior_config.sigma_z_fc, "posterior_config.sigma_z_fc")
    _require_positive_float(
        posterior_config.sigma_z_u_scale, "posterior_config.sigma_z_u_scale"
    )

    _require_bool(posterior_config.fix_u_scale, "posterior_config.fix_u_scale")
    _require_finite_float(
        posterior_config.fixed_z_u_scale, "posterior_config.fixed_z_u_scale"
    )


def sampler_validate_input(
    sampler_config,
    M: int,
    J: int,
) -> None:
    """Validate the compiled-chain and tuning config for the stockpiling sampler."""
    required = [
        "num_results",
        "num_burnin_steps",
        "chunk_size",
        "k_beta",
        "k_alpha",
        "k_v",
        "k_fc",
        "k_u_scale",
        "pilot_num_steps",
        "target_accept_low",
        "target_accept_high",
        "grow_factor",
        "shrink_factor",
        "max_tuning_rounds",
    ]
    missing = [name for name in required if not hasattr(sampler_config, name)]
    _require(not missing, "sampler_config missing fields: " + ", ".join(missing))

    _require_positive_int(sampler_config.num_results, "sampler_config.num_results")
    _require_nonnegative_int(
        sampler_config.num_burnin_steps, "sampler_config.num_burnin_steps"
    )
    _require_positive_int(sampler_config.chunk_size, "sampler_config.chunk_size")
    _require_positive_int(
        sampler_config.pilot_num_steps, "sampler_config.pilot_num_steps"
    )
    _require_positive_int(
        sampler_config.max_tuning_rounds, "sampler_config.max_tuning_rounds"
    )

    _require_open_unit_float(
        sampler_config.target_accept_low, "sampler_config.target_accept_low"
    )
    _require_open_unit_float(
        sampler_config.target_accept_high, "sampler_config.target_accept_high"
    )
    _require(
        sampler_config.target_accept_low < sampler_config.target_accept_high,
        "sampler_config.target_accept_low must be < sampler_config.target_accept_high.",
    )

    _require_positive_float(sampler_config.grow_factor, "sampler_config.grow_factor")
    _require(
        sampler_config.grow_factor > 1.0,
        f"sampler_config.grow_factor must be > 1; got {sampler_config.grow_factor}.",
    )
    _require_open_unit_float(
        sampler_config.shrink_factor, "sampler_config.shrink_factor"
    )

    _require_scalar_float64_tensor(sampler_config.k_beta, "sampler_config.k_beta")
    _require_all_finite(sampler_config.k_beta, "sampler_config.k_beta")
    _require_all_positive(sampler_config.k_beta, "sampler_config.k_beta")

    _require_vector_float64_tensor(sampler_config.k_alpha, J, "sampler_config.k_alpha")
    _require_all_finite(sampler_config.k_alpha, "sampler_config.k_alpha")
    _require_all_positive(sampler_config.k_alpha, "sampler_config.k_alpha")

    _require_vector_float64_tensor(sampler_config.k_v, J, "sampler_config.k_v")
    _require_all_finite(sampler_config.k_v, "sampler_config.k_v")
    _require_all_positive(sampler_config.k_v, "sampler_config.k_v")

    _require_vector_float64_tensor(sampler_config.k_fc, J, "sampler_config.k_fc")
    _require_all_finite(sampler_config.k_fc, "sampler_config.k_fc")
    _require_all_positive(sampler_config.k_fc, "sampler_config.k_fc")

    _require_vector_float64_tensor(
        sampler_config.k_u_scale, M, "sampler_config.k_u_scale"
    )
    _require_all_finite(sampler_config.k_u_scale, "sampler_config.k_u_scale")
    _require_all_positive(sampler_config.k_u_scale, "sampler_config.k_u_scale")


def init_state_validate_input(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    M: int,
    J: int,
) -> None:
    """Validate the external unconstrained initial chain state."""
    _require_scalar_float64_tensor(z_beta, "z_beta")
    _require_all_finite(z_beta, "z_beta")

    _require_vector_float64_tensor(z_alpha, J, "z_alpha")
    _require_all_finite(z_alpha, "z_alpha")

    _require_vector_float64_tensor(z_v, J, "z_v")
    _require_all_finite(z_v, "z_v")

    _require_vector_float64_tensor(z_fc, J, "z_fc")
    _require_all_finite(z_fc, "z_fc")

    _require_vector_float64_tensor(z_u_scale, M, "z_u_scale")
    _require_all_finite(z_u_scale, "z_u_scale")


def seed_validate_input(seed: tf.Tensor) -> None:
    """Validate the external stateless seed for the stockpiling chain."""
    _require_seed_input(seed, "seed")


def run_chain_validate_input(
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    inventory_maps,
    pi_I0: tf.Tensor,
    posterior_config,
    sampler_config,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    seed: tf.Tensor,
) -> None:
    """Validate all external inputs required to run the stockpiling chain."""
    observed_data_validate_input(
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        u_mj=u_mj,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        inventory_maps=inventory_maps,
        pi_I0=pi_I0,
    )
    posterior_validate_input(posterior_config)

    M = a_mnjt.shape[0]
    J = a_mnjt.shape[2]
    _require(
        M is not None and J is not None, "a_mnjt must have static M and J dimensions."
    )

    sampler_validate_input(sampler_config=sampler_config, M=M, J=J)
    init_state_validate_input(
        z_beta=z_beta,
        z_alpha=z_alpha,
        z_v=z_v,
        z_fc=z_fc,
        z_u_scale=z_u_scale,
        M=M,
        J=J,
    )
    seed_validate_input(seed)
