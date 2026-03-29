"""Input validation for the Ching-style stockpiling DGP and sampler."""

from __future__ import annotations

from numbers import Integral, Real
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


# =============================================================================
# DGP validation and normalization
# =============================================================================


def _as_np(x: Any, name: str, dtype: Any) -> np.ndarray:
    """Convert an external input to a NumPy array of the requested dtype."""
    try:
        return np.asarray(x, dtype=dtype)
    except Exception as exc:
        raise TypeError(f"{name} could not be converted to {dtype}.") from exc


def _require_ndim(x: np.ndarray, ndim: int, name: str) -> None:
    """Validate the rank of a NumPy array."""
    if x.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {x.shape}.")


def _require_shape(x: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    """Validate the exact shape of a NumPy array."""
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {x.shape}.")


def _require_finite_np(x: np.ndarray, name: str) -> None:
    """Validate that a NumPy array contains only finite entries."""
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite (no NaN/inf).")


def _validated_int_scalar(x: Any, name: str, min_value: int | None = None) -> int:
    """Validate and return an integer scalar."""
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


def _validated_real_scalar(
    x: Any,
    name: str,
    min_value: float | None = None,
    strict_lower: float | None = None,
) -> float:
    """Validate and return a finite real scalar."""
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
    if strict_lower is not None and value <= strict_lower:
        raise ValueError(f"{name} must be > {strict_lower}; got {value}.")
    return value


def _normalize_price_inputs(
    P_price_mj: Any,
    price_vals_mj: Any,
    M: int,
    J: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate and normalize price transitions and price levels."""
    P = _as_np(P_price_mj, "P_price_mj", np.float64)
    price_vals = _as_np(price_vals_mj, "price_vals_mj", np.float64)

    _require_ndim(P, 4, "P_price_mj")
    _require_ndim(price_vals, 3, "price_vals_mj")

    if P.shape[:2] != (M, J):
        raise ValueError(
            f"P_price_mj must have leading shape (M,J)=({M},{J}); got {P.shape[:2]}."
        )
    if price_vals.shape[:2] != (M, J):
        raise ValueError(
            "price_vals_mj must have leading shape "
            f"(M,J)=({M},{J}); got {price_vals.shape[:2]}."
        )

    S = int(P.shape[2])
    if P.shape[3] != S:
        raise ValueError(f"P_price_mj must have shape (M,J,S,S); got {P.shape}.")
    if S < 2:
        raise ValueError(f"P_price_mj must have S >= 2; got S={S}.")
    if price_vals.shape[2] != S:
        raise ValueError(
            f"price_vals_mj must have shape (M,J,S)=({M},{J},{S}); got {price_vals.shape}."
        )

    _require_finite_np(P, "P_price_mj")
    _require_finite_np(price_vals, "price_vals_mj")

    if np.any(P < 0.0):
        raise ValueError("P_price_mj must be non-negative.")
    if not np.all(np.abs(P.sum(axis=-1) - 1.0) <= ATOL_PROB):
        raise ValueError("P_price_mj must be row-stochastic along the last axis.")
    if np.any(price_vals <= 0.0):
        raise ValueError("price_vals_mj must be strictly positive.")

    return P, price_vals, S


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
    """Validate and normalize DGP inputs into canonical NumPy objects."""
    delta = _as_np(delta_true, "delta_true", np.float64)
    _require_ndim(delta, 1, "delta_true")
    _require_finite_np(delta, "delta_true")
    J = int(delta.shape[0])
    _require(J >= 1, f"delta_true must have length J >= 1; got J={J}.")

    E_bar = _as_np(E_bar_true, "E_bar_true", np.float64)
    _require_ndim(E_bar, 1, "E_bar_true")
    _require_finite_np(E_bar, "E_bar_true")
    M = int(E_bar.shape[0])
    _require(M >= 1, f"E_bar_true must have length M >= 1; got M={M}.")

    njt = _as_np(njt_true, "njt_true", np.float64)
    _require_ndim(njt, 2, "njt_true")
    _require_shape(njt, (M, J), "njt_true")
    _require_finite_np(njt, "njt_true")

    P, price_vals, S = _normalize_price_inputs(
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        M=M,
        J=J,
    )

    return {
        "seed": _validated_int_scalar(seed, "seed", min_value=0),
        "M": M,
        "N": _validated_int_scalar(N, "N", min_value=1),
        "J": J,
        "T": _validated_int_scalar(T, "T", min_value=1),
        "I_max": _validated_int_scalar(I_max, "I_max", min_value=0),
        "delta_true": delta,
        "E_bar_true": E_bar,
        "njt_true": njt,
        "P_price_mj": P,
        "price_vals_mj": price_vals,
        "waste_cost": _validated_real_scalar(
            waste_cost,
            "waste_cost",
            min_value=0.0,
        ),
        "tol": _validated_real_scalar(
            tol,
            "tol",
            min_value=0.0,
        ),
        "max_iter": _validated_int_scalar(max_iter, "max_iter", min_value=1),
        "S": S,
    }


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
    """Validate inputs to the stockpiling DGP generator."""
    normalize_stockpiling_dgp_inputs(
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


# =============================================================================
# Chain input validation
# =============================================================================


def _require_python_int(x: Any, name: str, min_value: int | None = None) -> int:
    """Validate and return a Python integer."""
    if isinstance(x, bool) or not isinstance(x, Integral):
        raise TypeError(f"{name} must be an int; got {type(x)}.")
    value = int(x)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {value}.")
    return value


def _require_python_float(
    x: Any,
    name: str,
    min_value: float | None = None,
    open_unit: bool = False,
) -> float:
    """Validate and return a finite Python real scalar."""
    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError(f"{name} must be a real scalar; got {type(x)}.")
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}; got {value}.")
    if open_unit and not (0.0 < value < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1; got {value}.")
    return value


def _require_tensor_rank(x: tf.Tensor, rank: int, name: str) -> None:
    """Validate the rank of a tensor."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got {x.shape.rank}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor has dtype tf.float64."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(
        x.dtype == tf.float64,
        f"{name} must have dtype tf.float64; got {x.dtype}.",
    )


def _require_integer_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that a tensor has integer dtype."""
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype.is_integer, f"{name} must have integer dtype; got {x.dtype}.")


def _require_scalar_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Validate a scalar tf.float64 tensor."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 0, name)


def _require_vector_float64_tensor(x: tf.Tensor, length: int, name: str) -> None:
    """Validate a length-fixed 1D tf.float64 tensor."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 1, name)
    _require(
        x.shape[0] == length,
        f"{name} must have shape ({length},); got {x.shape}.",
    )


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are finite."""
    _require(
        bool(tf.reduce_all(tf.math.is_finite(x)).numpy()),
        f"{name} must be finite (no NaN/inf).",
    )


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are non-negative."""
    _require(
        bool(tf.reduce_all(x >= tf.zeros((), dtype=x.dtype)).numpy()),
        f"{name} must be non-negative.",
    )


def _require_all_positive(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are strictly positive."""
    _require(
        bool(tf.reduce_all(x > tf.zeros((), dtype=x.dtype)).numpy()),
        f"{name} must be strictly positive.",
    )


def _require_all_binary(x: tf.Tensor, name: str) -> None:
    """Validate that an integer tensor contains only 0 and 1."""
    ok = tf.reduce_all(tf.logical_or(tf.equal(x, 0), tf.equal(x, 1)))
    _require(bool(ok.numpy()), f"{name} must contain only 0/1 values.")


def _require_all_in_open_unit(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries lie in the open unit interval."""
    zero = tf.zeros((), dtype=x.dtype)
    one = tf.ones((), dtype=x.dtype)
    ok = tf.reduce_all(tf.logical_and(x > zero, x < one))
    _require(
        bool(ok.numpy()),
        f"{name} must lie in the open unit interval entrywise.",
    )


def _require_prob_vector(x: tf.Tensor, name: str) -> None:
    """Validate a probability vector."""
    _require_float64_tensor(x, name)
    _require_tensor_rank(x, 1, name)
    _require(
        x.shape[0] is not None and x.shape[0] > 0,
        f"{name} must have static length > 0; got {x.shape}.",
    )
    _require_all_finite(x, name)
    _require_all_nonnegative(x, name)
    total = float(tf.reduce_sum(x).numpy())
    _require(abs(total - 1.0) <= ATOL_PROB, f"{name} must sum to 1; got sum={total}.")


def _require_row_stochastic_matrix(x: tf.Tensor, name: str) -> None:
    """Validate that the last axis is row-stochastic."""
    _require_float64_tensor(x, name)
    _require_all_finite(x, name)
    _require_all_nonnegative(x, name)
    row_sums = tf.reduce_sum(x, axis=-1)
    ok = tf.reduce_all(tf.abs(row_sums - 1.0) <= ATOL_PROB)
    _require(
        bool(ok.numpy()),
        f"{name} must be row-stochastic along the last axis.",
    )


def _require_seed_input(seed: tf.Tensor, name: str) -> None:
    """Validate a stateless RNG seed tensor of shape (2,)."""
    _require(tf.is_tensor(seed), f"{name} must be a tf.Tensor.")
    _require(
        seed.dtype.is_integer, f"{name} must have integer dtype; got {seed.dtype}."
    )
    _require(seed.shape.rank == 1, f"{name} must have rank 1; got {seed.shape.rank}.")
    _require(seed.shape[0] == 2, f"{name} must have shape (2,); got {seed.shape}.")
    ok = tf.reduce_all(seed >= tf.zeros((2,), dtype=seed.dtype))
    _require(bool(ok.numpy()), f"{name} must be non-negative.")


def _require_inventory_maps(inventory_maps: Any, I: int) -> None:
    """Validate the precomputed inventory maps tuple."""
    if not isinstance(inventory_maps, tuple):
        raise TypeError(f"inventory_maps must be a tuple; got {type(inventory_maps)}.")
    _require(len(inventory_maps) == 5, "inventory_maps must contain 5 tensors.")

    i_vals, stockout_mask, at_cap_mask, idx_down, idx_up = inventory_maps

    _require_integer_tensor(i_vals, "inventory_maps[0] (i_vals)")
    _require_tensor_rank(i_vals, 1, "inventory_maps[0] (i_vals)")
    _require(
        i_vals.shape[0] == I,
        f"i_vals must have shape ({I},); got {i_vals.shape}.",
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
        idx_up.shape[0] == I,
        f"idx_up must have shape ({I},); got {idx_up.shape}.",
    )


def observed_data_validate_input(
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    inventory_maps: Any,
) -> None:
    """Validate the observed panel and fixed tensors for the sampler."""
    _require_integer_tensor(a_mnjt, "a_mnjt")
    _require_integer_tensor(s_mjt, "s_mjt")
    _require_float64_tensor(u_mj, "u_mj")
    _require_float64_tensor(P_price_mj, "P_price_mj")
    _require_float64_tensor(price_vals_mj, "price_vals_mj")
    _require_float64_tensor(lambda_mn, "lambda_mn")
    _require_scalar_float64_tensor(waste_cost, "waste_cost")

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
    M_p, J_p, S, S2 = P_price_mj.shape
    M_v, J_v, S_v = price_vals_mj.shape
    M_l, N_l = lambda_mn.shape

    _require(
        M > 0 and N > 0 and J > 0 and T > 0,
        f"a_mnjt must have positive dimensions; got {a_mnjt.shape}.",
    )
    _require(
        (M_s, J_s, T_s) == (M, J, T),
        f"s_mjt must have shape (M,J,T)=({M},{J},{T}); got {s_mjt.shape}.",
    )
    _require(
        (M_u, J_u) == (M, J),
        f"u_mj must have shape (M,J)=({M},{J}); got {u_mj.shape}.",
    )
    _require(
        (M_p, J_p) == (M, J),
        f"P_price_mj must have leading shape (M,J)=({M},{J}); got {P_price_mj.shape[:2]}.",
    )
    _require(S == S2, f"P_price_mj must have shape (M,J,S,S); got {P_price_mj.shape}.")
    _require(S >= 2, f"P_price_mj must have S >= 2; got S={S}.")
    _require(
        (M_v, J_v, S_v) == (M, J, S),
        f"price_vals_mj must have shape (M,J,S)=({M},{J},{S}); got {price_vals_mj.shape}.",
    )
    _require(
        (M_l, N_l) == (M, N),
        f"lambda_mn must have shape (M,N)=({M},{N}); got {lambda_mn.shape}.",
    )

    _require(
        isinstance(inventory_maps, tuple) and len(inventory_maps) == 5,
        "inventory_maps must contain 5 tensors.",
    )
    i_vals = inventory_maps[0]
    _require_integer_tensor(i_vals, "inventory_maps[0] (i_vals)")
    _require_tensor_rank(i_vals, 1, "inventory_maps[0] (i_vals)")
    _require(
        i_vals.shape[0] is not None and i_vals.shape[0] > 0,
        f"inventory_maps[0] (i_vals) must have static length > 0; got {i_vals.shape}.",
    )
    I = i_vals.shape[0]
    _require_inventory_maps(inventory_maps, I)

    _require_all_binary(a_mnjt, "a_mnjt")
    _require_all_finite(u_mj, "u_mj")
    _require_all_finite(P_price_mj, "P_price_mj")
    _require_all_finite(price_vals_mj, "price_vals_mj")
    _require_all_finite(lambda_mn, "lambda_mn")
    _require_all_finite(waste_cost, "waste_cost")

    _require_row_stochastic_matrix(P_price_mj, "P_price_mj")
    _require_all_positive(price_vals_mj, "price_vals_mj")
    _require_all_in_open_unit(lambda_mn, "lambda_mn")

    s_ok = tf.reduce_all(
        tf.logical_and(
            s_mjt >= tf.zeros((), dtype=s_mjt.dtype),
            s_mjt < tf.constant(S, dtype=s_mjt.dtype),
        )
    )
    _require(bool(s_ok.numpy()), f"s_mjt must take integer values in [0, {S - 1}].")


def posterior_validate_input(posterior_config: Any) -> None:
    """Validate the posterior configuration."""
    required = [
        "tol",
        "max_iter",
        "eps",
        "sigma_z_beta",
        "sigma_z_alpha",
        "sigma_z_v",
        "sigma_z_fc",
        "sigma_z_u_scale",
    ]
    missing = [name for name in required if not hasattr(posterior_config, name)]
    _require(not missing, "posterior_config missing fields: " + ", ".join(missing))

    _require_python_float(
        posterior_config.tol,
        "posterior_config.tol",
        min_value=0.0,
    )
    _require_python_int(
        posterior_config.max_iter,
        "posterior_config.max_iter",
        min_value=1,
    )

    eps = _require_python_float(
        posterior_config.eps,
        "posterior_config.eps",
        min_value=0.0,
    )
    _require(
        0.0 < eps < 0.5,
        f"posterior_config.eps must satisfy 0 < eps < 0.5; got {eps}.",
    )

    _require_python_float(
        posterior_config.sigma_z_beta,
        "posterior_config.sigma_z_beta",
        min_value=0.0,
    )
    _require_python_float(
        posterior_config.sigma_z_alpha,
        "posterior_config.sigma_z_alpha",
        min_value=0.0,
    )
    _require_python_float(
        posterior_config.sigma_z_v,
        "posterior_config.sigma_z_v",
        min_value=0.0,
    )
    _require_python_float(
        posterior_config.sigma_z_fc,
        "posterior_config.sigma_z_fc",
        min_value=0.0,
    )
    _require_python_float(
        posterior_config.sigma_z_u_scale,
        "posterior_config.sigma_z_u_scale",
        min_value=0.0,
    )


def sampler_validate_input(sampler_config: Any, M: int, J: int) -> None:
    """Validate the sampler configuration."""
    required = [
        "num_results",
        "chunk_size",
        "k_beta",
        "k_alpha",
        "k_v",
        "k_fc",
        "k_u_scale",
    ]
    missing = [name for name in required if not hasattr(sampler_config, name)]
    _require(not missing, "sampler_config missing fields: " + ", ".join(missing))

    _require_python_int(
        sampler_config.num_results,
        "sampler_config.num_results",
        min_value=1,
    )
    _require_python_int(
        sampler_config.chunk_size,
        "sampler_config.chunk_size",
        min_value=1,
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
        sampler_config.k_u_scale,
        M,
        "sampler_config.k_u_scale",
    )
    _require_all_finite(sampler_config.k_u_scale, "sampler_config.k_u_scale")
    _require_all_nonnegative(sampler_config.k_u_scale, "sampler_config.k_u_scale")


def init_state_validate_input(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    M: int,
    J: int,
) -> None:
    """Validate the external unconstrained initial state."""
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
    """Validate the stateless RNG seed for the sampler."""
    _require_seed_input(seed, "seed")


def run_chain_validate_input(
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    inventory_maps: Any,
    posterior_config: Any,
    sampler_config: Any,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    seed: tf.Tensor,
) -> None:
    """Validate all external inputs required to run the sampler."""
    observed_data_validate_input(
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        u_mj=u_mj,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        inventory_maps=inventory_maps,
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
