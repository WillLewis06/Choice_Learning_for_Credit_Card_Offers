"""
bonus2_input_validation.py

Validate external inputs for the Bonus Q2 codebase.

This module supports two validation layers:

1) DGP / raw panel validation
   - validate_bonus2_dgp_inputs(...)
   - validate_bonus2_panel(...)

2) Refactored estimator-chain validation
   - preprocessing_validate_input(...)
   - observed_data_validate_input(...)
   - posterior_validate_input(...)
   - sampler_validate_input(...)
   - seed_validate_input(...)
   - run_chain_validate_input(...)

The DGP validators are NumPy/list based.
The refactored chain validators are TensorFlow based.

This module performs validation only. It does not apply defaults.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf


def _require(cond: bool, msg: str) -> None:
    """Raise ValueError when a validation condition fails."""
    if not cond:
        raise ValueError(msg)


def _require_type(x, t, msg: str) -> None:
    """Raise TypeError when a value is not of the required type."""
    if not isinstance(x, t):
        raise TypeError(msg)


def _require_int(x, name: str) -> None:
    """Validate that a value is a non-bool integer scalar."""
    _require_type(x, (int, np.integer), f"{name} must be an int; got {type(x)}.")
    _require(not isinstance(x, bool), f"{name} must be an int (not bool).")


def _require_positive_int(x, name: str) -> None:
    """Validate that an int is strictly positive."""
    _require_int(x, name)
    _require(int(x) > 0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_int(x, name: str) -> None:
    """Validate that an int is non-negative."""
    _require_int(x, name)
    _require(int(x) >= 0, f"{name} must be >= 0; got {x}.")


def _require_float(x, name: str) -> None:
    """Validate that a value is a non-bool floating scalar."""
    _require_type(
        x,
        (float, np.floating),
        f"{name} must be a float; got {type(x)}.",
    )
    _require(not isinstance(x, bool), f"{name} must be a float (not bool).")


def _require_finite_float(x, name: str) -> None:
    """Validate that a float is finite."""
    _require_float(x, name)
    _require(np.isfinite(float(x)), f"{name} must be finite; got {x}.")


def _require_positive_float(x, name: str) -> None:
    """Validate that a float is finite and strictly positive."""
    _require_finite_float(x, name)
    _require(float(x) > 0.0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_float(x, name: str) -> None:
    """Validate that a float is finite and non-negative."""
    _require_finite_float(x, name)
    _require(float(x) >= 0.0, f"{name} must be >= 0; got {x}.")


def _require_open_unit_float(x, name: str) -> None:
    """Validate that a float lies in the open unit interval."""
    _require_finite_float(x, name)
    _require(float(x) > 0.0, f"{name} must be > 0; got {x}.")
    _require(float(x) < 1.0, f"{name} must be < 1; got {x}.")


def _require_gt_one_float(x, name: str) -> None:
    """Validate that a float is greater than one."""
    _require_finite_float(x, name)
    _require(float(x) > 1.0, f"{name} must be > 1; got {x}.")


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
    _require(
        x.dtype.is_integer,
        f"{name} must have integer dtype; got {x.dtype}.",
    )


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are finite."""
    ok = bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
    _require(ok, f"{name} must be finite (no NaN/inf).")


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are non-negative."""
    ok = bool(tf.reduce_all(x >= tf.cast(0, x.dtype)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_all_binary_int(x: tf.Tensor, name: str) -> None:
    """Validate that an integer tensor contains only 0/1 values."""
    zero = tf.cast(0, x.dtype)
    one = tf.cast(1, x.dtype)
    ok = bool(tf.reduce_all((x == zero) | (x == one)).numpy())
    _require(ok, f"{name} must contain only 0/1 values.")


def _require_static_shape_known(x: tf.Tensor, name: str) -> None:
    """Validate that all dimensions of a tensor are statically known."""
    _require(
        all(dim is not None for dim in x.shape),
        f"{name} must have fully static shape; got {x.shape}.",
    )


def _require_positive_static_dim(dim: int | None, name: str) -> None:
    """Validate that a static tensor dimension is known and strictly positive."""
    _require(dim is not None, f"{name} must be statically known.")
    _require(dim > 0, f"{name} must be > 0; got {dim}.")


def _require_seed_input(seed: tf.Tensor, name: str) -> None:
    """Validate a stateless RNG seed tensor of shape (2,)."""
    _require(tf.is_tensor(seed), f"{name} must be a tf.Tensor.")
    _require(
        seed.dtype.is_integer,
        f"{name} must have integer dtype; got {seed.dtype}.",
    )
    _require(
        seed.shape.rank == 1,
        f"{name} must have rank 1; got rank {seed.shape.rank}.",
    )
    _require(
        seed.shape[0] == 2,
        f"{name} must have shape (2,); got {seed.shape}.",
    )
    ok = bool(tf.reduce_all(seed >= tf.cast(0, seed.dtype)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_choice_codes(y_mit: tf.Tensor, num_products: int, name: str) -> None:
    """Validate that choices are coded as 0 for outside and 1..J for inside."""
    y_min = int(tf.reduce_min(y_mit).numpy())
    y_max = int(tf.reduce_max(y_mit).numpy())
    _require(
        y_min >= 0,
        f"{name} must have min >= 0; got min={y_min}.",
    )
    _require(
        y_max <= num_products,
        f"{name} must have max <= J={num_products}; got max={y_max}.",
    )


def _require_neighbors_m(
    neighbors_m,
    num_markets: int,
    num_consumers: int,
    name: str,
) -> None:
    """Validate the within-market neighbor-list structure."""
    _require_type(
        neighbors_m,
        (list, tuple),
        f"{name} must be a list/tuple of length M={num_markets}; got {type(neighbors_m)}.",
    )
    _require(
        len(neighbors_m) == num_markets,
        f"{name} must have length M={num_markets}; got {len(neighbors_m)}.",
    )

    for m, neighbors_i in enumerate(neighbors_m):
        _require_type(
            neighbors_i,
            (list, tuple),
            f"{name}[{m}] must be a list/tuple of length N={num_consumers}; got {type(neighbors_i)}.",
        )
        _require(
            len(neighbors_i) == num_consumers,
            f"{name}[{m}] must have length N={num_consumers}; got {len(neighbors_i)}.",
        )

        for i, nbrs in enumerate(neighbors_i):
            _require_type(
                nbrs,
                (list, tuple),
                f"{name}[{m}][{i}] must be a list/tuple of neighbor indices; got {type(nbrs)}.",
            )

            seen: set[int] = set()
            for k in nbrs:
                _require_int(k, f"{name}[{m}][{i}] neighbor")
                k_int = int(k)
                _require(
                    0 <= k_int < num_consumers,
                    f"{name}[{m}][{i}] neighbor index must lie in [0,{num_consumers - 1}]; got {k_int}.",
                )
                _require(
                    k_int != i,
                    f"{name}[{m}][{i}] must not contain self-edge {i}.",
                )
                _require(
                    k_int not in seen,
                    f"{name}[{m}][{i}] contains duplicate neighbor index {k_int}.",
                )
                seen.add(k_int)


def _as_numpy_array(x, name: str, dtype=None) -> np.ndarray:
    """Convert a raw input to a NumPy array for DGP/panel validation."""
    try:
        return np.asarray(x, dtype=dtype)
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to a NumPy array.") from exc


def _require_numpy_rank(x: np.ndarray, rank: int, name: str) -> None:
    """Validate NumPy array rank."""
    _require(
        x.ndim == rank,
        f"{name} must have rank {rank}; got rank {x.ndim}.",
    )


def _require_numpy_finite(x: np.ndarray, name: str) -> None:
    """Validate that a NumPy array contains only finite values."""
    _require(np.all(np.isfinite(x)), f"{name} must be finite (no NaN/inf).")


def _require_numpy_nonnegative(x: np.ndarray, name: str) -> None:
    """Validate that a NumPy array is elementwise non-negative."""
    _require(np.all(x >= 0), f"{name} must be non-negative.")


def _require_numpy_integer_array(x: np.ndarray, name: str) -> None:
    """Validate that a NumPy array has integer dtype."""
    _require(
        np.issubdtype(x.dtype, np.integer),
        f"{name} must have integer dtype; got {x.dtype}.",
    )


def _require_numpy_binary_array(x: np.ndarray, name: str) -> None:
    """Validate that a NumPy integer array contains only 0/1 values."""
    _require_numpy_integer_array(x, name)
    _require(
        np.all((x == 0) | (x == 1)),
        f"{name} must contain only 0/1 values.",
    )


def _require_choice_codes_np(y_mit: np.ndarray, num_products: int, name: str) -> None:
    """Validate NumPy-coded choices with outside=0 and inside=1..J."""
    y_min = int(np.min(y_mit))
    y_max = int(np.max(y_mit))
    _require(
        y_min >= 0,
        f"{name} must have min >= 0; got min={y_min}.",
    )
    _require(
        y_max <= num_products,
        f"{name} must have max <= J={num_products}; got max={y_max}.",
    )


def validate_bonus2_dgp_inputs(
    delta_mj,
    N: int,
    T: int,
    avg_friends: float,
    params_true: dict,
    decay: float,
    seed: int,
    season_period: int,
    friends_sd: float,
    K: int,
    lookback: int,
) -> None:
    """Validate the raw inputs to simulate_bonus2_dgp(...)."""
    delta = _as_numpy_array(delta_mj, "delta_mj", dtype=np.float64)
    _require_numpy_rank(delta, 2, "delta_mj")
    _require_numpy_finite(delta, "delta_mj")

    M, J = delta.shape
    _require(M > 0, f"delta_mj.shape[0] must be > 0; got {M}.")
    _require(J > 0, f"delta_mj.shape[1] must be > 0; got {J}.")

    _require_positive_int(N, "N")
    _require_positive_int(T, "T")
    _require_nonnegative_float(avg_friends, "avg_friends")
    _require_open_unit_float(decay, "decay")
    _require_nonnegative_int(seed, "seed")
    _require_positive_int(season_period, "season_period")
    _require_nonnegative_float(friends_sd, "friends_sd")
    _require_positive_int(K, "K")
    _require_positive_int(lookback, "lookback")

    _require_type(
        params_true, dict, f"params_true must be a dict; got {type(params_true)}."
    )
    required_keys = (
        "habit_mean",
        "habit_sd",
        "peer_mean",
        "peer_sd",
        "mktprod_sd",
        "weekend_prod_sd",
        "season_mkt_sd",
    )
    missing = [k for k in required_keys if k not in params_true]
    _require(not missing, "params_true missing keys: " + ", ".join(missing))

    _require_finite_float(params_true["habit_mean"], "params_true['habit_mean']")
    _require_nonnegative_float(params_true["habit_sd"], "params_true['habit_sd']")
    _require_finite_float(params_true["peer_mean"], "params_true['peer_mean']")
    _require_nonnegative_float(params_true["peer_sd"], "params_true['peer_sd']")
    _require_nonnegative_float(params_true["mktprod_sd"], "params_true['mktprod_sd']")
    _require_nonnegative_float(
        params_true["weekend_prod_sd"],
        "params_true['weekend_prod_sd']",
    )
    _require_nonnegative_float(
        params_true["season_mkt_sd"],
        "params_true['season_mkt_sd']",
    )


def validate_bonus2_panel(panel: dict) -> None:
    """Validate the raw panel dict produced by simulate_bonus2_dgp(...)."""
    _require_type(panel, dict, f"panel must be a dict; got {type(panel)}.")

    required_keys = (
        "y_mit",
        "delta_mj",
        "is_weekend_t",
        "neighbors_m",
        "lookback",
        "season_sin_kt",
        "season_cos_kt",
        "decay",
    )
    missing = [k for k in required_keys if k not in panel]
    _require(not missing, "panel missing keys: " + ", ".join(missing))

    y_mit = _as_numpy_array(panel["y_mit"], "panel['y_mit']")
    delta_mj = _as_numpy_array(panel["delta_mj"], "panel['delta_mj']", dtype=np.float64)
    is_weekend_t = _as_numpy_array(panel["is_weekend_t"], "panel['is_weekend_t']")
    season_sin_kt = _as_numpy_array(
        panel["season_sin_kt"],
        "panel['season_sin_kt']",
        dtype=np.float64,
    )
    season_cos_kt = _as_numpy_array(
        panel["season_cos_kt"],
        "panel['season_cos_kt']",
        dtype=np.float64,
    )

    _require_numpy_rank(y_mit, 3, "panel['y_mit']")
    _require_numpy_rank(delta_mj, 2, "panel['delta_mj']")
    _require_numpy_rank(is_weekend_t, 1, "panel['is_weekend_t']")
    _require_numpy_rank(season_sin_kt, 2, "panel['season_sin_kt']")
    _require_numpy_rank(season_cos_kt, 2, "panel['season_cos_kt']")

    _require_numpy_integer_array(y_mit, "panel['y_mit']")
    _require_numpy_integer_array(is_weekend_t, "panel['is_weekend_t']")
    _require_numpy_finite(delta_mj, "panel['delta_mj']")
    _require_numpy_finite(season_sin_kt, "panel['season_sin_kt']")
    _require_numpy_finite(season_cos_kt, "panel['season_cos_kt']")
    _require_numpy_binary_array(is_weekend_t, "panel['is_weekend_t']")

    M, N, T = y_mit.shape
    M_delta, J = delta_mj.shape
    T_weekend = is_weekend_t.shape[0]
    K_sin, T_sin = season_sin_kt.shape
    K_cos, T_cos = season_cos_kt.shape

    _require(M > 0, f"panel['y_mit'].shape[0] must be > 0; got {M}.")
    _require(N > 0, f"panel['y_mit'].shape[1] must be > 0; got {N}.")
    _require(T > 0, f"panel['y_mit'].shape[2] must be > 0; got {T}.")
    _require(
        M_delta == M, f"panel['delta_mj'] must have M={M}; got shape {delta_mj.shape}."
    )
    _require(J > 0, f"panel['delta_mj'].shape[1] must be > 0; got {J}.")
    _require(
        T_weekend == T,
        f"panel['is_weekend_t'] must have shape (T,) with T={T}; got {is_weekend_t.shape}.",
    )
    _require(
        season_sin_kt.shape == season_cos_kt.shape,
        "panel['season_sin_kt'] and panel['season_cos_kt'] must have the same shape; "
        f"got {season_sin_kt.shape} and {season_cos_kt.shape}.",
    )
    _require(K_sin > 0, f"panel['season_sin_kt'].shape[0] must be > 0; got {K_sin}.")
    _require(
        T_sin == T,
        f"panel['season_sin_kt'] must have shape (K, T) with T={T}; got {season_sin_kt.shape}.",
    )
    _require(
        T_cos == T,
        f"panel['season_cos_kt'] must have shape (K, T) with T={T}; got {season_cos_kt.shape}.",
    )

    _require_choice_codes_np(y_mit, num_products=J, name="panel['y_mit']")
    _require_neighbors_m(
        neighbors_m=panel["neighbors_m"],
        num_markets=M,
        num_consumers=N,
        name="panel['neighbors_m']",
    )
    _require_positive_int(panel["lookback"], "panel['lookback']")
    _require_open_unit_float(panel["decay"], "panel['decay']")


def preprocessing_validate_input(
    y_mit: tf.Tensor,
    neighbors_m,
    lookback: int,
    decay: float,
) -> None:
    """Validate inputs used to build deterministic states before MCMC."""
    _require_integer_tensor(y_mit, "y_mit")
    _require_tensor_rank(y_mit, 3, "y_mit")
    _require_static_shape_known(y_mit, "y_mit")

    M = y_mit.shape[0]
    N = y_mit.shape[1]
    T = y_mit.shape[2]

    _require_positive_static_dim(M, "y_mit.shape[0]")
    _require_positive_static_dim(N, "y_mit.shape[1]")
    _require_positive_static_dim(T, "y_mit.shape[2]")

    _require_positive_int(lookback, "lookback")
    _require_open_unit_float(decay, "decay")

    _require_neighbors_m(
        neighbors_m=neighbors_m,
        num_markets=M,
        num_consumers=N,
        name="neighbors_m",
    )

    y_min = int(tf.reduce_min(y_mit).numpy())
    _require(
        y_min >= 0,
        f"y_mit must have min >= 0; got min={y_min}.",
    )


def observed_data_validate_input(
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> None:
    """Validate the fixed observed tensors consumed by Bonus2PosteriorTF."""
    _require_integer_tensor(y_mit, "y_mit")
    _require_float64_tensor(delta_mj, "delta_mj")
    _require_integer_tensor(is_weekend_t, "is_weekend_t")
    _require_float64_tensor(season_sin_kt, "season_sin_kt")
    _require_float64_tensor(season_cos_kt, "season_cos_kt")
    _require_float64_tensor(h_mntj, "h_mntj")
    _require_float64_tensor(p_mntj, "p_mntj")

    _require_tensor_rank(y_mit, 3, "y_mit")
    _require_tensor_rank(delta_mj, 2, "delta_mj")
    _require_tensor_rank(is_weekend_t, 1, "is_weekend_t")
    _require_tensor_rank(season_sin_kt, 2, "season_sin_kt")
    _require_tensor_rank(season_cos_kt, 2, "season_cos_kt")
    _require_tensor_rank(h_mntj, 4, "h_mntj")
    _require_tensor_rank(p_mntj, 4, "p_mntj")

    _require_static_shape_known(y_mit, "y_mit")
    _require_static_shape_known(delta_mj, "delta_mj")
    _require_static_shape_known(is_weekend_t, "is_weekend_t")
    _require_static_shape_known(season_sin_kt, "season_sin_kt")
    _require_static_shape_known(season_cos_kt, "season_cos_kt")
    _require_static_shape_known(h_mntj, "h_mntj")
    _require_static_shape_known(p_mntj, "p_mntj")

    M = y_mit.shape[0]
    N = y_mit.shape[1]
    T = y_mit.shape[2]
    M_delta = delta_mj.shape[0]
    J = delta_mj.shape[1]
    T_weekend = is_weekend_t.shape[0]
    K_sin = season_sin_kt.shape[0]
    T_sin = season_sin_kt.shape[1]
    K_cos = season_cos_kt.shape[0]
    T_cos = season_cos_kt.shape[1]

    _require_positive_static_dim(M, "y_mit.shape[0]")
    _require_positive_static_dim(N, "y_mit.shape[1]")
    _require_positive_static_dim(T, "y_mit.shape[2]")
    _require_positive_static_dim(M_delta, "delta_mj.shape[0]")
    _require_positive_static_dim(J, "delta_mj.shape[1]")
    _require_positive_static_dim(T_weekend, "is_weekend_t.shape[0]")
    _require_positive_static_dim(K_sin, "season_sin_kt.shape[0]")
    _require_positive_static_dim(T_sin, "season_sin_kt.shape[1]")
    _require_positive_static_dim(K_cos, "season_cos_kt.shape[0]")
    _require_positive_static_dim(T_cos, "season_cos_kt.shape[1]")

    _require(
        M_delta == M,
        f"delta_mj must have shape (M, J) with M={M}; got {delta_mj.shape}.",
    )
    _require(
        T_weekend == T,
        f"is_weekend_t must have shape (T,) with T={T}; got {is_weekend_t.shape}.",
    )
    _require(
        season_sin_kt.shape == season_cos_kt.shape,
        "season_sin_kt and season_cos_kt must have the same shape; "
        f"got {season_sin_kt.shape} and {season_cos_kt.shape}.",
    )
    _require(
        T_sin == T,
        f"season_sin_kt must have shape (K, T) with T={T}; got {season_sin_kt.shape}.",
    )
    _require(
        T_cos == T,
        f"season_cos_kt must have shape (K, T) with T={T}; got {season_cos_kt.shape}.",
    )
    _require(
        h_mntj.shape == (M, N, T, J),
        f"h_mntj must have shape (M, N, T, J)=({M}, {N}, {T}, {J}); got {h_mntj.shape}.",
    )
    _require(
        p_mntj.shape == (M, N, T, J),
        f"p_mntj must have shape (M, N, T, J)=({M}, {N}, {T}, {J}); got {p_mntj.shape}.",
    )

    _require_all_finite(delta_mj, "delta_mj")
    _require_all_finite(season_sin_kt, "season_sin_kt")
    _require_all_finite(season_cos_kt, "season_cos_kt")
    _require_all_finite(h_mntj, "h_mntj")
    _require_all_finite(p_mntj, "p_mntj")

    _require_all_nonnegative(h_mntj, "h_mntj")
    _require_all_nonnegative(p_mntj, "p_mntj")
    _require_all_binary_int(is_weekend_t, "is_weekend_t")
    _require_choice_codes(y_mit=y_mit, num_products=J, name="y_mit")


def posterior_validate_input(posterior_config) -> None:
    """Validate the posterior config required to construct Bonus2PosteriorTF."""
    required = [
        "sigma_z_beta_intercept_j",
        "sigma_z_beta_habit_j",
        "sigma_z_beta_peer_j",
        "sigma_z_beta_weekend_jw",
        "sigma_z_a_m",
        "sigma_z_b_m",
    ]
    missing = [name for name in required if not hasattr(posterior_config, name)]
    _require(not missing, "posterior_config missing fields: " + ", ".join(missing))

    _require_positive_float(
        posterior_config.sigma_z_beta_intercept_j,
        "posterior_config.sigma_z_beta_intercept_j",
    )
    _require_positive_float(
        posterior_config.sigma_z_beta_habit_j,
        "posterior_config.sigma_z_beta_habit_j",
    )
    _require_positive_float(
        posterior_config.sigma_z_beta_peer_j,
        "posterior_config.sigma_z_beta_peer_j",
    )
    _require_positive_float(
        posterior_config.sigma_z_beta_weekend_jw,
        "posterior_config.sigma_z_beta_weekend_jw",
    )
    _require_positive_float(
        posterior_config.sigma_z_a_m,
        "posterior_config.sigma_z_a_m",
    )
    _require_positive_float(
        posterior_config.sigma_z_b_m,
        "posterior_config.sigma_z_b_m",
    )


def sampler_validate_input(sampler_config) -> None:
    """Validate the sampler and tuning config for the Bonus2 chain."""
    required = [
        "num_results",
        "num_burnin_steps",
        "chunk_size",
        "k_beta_intercept",
        "k_beta_habit",
        "k_beta_peer",
        "k_beta_weekend",
        "k_a",
        "k_b",
        "pilot_length",
        "target_low",
        "target_high",
        "max_rounds",
        "factor",
    ]
    missing = [name for name in required if not hasattr(sampler_config, name)]
    _require(not missing, "sampler_config missing fields: " + ", ".join(missing))

    _require_positive_int(sampler_config.num_results, "sampler_config.num_results")
    _require_nonnegative_int(
        sampler_config.num_burnin_steps,
        "sampler_config.num_burnin_steps",
    )
    _require_positive_int(sampler_config.chunk_size, "sampler_config.chunk_size")

    _require_positive_float(
        sampler_config.k_beta_intercept,
        "sampler_config.k_beta_intercept",
    )
    _require_positive_float(
        sampler_config.k_beta_habit,
        "sampler_config.k_beta_habit",
    )
    _require_positive_float(
        sampler_config.k_beta_peer,
        "sampler_config.k_beta_peer",
    )
    _require_positive_float(
        sampler_config.k_beta_weekend,
        "sampler_config.k_beta_weekend",
    )
    _require_positive_float(sampler_config.k_a, "sampler_config.k_a")
    _require_positive_float(sampler_config.k_b, "sampler_config.k_b")

    _require_positive_int(
        sampler_config.pilot_length,
        "sampler_config.pilot_length",
    )
    _require_open_unit_float(
        sampler_config.target_low,
        "sampler_config.target_low",
    )
    _require_open_unit_float(
        sampler_config.target_high,
        "sampler_config.target_high",
    )
    _require(
        sampler_config.target_low < sampler_config.target_high,
        "sampler_config.target_low must be < sampler_config.target_high; "
        f"got {sampler_config.target_low} >= {sampler_config.target_high}.",
    )
    _require_positive_int(
        sampler_config.max_rounds,
        "sampler_config.max_rounds",
    )
    _require_gt_one_float(sampler_config.factor, "sampler_config.factor")


def seed_validate_input(seed: tf.Tensor) -> None:
    """Validate the external chain seed tensor."""
    _require_seed_input(seed, "seed")


def run_chain_validate_input(
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
    posterior_config,
    sampler_config,
    seed: tf.Tensor,
) -> None:
    """Validate all external inputs required by the Bonus2 MCMC chain."""
    observed_data_validate_input(
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
    )
    posterior_validate_input(posterior_config)
    sampler_validate_input(sampler_config)
    seed_validate_input(seed)
