"""
Input validation for the choice-learn + Lu shrinkage estimator.

Scope:
  - Validate external data tensors passed into the estimator (delta_cl, qjt, q0t).
  - Validate external configuration mappings (init_config, fit_config).

Policy:
  - Reject missing or invalid inputs outright (no defaults, no fallbacks).
  - Do not coerce or cast inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tensorflow as tf


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _require_type(x, types, name: str) -> None:
    if not isinstance(x, types):
        raise TypeError(f"{name} must be {types}, got {type(x)}")


def _require_mapping(x, name: str) -> None:
    _require_type(x, Mapping, name)


def _require_has_keys(section: Mapping, keys: Sequence[str], name: str) -> None:
    missing = [k for k in keys if k not in section]
    _require(not missing, f"{name} missing keys: {missing}")


def _require_int(x, name: str) -> None:
    _require_type(x, int, name)


def _require_positive_int(x, name: str) -> None:
    _require_int(x, name)
    _require(x > 0, f"{name} must be > 0")


def _require_floatlike(x, name: str) -> None:
    _require_type(x, (int, float), name)


def _require_nonnegative_floatlike(x, name: str) -> None:
    _require_floatlike(x, name)
    _require(float(x) >= 0.0, f"{name} must be >= 0")


def _require_positive_floatlike(x, name: str) -> None:
    _require_floatlike(x, name)
    _require(float(x) > 0.0, f"{name} must be > 0")


def _require_prob_band(low, high, name_low: str, name_high: str) -> None:
    _require_floatlike(low, name_low)
    _require_floatlike(high, name_high)
    lo = float(low)
    hi = float(high)
    _require(0.0 <= lo <= 1.0, f"{name_low} must be in [0, 1]")
    _require(0.0 <= hi <= 1.0, f"{name_high} must be in [0, 1]")
    _require(lo <= hi, f"{name_low} must be <= {name_high}")


def _require_tf_tensor(x, name: str) -> None:
    if not isinstance(x, (tf.Tensor, tf.Variable)):
        raise TypeError(f"{name} must be a tf.Tensor or tf.Variable, got {type(x)}")


def _require_float64_tensor(x, name: str) -> None:
    _require_tf_tensor(x, name)
    _require(x.dtype == tf.float64, f"{name} must have dtype tf.float64, got {x.dtype}")


def _require_static_shape(x: tf.Tensor, expected: tuple[int, ...], name: str) -> None:
    shape = x.shape
    _require(
        shape.rank == len(expected),
        f"{name} must have rank {len(expected)}, got {shape.rank}",
    )
    dims = shape.as_list()
    _require(
        all(d is not None for d in dims),
        f"{name} must have a fully known static shape, got {shape}",
    )
    _require(
        tuple(dims) == expected, f"{name} must have shape {expected}, got {tuple(dims)}"
    )


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(x, f"{name} must be finite")


def _require_nonnegative(x: tf.Tensor, name: str) -> None:
    tf.debugging.assert_greater_equal(
        x, tf.zeros([], tf.float64), f"{name} must be >= 0"
    )


def _require_binary_01(x: tf.Tensor, name: str) -> None:
    ok = tf.reduce_all(tf.logical_or(tf.equal(x, 0.0), tf.equal(x, 1.0)))
    _require(bool(ok.numpy()), f"{name} must be binary in {{0,1}}")


def _require_open_unit_interval(x: tf.Tensor, name: str) -> None:
    gt0 = tf.reduce_all(x > 0.0)
    lt1 = tf.reduce_all(x < 1.0)
    _require(
        bool(gt0.numpy()) and bool(lt1.numpy()),
        f"{name} must satisfy 0 < {name} < 1 elementwise",
    )


def validate_data_inputs(
    delta_cl: tf.Tensor, qjt: tf.Tensor, q0t: tf.Tensor
) -> tuple[int, int]:
    """Validate external data tensors and return (T, J)."""
    _require_float64_tensor(delta_cl, "delta_cl")
    _require_float64_tensor(qjt, "qjt")
    _require_float64_tensor(q0t, "q0t")

    _require(
        delta_cl.shape.rank == 2,
        f"delta_cl must have rank 2, got {delta_cl.shape.rank}",
    )
    _require(qjt.shape.rank == 2, f"qjt must have rank 2, got {qjt.shape.rank}")
    _require(q0t.shape.rank == 1, f"q0t must have rank 1, got {q0t.shape.rank}")

    dc = delta_cl.shape.as_list()
    qj = qjt.shape.as_list()
    q0 = q0t.shape.as_list()
    _require(
        all(d is not None for d in dc),
        f"delta_cl must have static shape (T,J), got {delta_cl.shape}",
    )
    _require(
        all(d is not None for d in qj),
        f"qjt must have static shape (T,J), got {qjt.shape}",
    )
    _require(
        all(d is not None for d in q0),
        f"q0t must have static shape (T,), got {q0t.shape}",
    )

    T, J = int(dc[0]), int(dc[1])
    _require(
        qj[0] == T and qj[1] == J,
        f"qjt shape must match delta_cl shape {(T, J)}, got {tuple(qj)}",
    )
    _require(q0[0] == T, f"q0t length must match T={T}, got {q0[0]}")

    _require_all_finite(delta_cl, "delta_cl")
    _require_all_finite(qjt, "qjt")
    _require_all_finite(q0t, "q0t")

    _require_nonnegative(qjt, "qjt")
    _require_nonnegative(q0t, "q0t")

    return T, J


def validate_init_config(config: Mapping[str, object], T: int, J: int) -> None:
    """Validate estimator construction config (priors + initial state)."""
    _require_mapping(config, "init_config")
    _require_positive_int(T, "T")
    _require_positive_int(J, "J")

    _require_has_keys(config, ["seed", "posterior", "init_state"], "init_config")

    seed = config["seed"]
    posterior = config["posterior"]
    init_state = config["init_state"]

    _require_int(seed, "seed")
    _require_mapping(posterior, "posterior")
    _require_mapping(init_state, "init_state")

    posterior_keys = [
        "alpha_mean",
        "alpha_var",
        "E_bar_mean",
        "E_bar_var",
        "T0_sq",
        "T1_sq",
        "a_phi",
        "b_phi",
    ]
    _require_has_keys(posterior, posterior_keys, "posterior")

    _require_floatlike(posterior["alpha_mean"], "posterior.alpha_mean")
    _require_positive_floatlike(posterior["alpha_var"], "posterior.alpha_var")
    _require_floatlike(posterior["E_bar_mean"], "posterior.E_bar_mean")
    _require_positive_floatlike(posterior["E_bar_var"], "posterior.E_bar_var")
    _require_positive_floatlike(posterior["T0_sq"], "posterior.T0_sq")
    _require_positive_floatlike(posterior["T1_sq"], "posterior.T1_sq")
    _require_positive_floatlike(posterior["a_phi"], "posterior.a_phi")
    _require_positive_floatlike(posterior["b_phi"], "posterior.b_phi")
    _require(
        float(posterior["T1_sq"]) > float(posterior["T0_sq"]),
        "posterior.T1_sq must be > posterior.T0_sq",
    )

    init_keys = ["alpha", "E_bar", "njt", "gamma", "phi"]
    _require_has_keys(init_state, init_keys, "init_state")

    alpha0 = init_state["alpha"]
    E_bar0 = init_state["E_bar"]
    njt0 = init_state["njt"]
    gamma0 = init_state["gamma"]
    phi0 = init_state["phi"]

    _require_float64_tensor(alpha0, "init_state.alpha")
    _require_float64_tensor(E_bar0, "init_state.E_bar")
    _require_float64_tensor(njt0, "init_state.njt")
    _require_float64_tensor(gamma0, "init_state.gamma")
    _require_float64_tensor(phi0, "init_state.phi")

    _require_static_shape(alpha0, (), "init_state.alpha")
    _require_static_shape(E_bar0, (T,), "init_state.E_bar")
    _require_static_shape(njt0, (T, J), "init_state.njt")
    _require_static_shape(gamma0, (T, J), "init_state.gamma")
    _require_static_shape(phi0, (T,), "init_state.phi")

    _require_all_finite(alpha0, "init_state.alpha")
    _require_all_finite(E_bar0, "init_state.E_bar")
    _require_all_finite(njt0, "init_state.njt")
    _require_all_finite(gamma0, "init_state.gamma")
    _require_all_finite(phi0, "init_state.phi")

    _require_binary_01(gamma0, "init_state.gamma")
    _require_open_unit_interval(phi0, "init_state.phi")


def validate_fit_config(config: Mapping[str, object]) -> None:
    """Validate fit() configuration (run length + tuning settings)."""
    _require_mapping(config, "fit_config")

    required = [
        "n_iter",
        "pilot_length",
        "ridge",
        "target_low",
        "target_high",
        "max_rounds",
        "factor_rw",
        "factor_tmh",
        "k_alpha0",
        "k_E_bar0",
        "k_njt0",
        "tune_seed",
    ]
    _require_has_keys(config, required, "fit_config")

    n_iter = config["n_iter"]
    pilot_length = config["pilot_length"]

    _require_positive_int(n_iter, "n_iter")
    _require_positive_int(pilot_length, "pilot_length")
    _require(pilot_length <= n_iter, "pilot_length must be <= n_iter")

    _require_nonnegative_floatlike(config["ridge"], "ridge")
    _require_prob_band(
        config["target_low"], config["target_high"], "target_low", "target_high"
    )
    _require_positive_int(config["max_rounds"], "max_rounds")

    _require_positive_floatlike(config["factor_rw"], "factor_rw")
    _require_positive_floatlike(config["factor_tmh"], "factor_tmh")
    _require(float(config["factor_rw"]) > 1.0, "factor_rw must be > 1")
    _require(float(config["factor_tmh"]) > 1.0, "factor_tmh must be > 1")

    _require_positive_floatlike(config["k_alpha0"], "k_alpha0")
    _require_positive_floatlike(config["k_E_bar0"], "k_E_bar0")
    _require_positive_floatlike(config["k_njt0"], "k_njt0")

    _require_int(config["tune_seed"], "tune_seed")


def fit_validate_input(*args, **kwargs):
    """Deprecated. Use validate_fit_config(fit_config) instead."""
    raise RuntimeError(
        "fit_validate_input is deprecated. Use validate_fit_config(fit_config)."
    )
