"""Validate external inputs for the Lu shrinkage estimator.

This module validates observed data, posterior config values, shrinkage config
values, and the external chain seed. It does not validate internal sampler
state and does not perform coercion or defaulting.
"""

from __future__ import annotations

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


def _require_open_unit_float(x, name: str) -> None:
    """Validate that a float lies in the open unit interval."""

    _require_finite_float(x, name)
    _require(x > 0.0, f"{name} must be > 0; got {x}.")
    _require(x < 1.0, f"{name} must be < 1; got {x}.")


def _require_gt_one_float(x, name: str) -> None:
    """Validate that a float is greater than one."""

    _require_finite_float(x, name)
    _require(x > 1.0, f"{name} must be > 1; got {x}.")


def _require_tensor_rank(x: tf.Tensor, rank: int, name: str) -> None:
    """Validate the rank of a tensor input."""

    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got rank {x.shape.rank}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Validate that an input is a tf.float64 tensor."""

    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype == tf.float64, f"{name} must be tf.float64; got {x.dtype}.")


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are finite."""

    ok = bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
    _require(ok, f"{name} must be finite (no NaN/inf).")


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    """Validate that all tensor entries are non-negative."""

    ok = bool(tf.reduce_all(x >= tf.constant(0.0, tf.float64)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_static_matrix_shape(x: tf.Tensor, name: str) -> None:
    """Validate a static non-empty matrix shape."""

    _require_tensor_rank(x, 2, name)
    _require(
        x.shape[0] is not None and x.shape[1] is not None,
        f"{name} must have static shape (T, J); got {x.shape}.",
    )
    _require(x.shape[0] > 0, f"{name} must have T > 0; got shape {x.shape}.")
    _require(x.shape[1] > 0, f"{name} must have J > 0; got shape {x.shape}.")


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

    ok = bool(tf.reduce_all(seed >= tf.constant(0, dtype=seed.dtype)).numpy())
    _require(ok, f"{name} must be non-negative.")


def observed_data_validate_input(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
) -> None:
    """Validate the observed data tensors for the shrinkage chain."""

    # Require all observed tensors to arrive already normalized to tf.float64.
    _require_float64_tensor(pjt, "pjt")
    _require_float64_tensor(wjt, "wjt")
    _require_float64_tensor(qjt, "qjt")
    _require_float64_tensor(q0t, "q0t")

    # Require compatible market-product dimensions across the observed arrays.
    _require_static_matrix_shape(pjt, "pjt")
    _require_static_matrix_shape(wjt, "wjt")
    _require_static_matrix_shape(qjt, "qjt")
    _require_tensor_rank(q0t, 1, "q0t")

    _require(
        q0t.shape[0] is not None,
        f"q0t must have static shape (T,); got {q0t.shape}.",
    )

    T = pjt.shape[0]
    J = pjt.shape[1]

    _require(
        wjt.shape == pjt.shape,
        f"wjt must have shape (T, J)={pjt.shape}; got {wjt.shape}.",
    )
    _require(
        qjt.shape == pjt.shape,
        f"qjt must have shape (T, J)={pjt.shape}; got {qjt.shape}.",
    )
    _require(
        q0t.shape[0] == T,
        f"q0t must have length T={T}; got {q0t.shape[0]}.",
    )

    _require(T > 0, f"pjt must have T > 0; got T={T}.")
    _require(J > 0, f"pjt must have J > 0; got J={J}.")

    # Check numerical validity and count non-negativity.
    _require_all_finite(pjt, "pjt")
    _require_all_finite(wjt, "wjt")
    _require_all_finite(qjt, "qjt")
    _require_all_finite(q0t, "q0t")

    _require_all_nonnegative(qjt, "qjt")
    _require_all_nonnegative(q0t, "q0t")


def posterior_validate_input(posterior_config) -> None:
    """Validate the posterior config required to construct LuPosteriorTF."""

    # Check that all required posterior-config attributes are present.
    required = [
        "n_draws",
        "seed",
        "eps",
        "beta_p_mean",
        "beta_p_var",
        "beta_w_mean",
        "beta_w_var",
        "r_mean",
        "r_var",
        "E_bar_mean",
        "E_bar_var",
        "T0_sq",
        "T1_sq",
        "a_phi",
        "b_phi",
    ]
    missing = [name for name in required if not hasattr(posterior_config, name)]
    _require(not missing, "posterior_config missing fields: " + ", ".join(missing))

    # Validate the simulation and numerical-control parameters.
    _require_positive_int(posterior_config.n_draws, "posterior_config.n_draws")
    _require_nonnegative_int(posterior_config.seed, "posterior_config.seed")

    _require_finite_float(posterior_config.eps, "posterior_config.eps")
    _require(
        0.0 < posterior_config.eps < 0.5,
        f"posterior_config.eps must satisfy 0 < eps < 0.5; got {posterior_config.eps}.",
    )

    # Validate the Gaussian prior location parameters.
    _require_finite_float(posterior_config.beta_p_mean, "posterior_config.beta_p_mean")
    _require_finite_float(posterior_config.beta_w_mean, "posterior_config.beta_w_mean")
    _require_finite_float(posterior_config.r_mean, "posterior_config.r_mean")
    _require_finite_float(posterior_config.E_bar_mean, "posterior_config.E_bar_mean")

    # Validate the Gaussian and spike-slab scale parameters.
    _require_positive_float(posterior_config.beta_p_var, "posterior_config.beta_p_var")
    _require_positive_float(posterior_config.beta_w_var, "posterior_config.beta_w_var")
    _require_positive_float(posterior_config.r_var, "posterior_config.r_var")
    _require_positive_float(posterior_config.E_bar_var, "posterior_config.E_bar_var")

    _require_positive_float(posterior_config.T0_sq, "posterior_config.T0_sq")
    _require_positive_float(posterior_config.T1_sq, "posterior_config.T1_sq")
    _require(
        posterior_config.T1_sq > posterior_config.T0_sq,
        "posterior_config.T1_sq must be > posterior_config.T0_sq; "
        f"got {posterior_config.T1_sq} <= {posterior_config.T0_sq}.",
    )

    # Validate the Beta prior hyperparameters for the inclusion process.
    _require_positive_float(posterior_config.a_phi, "posterior_config.a_phi")
    _require_positive_float(posterior_config.b_phi, "posterior_config.b_phi")


def shrinkage_validate_input(shrinkage_config) -> None:
    """Validate the sampler and tuning config for the shrinkage chain."""

    # Check that all required sampler and tuning attributes are present.
    required = [
        "num_results",
        "num_burnin_steps",
        "chunk_size",
        "k_beta",
        "k_r",
        "k_E_bar",
        "k_njt",
        "pilot_length",
        "target_low",
        "target_high",
        "max_rounds",
        "factor",
    ]
    missing = [name for name in required if not hasattr(shrinkage_config, name)]
    _require(not missing, "shrinkage_config missing fields: " + ", ".join(missing))

    # Validate the chain-length and chunking controls.
    _require_positive_int(shrinkage_config.num_results, "shrinkage_config.num_results")
    _require_nonnegative_int(
        shrinkage_config.num_burnin_steps,
        "shrinkage_config.num_burnin_steps",
    )
    _require_positive_int(shrinkage_config.chunk_size, "shrinkage_config.chunk_size")

    # Validate the proposal scales for the continuous update blocks.
    _require_positive_float(shrinkage_config.k_beta, "shrinkage_config.k_beta")
    _require_positive_float(shrinkage_config.k_r, "shrinkage_config.k_r")
    _require_positive_float(shrinkage_config.k_E_bar, "shrinkage_config.k_E_bar")
    _require_positive_float(shrinkage_config.k_njt, "shrinkage_config.k_njt")

    # Validate the controls for the acceptance-band tuning routine.
    _require_positive_int(
        shrinkage_config.pilot_length,
        "shrinkage_config.pilot_length",
    )
    _require_positive_int(shrinkage_config.max_rounds, "shrinkage_config.max_rounds")
    _require_open_unit_float(shrinkage_config.target_low, "shrinkage_config.target_low")
    _require_open_unit_float(
        shrinkage_config.target_high,
        "shrinkage_config.target_high",
    )
    _require(
        shrinkage_config.target_low < shrinkage_config.target_high,
        "shrinkage_config.target_low must be < shrinkage_config.target_high; "
        f"got {shrinkage_config.target_low} >= {shrinkage_config.target_high}.",
    )
    _require_gt_one_float(shrinkage_config.factor, "shrinkage_config.factor")


def seed_validate_input(seed: tf.Tensor) -> None:
    """Validate the external chain seed tensor."""

    _require_seed_input(seed, "seed")


def run_chain_validate_input(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config,
    shrinkage_config,
    seed: tf.Tensor,
) -> None:
    """Validate the full external input set required by run_chain."""

    # Delegate validation by input category.
    observed_data_validate_input(
        pjt=pjt,
        wjt=wjt,
        qjt=qjt,
        q0t=q0t,
    )
    posterior_validate_input(posterior_config)
    shrinkage_validate_input(shrinkage_config)
    seed_validate_input(seed)
