"""Input validation for the choice-learn shrinkage sampler."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import tensorflow as tf

from lu.choice_learn.cl_posterior import ChoiceLearnPosteriorConfig

if TYPE_CHECKING:
    from lu.choice_learn.cl_shrinkage import ChoiceLearnShrinkageConfig


def _require(condition: bool, message: str) -> None:
    """Raise a ValueError when a validation condition fails."""

    if not condition:
        raise ValueError(message)


def _require_int(x, name: str) -> None:
    """Require a Python int."""

    _require(
        isinstance(x, int) and not isinstance(x, bool),
        f"{name} must be an int; got {type(x).__name__}.",
    )


def _require_positive_int(x, name: str) -> None:
    """Require a strictly positive Python int."""

    _require_int(x, name)
    _require(x > 0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_int(x, name: str) -> None:
    """Require a nonnegative Python int."""

    _require_int(x, name)
    _require(x >= 0, f"{name} must be >= 0; got {x}.")


def _require_float(x, name: str) -> None:
    """Require a real Python scalar usable as a float."""

    _require(
        isinstance(x, (int, float)) and not isinstance(x, bool),
        f"{name} must be a float; got {type(x).__name__}.",
    )


def _require_finite_float(x, name: str) -> None:
    """Require a finite real scalar."""

    _require_float(x, name)
    _require(math.isfinite(float(x)), f"{name} must be finite; got {x}.")


def _require_positive_float(x, name: str) -> None:
    """Require a strictly positive finite real scalar."""

    _require_finite_float(x, name)
    _require(float(x) > 0.0, f"{name} must be > 0; got {x}.")


def _require_open_unit_float(x, name: str) -> None:
    """Require a finite real scalar in the open unit interval."""

    _require_finite_float(x, name)
    _require(0.0 < float(x) < 1.0, f"{name} must satisfy 0 < {name} < 1; got {x}.")


def _require_tensor_input(x, name: str) -> None:
    """Require a TensorFlow tensor input."""

    _require(
        isinstance(x, tf.Tensor),
        f"{name} must be a tf.Tensor; got {type(x).__name__}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    """Require a float64 tensor."""

    _require_tensor_input(x, name)
    _require(
        x.dtype == tf.float64,
        f"{name} must have dtype tf.float64; got {x.dtype}.",
    )


def _require_rank(x: tf.Tensor, rank: int, name: str) -> None:
    """Require a fixed tensor rank."""

    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got rank {x.shape.rank}.",
    )


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    """Require all tensor entries to be finite."""

    ok = bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
    _require(ok, f"{name} must contain only finite values.")


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    """Require all tensor entries to be nonnegative."""

    ok = bool(tf.reduce_all(x >= tf.zeros([], dtype=x.dtype)).numpy())
    _require(ok, f"{name} must contain only nonnegative values.")


def _require_seed_input(seed: tf.Tensor, name: str) -> None:
    """Require a stateless TensorFlow RNG seed of shape ``(2,)``."""

    _require_tensor_input(seed, name)
    _require(
        seed.dtype.is_integer,
        f"{name} must have an integer dtype; got {seed.dtype}.",
    )
    _require_rank(seed, 1, name)

    shape = seed.shape.as_list()
    _require(shape == [2], f"{name} must have shape (2,); got {tuple(shape)}.")

    ok = bool(tf.reduce_all(seed >= tf.zeros([2], dtype=seed.dtype)).numpy())
    _require(ok, f"{name} must contain only nonnegative values.")


def observed_data_validate_input(
    delta_cl: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
) -> None:
    """Validate the observed tensors used by the shrinkage sampler."""

    _require_float64_tensor(delta_cl, "delta_cl")
    _require_float64_tensor(qjt, "qjt")
    _require_float64_tensor(q0t, "q0t")

    _require_rank(delta_cl, 2, "delta_cl")
    _require_rank(qjt, 2, "qjt")
    _require_rank(q0t, 1, "q0t")

    delta_shape = delta_cl.shape.as_list()
    qjt_shape = qjt.shape.as_list()
    q0t_shape = q0t.shape.as_list()

    # Static shapes are required because the sampler logic assumes fixed market
    # and product dimensions throughout the compiled chain.
    _require(
        all(dim is not None for dim in delta_shape),
        f"delta_cl must have static shape (T, J); got {delta_cl.shape}.",
    )
    _require(
        all(dim is not None for dim in qjt_shape),
        f"qjt must have static shape (T, J); got {qjt.shape}.",
    )
    _require(
        all(dim is not None for dim in q0t_shape),
        f"q0t must have static shape (T,); got {q0t.shape}.",
    )

    T = delta_shape[0]
    J = delta_shape[1]

    _require(T > 0, f"delta_cl must have T > 0; got T={T}.")
    _require(J > 0, f"delta_cl must have J > 0; got J={J}.")

    _require(
        qjt_shape == [T, J],
        f"qjt must have shape ({T}, {J}); got {tuple(qjt_shape)}.",
    )
    _require(
        q0t_shape == [T],
        f"q0t must have shape ({T},); got {tuple(q0t_shape)}.",
    )

    _require_all_finite(delta_cl, "delta_cl")
    _require_all_finite(qjt, "qjt")
    _require_all_finite(q0t, "q0t")

    _require_all_nonnegative(qjt, "qjt")
    _require_all_nonnegative(q0t, "q0t")


def posterior_validate_input(
    posterior_config: ChoiceLearnPosteriorConfig,
) -> None:
    """Validate a ``ChoiceLearnPosteriorConfig`` instance."""

    _require_finite_float(posterior_config.alpha_mean, "posterior_config.alpha_mean")
    _require_finite_float(posterior_config.E_bar_mean, "posterior_config.E_bar_mean")

    _require_positive_float(posterior_config.alpha_var, "posterior_config.alpha_var")
    _require_positive_float(posterior_config.E_bar_var, "posterior_config.E_bar_var")

    _require_positive_float(posterior_config.T0_sq, "posterior_config.T0_sq")
    _require_positive_float(posterior_config.T1_sq, "posterior_config.T1_sq")

    # The slab variance must exceed the spike variance.
    _require(
        float(posterior_config.T1_sq) > float(posterior_config.T0_sq),
        "posterior_config.T1_sq must be > posterior_config.T0_sq; "
        f"got {posterior_config.T1_sq} <= {posterior_config.T0_sq}.",
    )

    _require_positive_float(posterior_config.a_phi, "posterior_config.a_phi")
    _require_positive_float(posterior_config.b_phi, "posterior_config.b_phi")


def shrinkage_validate_input(
    shrinkage_config: ChoiceLearnShrinkageConfig,
) -> None:
    """Validate the sampler and tuning configuration for the shrinkage chain."""

    _require_positive_int(shrinkage_config.num_results, "shrinkage_config.num_results")
    _require_nonnegative_int(
        shrinkage_config.num_burnin_steps,
        "shrinkage_config.num_burnin_steps",
    )
    _require_positive_int(shrinkage_config.chunk_size, "shrinkage_config.chunk_size")

    _require_positive_float(shrinkage_config.k_alpha, "shrinkage_config.k_alpha")
    _require_positive_float(shrinkage_config.k_E_bar, "shrinkage_config.k_E_bar")
    _require_positive_float(shrinkage_config.k_njt, "shrinkage_config.k_njt")

    _require_positive_int(
        shrinkage_config.pilot_length,
        "shrinkage_config.pilot_length",
    )
    _require_positive_int(
        shrinkage_config.max_rounds,
        "shrinkage_config.max_rounds",
    )

    # These are acceptance-rate targets for proposal tuning.
    _require_open_unit_float(
        shrinkage_config.target_low,
        "shrinkage_config.target_low",
    )
    _require_open_unit_float(
        shrinkage_config.target_high,
        "shrinkage_config.target_high",
    )
    _require(
        float(shrinkage_config.target_low) < float(shrinkage_config.target_high),
        "shrinkage_config.target_low must be < shrinkage_config.target_high; "
        f"got {shrinkage_config.target_low} >= {shrinkage_config.target_high}.",
    )

    _require_finite_float(shrinkage_config.factor, "shrinkage_config.factor")
    _require(
        float(shrinkage_config.factor) > 1.0,
        f"shrinkage_config.factor must be > 1; got {shrinkage_config.factor}.",
    )


def run_chain_validate_input(
    delta_cl: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config: ChoiceLearnPosteriorConfig,
    shrinkage_config: ChoiceLearnShrinkageConfig,
    seed: tf.Tensor,
) -> None:
    """Validate the full external input set required by ``run_chain``."""

    observed_data_validate_input(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
    )
    posterior_validate_input(posterior_config)
    shrinkage_validate_input(shrinkage_config)

    # The chain uses stateless TensorFlow RNG throughout, so the external seed
    # must match TensorFlow's expected two-integer format.
    _require_seed_input(seed, "seed")
