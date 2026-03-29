"""Unit tests for lu.choice_learn.cl_tuning.

This file targets the refactored tuning API:
- _tune_block(...)
- tune_shrinkage(...)

The old estimator-style tune view, tune_k(...), and update_* patching workflow
are no longer part of the current design and are not tested here.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import tensorflow as tf

import lu.choice_learn.cl_tuning as cl_tuning
from lu.choice_learn.cl_posterior import (
    ChoiceLearnPosteriorConfig,
    ChoiceLearnPosteriorTF,
)
from lu.choice_learn.cl_shrinkage import (
    ChoiceLearnShrinkageConfig,
    ChoiceLearnShrinkageState,
)

DTYPE = tf.float64
SEED_DTYPE = tf.int32
ATOL = 1e-12


def _tf(x) -> tf.Tensor:
    """Create a tf.float64 constant."""
    return tf.constant(x, dtype=DTYPE)


def _seed(a: int, b: int) -> tf.Tensor:
    """Create a stateless seed tensor."""
    return tf.constant([a, b], dtype=SEED_DTYPE)


def _assert_scalar_positive(x: tf.Tensor) -> None:
    """Assert that x is a finite positive scalar tensor."""
    x_t = tf.convert_to_tensor(x, dtype=DTYPE)
    if x_t.shape != ():
        raise AssertionError(f"Expected scalar tensor, got shape {x_t.shape}.")
    x_val = float(x_t.numpy())
    if not np.isfinite(x_val):
        raise AssertionError("Expected finite scalar.")
    if not (x_val > 0.0):
        raise AssertionError(f"Expected positive scalar, got {x_val}.")


def _make_posterior_config() -> ChoiceLearnPosteriorConfig:
    """Build a small posterior config for tuning tests."""
    return ChoiceLearnPosteriorConfig(
        alpha_mean=1.0,
        alpha_var=1.0,
        E_bar_mean=0.0,
        E_bar_var=1.0,
        T0_sq=1e-2,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
    )


def _make_posterior() -> ChoiceLearnPosteriorTF:
    """Create the refactored posterior object."""
    return ChoiceLearnPosteriorTF(config=_make_posterior_config())


def _make_problem() -> dict:
    """Canonical tiny panel used in tuning tests."""
    T, J = 2, 3

    delta_cl = _tf([[0.2, -0.1, 0.05], [0.0, 0.3, -0.2]])
    qjt = _tf([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]])
    q0t = _tf([20.0, 15.0])

    initial_state = ChoiceLearnShrinkageState(
        alpha=_tf(1.2),
        E_bar=_tf([0.05, -0.10]),
        njt=_tf([[0.00, 0.20, -0.10], [0.05, -0.02, 0.00]]),
        gamma=_tf([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
        "initial_state": initial_state,
    }


def _make_shrinkage_config(
    k_alpha: float = 0.10,
    k_E_bar: float = 0.10,
    k_njt: float = 0.10,
) -> ChoiceLearnShrinkageConfig:
    """Build a small sampler config for tuning tests."""
    return ChoiceLearnShrinkageConfig(
        num_results=5,
        num_burnin_steps=3,
        chunk_size=4,
        k_alpha=float(k_alpha),
        k_E_bar=float(k_E_bar),
        k_njt=float(k_njt),
        pilot_length=2,
        target_low=0.30,
        target_high=0.50,
        max_rounds=3,
        factor=1.25,
    )


def _snapshot_state(state: ChoiceLearnShrinkageState) -> dict[str, object]:
    """Take a NumPy snapshot of the immutable chain state."""
    return {
        "alpha": float(state.alpha.numpy()),
        "E_bar": state.E_bar.numpy().copy(),
        "njt": state.njt.numpy().copy(),
        "gamma": state.gamma.numpy().copy(),
    }


def _assert_state_unchanged(
    before: dict[str, object],
    after: ChoiceLearnShrinkageState,
) -> None:
    """Assert that a chain state matches a saved snapshot."""
    assert before["alpha"] == float(after.alpha.numpy())
    assert np.array_equal(before["E_bar"], after.E_bar.numpy())
    assert np.array_equal(before["njt"], after.njt.numpy())
    assert np.array_equal(before["gamma"], after.gamma.numpy())


# -----------------------------------------------------------------------------
# _tune_block unit tests
# -----------------------------------------------------------------------------
def test_tune_block_shrinks_k_when_acceptance_below_band():
    theta0 = _tf(0.0)
    k0 = _tf(1.0)

    def step_fn(
        theta: tf.Tensor, k: tf.Tensor, seed: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del theta, k, seed
        return _tf(0.0), _tf(0.0)

    with patch("builtins.print"):
        k_out, theta_out = cl_tuning._tune_block(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=2,
            target_low=0.3,
            target_high=0.5,
            max_rounds=3,
            factor=1.1,
            name="shrink_case",
            seed=_seed(1, 2),
        )

    _assert_scalar_positive(k_out)
    assert float(k_out.numpy()) < float(k0.numpy())
    assert theta_out.shape == theta0.shape


def test_tune_block_grows_k_when_acceptance_above_band():
    theta0 = _tf(0.0)
    k0 = _tf(1.0)

    def step_fn(
        theta: tf.Tensor, k: tf.Tensor, seed: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del theta, k, seed
        return _tf(0.0), _tf(1.0)

    with patch("builtins.print"):
        k_out, theta_out = cl_tuning._tune_block(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=2,
            target_low=0.3,
            target_high=0.5,
            max_rounds=3,
            factor=1.1,
            name="grow_case",
            seed=_seed(3, 4),
        )

    _assert_scalar_positive(k_out)
    assert float(k_out.numpy()) > float(k0.numpy())
    assert theta_out.shape == theta0.shape


def test_tune_block_keeps_k_when_acceptance_in_band():
    theta0 = _tf(0.0)
    k0 = _tf(1.0)

    def step_fn(
        theta: tf.Tensor, k: tf.Tensor, seed: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del k, seed
        return theta, _tf(0.4)

    with patch("builtins.print"):
        k_out, theta_out = cl_tuning._tune_block(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=5,
            target_low=0.3,
            target_high=0.5,
            max_rounds=10,
            factor=1.1,
            name="in_band_case",
            seed=_seed(5, 6),
        )

    assert float(k_out.numpy()) == float(k0.numpy())
    assert theta_out.shape == theta0.shape


def test_tune_block_preserves_theta_shape_for_scalar_vector_and_matrix_cases():
    k0 = _tf(1.0)

    for theta0 in [
        _tf(0.0),
        _tf([0.0, 1.0, -1.0]),
        _tf([[0.0, 1.0], [-1.0, 2.0]]),
    ]:

        def step_fn(
            theta: tf.Tensor, k: tf.Tensor, seed: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            del k, seed
            return theta, _tf(0.4)

        with patch("builtins.print"):
            k_out, theta_out = cl_tuning._tune_block(
                theta0=theta0,
                step_fn=step_fn,
                k0=k0,
                pilot_length=3,
                target_low=0.3,
                target_high=0.5,
                max_rounds=5,
                factor=1.1,
                name="shape_case",
                seed=_seed(7, 8),
            )

        _assert_scalar_positive(k_out)
        assert tuple(theta_out.shape) == tuple(theta0.shape)


def test_tune_block_carries_forward_terminal_theta_between_rounds():
    theta0 = _tf(0.0)
    k0 = _tf(1.0)

    def step_fn(
        theta: tf.Tensor, k: tf.Tensor, seed: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del k, seed
        return theta + _tf(1.0), _tf(0.0)

    with patch("builtins.print"):
        k_out, theta_out = cl_tuning._tune_block(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=3,
            target_low=0.3,
            target_high=0.5,
            max_rounds=2,
            factor=2.0,
            name="carry_forward_case",
            seed=_seed(9, 10),
        )

    _assert_scalar_positive(k_out)
    assert float(theta_out.numpy()) == 6.0


# -----------------------------------------------------------------------------
# tune_shrinkage orchestration tests
# -----------------------------------------------------------------------------
def test_tune_shrinkage_returns_updated_config_and_preserves_non_tuned_fields():
    posterior = _make_posterior()
    problem = _make_problem()
    initial_state = problem["initial_state"]
    config = _make_shrinkage_config()

    tuned_values = {
        "alpha": (_tf(0.11), initial_state.alpha),
        "E_bar": (_tf(0.22), initial_state.E_bar),
        "njt": (_tf(0.33), initial_state.njt),
    }

    def stub_tune_block(
        theta0: tf.Tensor,
        step_fn,
        k0: tf.Tensor,
        pilot_length: int,
        target_low: float,
        target_high: float,
        max_rounds: int,
        factor: float,
        name: str,
        seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del (
            theta0,
            step_fn,
            k0,
            pilot_length,
            target_low,
            target_high,
            max_rounds,
            factor,
            seed,
        )
        return tuned_values[name]

    with patch.object(cl_tuning, "_tune_block", side_effect=stub_tune_block):
        tuned_config = cl_tuning.tune_shrinkage(
            posterior=posterior,
            qjt=problem["qjt"],
            q0t=problem["q0t"],
            delta_cl=problem["delta_cl"],
            initial_state=initial_state,
            shrinkage_config=config,
            seed=_seed(11, 12),
        )

    assert tuned_config.k_alpha == 0.11
    assert tuned_config.k_E_bar == 0.22
    assert tuned_config.k_njt == 0.33

    assert tuned_config.num_results == config.num_results
    assert tuned_config.num_burnin_steps == config.num_burnin_steps
    assert tuned_config.chunk_size == config.chunk_size
    assert tuned_config.pilot_length == config.pilot_length
    assert tuned_config.target_low == config.target_low
    assert tuned_config.target_high == config.target_high
    assert tuned_config.max_rounds == config.max_rounds
    assert tuned_config.factor == config.factor


def test_tune_shrinkage_does_not_mutate_initial_state_or_input_config():
    posterior = _make_posterior()
    problem = _make_problem()
    initial_state = problem["initial_state"]
    config = _make_shrinkage_config()

    before_state = _snapshot_state(initial_state)
    before_config = config

    def stub_tune_block(
        theta0: tf.Tensor,
        step_fn,
        k0: tf.Tensor,
        pilot_length: int,
        target_low: float,
        target_high: float,
        max_rounds: int,
        factor: float,
        name: str,
        seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del (
            step_fn,
            pilot_length,
            target_low,
            target_high,
            max_rounds,
            factor,
            name,
            seed,
        )
        return k0 + _tf(0.01), theta0

    with patch.object(cl_tuning, "_tune_block", side_effect=stub_tune_block):
        tuned_config = cl_tuning.tune_shrinkage(
            posterior=posterior,
            qjt=problem["qjt"],
            q0t=problem["q0t"],
            delta_cl=problem["delta_cl"],
            initial_state=initial_state,
            shrinkage_config=config,
            seed=_seed(13, 14),
        )

    _assert_state_unchanged(before_state, initial_state)
    assert config == before_config
    assert tuned_config is not config
    assert tuned_config.k_alpha != config.k_alpha
    assert tuned_config.k_E_bar != config.k_E_bar
    assert tuned_config.k_njt != config.k_njt


def test_tune_shrinkage_calls_blocks_in_correct_order_with_same_factor_and_distinct_seeds():
    posterior = _make_posterior()
    problem = _make_problem()
    initial_state = problem["initial_state"]
    config = _make_shrinkage_config()

    calls: list[dict[str, object]] = []

    def stub_tune_block(
        theta0: tf.Tensor,
        step_fn,
        k0: tf.Tensor,
        pilot_length: int,
        target_low: float,
        target_high: float,
        max_rounds: int,
        factor: float,
        name: str,
        seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del step_fn, pilot_length, target_low, target_high, max_rounds
        calls.append(
            {
                "name": name,
                "factor": float(factor),
                "seed": tuple(int(v) for v in seed.numpy().tolist()),
                "k0": float(k0.numpy()),
                "theta_shape": tuple(theta0.shape),
            }
        )
        return k0, theta0

    with patch.object(cl_tuning, "_tune_block", side_effect=stub_tune_block):
        _ = cl_tuning.tune_shrinkage(
            posterior=posterior,
            qjt=problem["qjt"],
            q0t=problem["q0t"],
            delta_cl=problem["delta_cl"],
            initial_state=initial_state,
            shrinkage_config=config,
            seed=_seed(15, 16),
        )

    assert [call["name"] for call in calls] == ["alpha", "E_bar", "njt"]

    for call in calls:
        assert abs(call["factor"] - config.factor) <= ATOL

    seed_tuples = [call["seed"] for call in calls]
    assert len(seed_tuples) == 3
    assert len(set(seed_tuples)) == 3

    assert calls[0]["theta_shape"] == ()
    assert calls[1]["theta_shape"] == (problem["T"],)
    assert calls[2]["theta_shape"] == (problem["T"], problem["J"])
