"""Unit tests for lu.choice_learn.cl_updates.

This file targets the refactored update API:
- alpha_one_step
- E_bar_one_step
- njt_one_step
- gamma_one_step

Constraints
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared assertion helpers are imported from `lu_conftest` (a normal Python module).
- Stateless TensorFlow seeds are used to avoid cross-test coupling.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.choice_learn.cl_posterior import (
    ChoiceLearnPosteriorConfig,
    ChoiceLearnPosteriorTF,
)
from lu.choice_learn.cl_updates import (
    E_bar_one_step,
    alpha_one_step,
    gamma_one_step,
    njt_one_step,
)
from lu_conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
)

DTYPE = tf.float64
SEED_DTYPE = tf.int32

K_ALPHA = tf.constant(0.1, dtype=DTYPE)
K_E_BAR = tf.constant(0.1, dtype=DTYPE)
K_NJT = tf.constant(0.1, dtype=DTYPE)


def _seed(a: int, b: int) -> tf.Tensor:
    """Create a stateless seed tensor of shape (2,)."""
    return tf.constant([a, b], dtype=SEED_DTYPE)


def _posterior_config() -> ChoiceLearnPosteriorConfig:
    """Build a small posterior config for one-step update tests."""
    return ChoiceLearnPosteriorConfig(
        alpha_mean=1.0,
        alpha_var=1.0,
        E_bar_mean=0.0,
        E_bar_var=1.0,
        T0_sq=0.01,
        T1_sq=1.0,
        a_phi=2.0,
        b_phi=2.0,
    )


def _posterior() -> ChoiceLearnPosteriorTF:
    """Create the refactored choice-learn posterior object."""
    return ChoiceLearnPosteriorTF(config=_posterior_config())


def _tiny_problem() -> dict:
    """
    Canonical tiny panel used in one-step update tests.

    Shapes
    - delta_cl, qjt: (T, J) with T=2, J=3
    - q0t, E_bar: (T,)
    - alpha: scalar
    - njt, gamma: (T, J)
    """
    T, J = 2, 3

    delta_cl = tf.constant([[0.2, -0.1, 0.05], [0.0, 0.3, -0.2]], dtype=DTYPE)

    qjt = tf.constant([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=DTYPE)
    q0t = tf.constant([20.0, 15.0], dtype=DTYPE)

    alpha = tf.constant(1.2, dtype=DTYPE)
    E_bar = tf.constant([0.1, -0.2], dtype=DTYPE)

    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=DTYPE)
    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=DTYPE)

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
        "alpha": alpha,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
    }


def _assert_binary_accept_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance indicator in {0.0, 1.0}."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(
            f"Expected scalar acceptance indicator, got shape {x_t.shape}."
        )
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")

    x_val = float(x_t.numpy())
    if x_val not in (0.0, 1.0):
        raise AssertionError(
            f"Expected acceptance indicator in {{0.0, 1.0}}, got {x_val}."
        )


def _assert_accept_rate_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance rate in [0.0, 1.0]."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(f"Expected scalar acceptance rate, got shape {x_t.shape}.")
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")

    x_val = float(x_t.numpy())
    if not np.isfinite(x_val):
        raise AssertionError("Acceptance rate is not finite.")
    if not (0.0 <= x_val <= 1.0):
        raise AssertionError(f"Acceptance rate out of bounds: {x_val} not in [0, 1].")


def test_alpha_one_step_returns_scalar_and_binary_accept_indicator():
    posterior = _posterior()
    tiny = _tiny_problem()

    alpha_new, accepted = alpha_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_alpha=K_ALPHA,
        seed=_seed(1, 2),
    )

    assert alpha_new.shape == ()
    assert alpha_new.dtype == DTYPE
    assert_all_finite_tf(alpha_new)
    _assert_binary_accept_scalar(accepted)


def test_alpha_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(11, 12)

    out_1 = alpha_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_alpha=K_ALPHA,
        seed=seed,
    )
    out_2 = alpha_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_alpha=K_ALPHA,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])


def test_E_bar_one_step_returns_vector_and_accept_rate():
    posterior = _posterior()
    tiny = _tiny_problem()
    T = tiny["T"]

    E_bar_new, accept_rate = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=_seed(3, 4),
    )

    assert tuple(E_bar_new.shape) == (T,)
    assert E_bar_new.dtype == DTYPE
    assert_all_finite_tf(E_bar_new)
    _assert_accept_rate_scalar(accept_rate)


def test_E_bar_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(13, 14)

    out_1 = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=seed,
    )
    out_2 = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])


def test_njt_one_step_returns_matrix_and_accept_rate():
    posterior = _posterior()
    tiny = _tiny_problem()
    T, J = tiny["T"], tiny["J"]

    njt_new, accept_rate = njt_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        k_njt=K_NJT,
        seed=_seed(5, 6),
    )

    assert tuple(njt_new.shape) == (T, J)
    assert njt_new.dtype == DTYPE
    assert_all_finite_tf(njt_new)
    _assert_accept_rate_scalar(accept_rate)


def test_njt_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(15, 16)

    out_1 = njt_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        k_njt=K_NJT,
        seed=seed,
    )
    out_2 = njt_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        delta_cl=tiny["delta_cl"],
        alpha=tiny["alpha"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        k_njt=K_NJT,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])


def test_gamma_one_step_returns_binary_matrix():
    posterior = _posterior()
    tiny = _tiny_problem()
    T, J = tiny["T"], tiny["J"]

    gamma_new = gamma_one_step(
        posterior=posterior,
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        seed=_seed(7, 8),
    )

    assert tuple(gamma_new.shape) == (T, J)
    assert gamma_new.dtype == DTYPE
    assert_all_finite_tf(gamma_new)
    assert_binary_01_tf(gamma_new)


def test_gamma_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(17, 18)

    out_1 = gamma_one_step(
        posterior=posterior,
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        seed=seed,
    )
    out_2 = gamma_one_step(
        posterior=posterior,
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        seed=seed,
    )

    tf.debugging.assert_equal(out_1, out_2)
