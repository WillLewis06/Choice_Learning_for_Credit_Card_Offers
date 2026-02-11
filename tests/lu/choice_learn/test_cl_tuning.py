"""
Unit tests for lu.choice_learn.cl_tuning.

Constraints for this test module
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared assertion helpers are imported from `lu_conftest` (a normal Python module).
- Patching uses unittest.mock.patch (not the pytest monkeypatch fixture).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import lu.choice_learn.cl_tuning as cl_tuning
from lu.choice_learn.cl_shrinkage import ChoiceLearnShrinkageEstimator
from lu_conftest import assert_all_finite_tf

DTYPE = tf.float64


# -----------------------------------------------------------------------------
# Helpers (test-only)
# -----------------------------------------------------------------------------
def _tiny_cl_data_np(seed: int = 0) -> dict:
    """
    Tiny (T=2, J=3) choice-learn panel for tuning tests.

    Keys
    - delta_cl: (T,J) np.float64
    - qjt: (T,J) np.float64
    - q0t: (T,) np.float64
    """
    rng = np.random.default_rng(seed)
    T, J = 2, 3

    delta_cl = rng.normal(size=(T, J)).astype(np.float64)
    qjt = rng.integers(low=1, high=15, size=(T, J)).astype(np.float64)
    q0t = rng.integers(low=5, high=25, size=(T,)).astype(np.float64)

    return {"T": T, "J": J, "delta_cl": delta_cl, "qjt": qjt, "q0t": q0t}


def _build_shrink(tiny_data: dict, seed: int = 123) -> ChoiceLearnShrinkageEstimator:
    return ChoiceLearnShrinkageEstimator(
        delta_cl=tiny_data["delta_cl"],
        qjt=tiny_data["qjt"],
        q0t=tiny_data["q0t"],
        seed=seed,
    )


def _set_tuning_params(
    shrink: ChoiceLearnShrinkageEstimator,
    pilot_length: int = 1,
    ridge: float = 1e-6,
    target_low: float = 0.3,
    target_high: float = 0.5,
    max_rounds: int = 1,
    factor_rw: float = 1.1,
    factor_tmh: float = 1.5,
) -> None:
    shrink.pilot_length = pilot_length
    shrink.ridge = ridge
    shrink.target_low = target_low
    shrink.target_high = target_high
    shrink.max_rounds = max_rounds
    shrink.factor_rw = factor_rw
    shrink.factor_tmh = factor_tmh


def _assert_scalar_positive(x: tf.Tensor) -> None:
    x_t = tf.convert_to_tensor(x, dtype=DTYPE)
    assert x_t.shape == ()
    assert_all_finite_tf(x_t)
    assert float(x_t.numpy()) > 0.0


def _snapshot_state(shrink: ChoiceLearnShrinkageEstimator) -> dict:
    return {
        "alpha": float(shrink.alpha.read_value().numpy()),
        "E_bar": shrink.E_bar.read_value().numpy().copy(),
        "njt": shrink.njt.read_value().numpy().copy(),
        "gamma": shrink.gamma.read_value().numpy().copy(),
        "phi": shrink.phi.read_value().numpy().copy(),
    }


def _assert_state_unchanged(before: dict, after: dict) -> None:
    assert before["alpha"] == after["alpha"]
    assert np.array_equal(before["E_bar"], after["E_bar"])
    assert np.array_equal(before["njt"], after["njt"])
    assert np.array_equal(before["gamma"], after["gamma"])
    assert np.array_equal(before["phi"], after["phi"])


@contextmanager
def _patched_updates(accept_all: bool):
    """
    Patch cl_tuning update_* functions so that:
    - proposals do not change the pilot state (identity updates)
    - acceptance is deterministic (all accept or all reject)
    """
    accept_bool = tf.constant(bool(accept_all), dtype=tf.bool)

    def stub_update_alpha(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        k_alpha: tf.Tensor,
    ):
        return alpha, accept_bool

    def stub_update_E_bar(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
        k_E_bar: tf.Tensor,
    ):
        accepted_vec = tf.fill(tf.shape(E_bar), accept_bool)
        return E_bar, accepted_vec

    def stub_update_njt(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
        k_njt: tf.Tensor,
        ridge: tf.Tensor,
    ):
        if accept_all:
            acc_sum = tf.cast(tf.shape(njt)[0], tf.float64)
        else:
            acc_sum = tf.constant(0.0, tf.float64)
        return njt, acc_sum

    with patch.object(cl_tuning, "update_alpha", stub_update_alpha):
        with patch.object(cl_tuning, "update_E_bar", stub_update_E_bar):
            with patch.object(cl_tuning, "update_njt", stub_update_njt):
                yield


# -----------------------------------------------------------------------------
# tune_k unit tests
# -----------------------------------------------------------------------------
def test_tune_k_validate_input_rejects_invalid_args():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(0.0, tf.float64)

    bad_calls = [
        dict(
            pilot_length=0,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name="x",
        ),
        dict(
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=0,
            factor=1.1,
            name="x",
        ),
        dict(
            pilot_length=1,
            target_low=0.6,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name="x",
        ),
        dict(
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.0,
            name="x",
        ),
        dict(pilot_length=1, target_low=0.3, target_high=0.5, max_rounds=1, factor=1.1, name=123),  # type: ignore[arg-type]
    ]

    for kw in bad_calls:
        with pytest.raises(Exception):
            cl_tuning.tune_k(theta0=theta0, step_fn=step_fn, k0=k0, **kw)


def test_tune_k_shrinks_k_when_acceptance_below_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(0.0, tf.float64)  # always reject

    k_out = cl_tuning.tune_k(
        theta0=theta0,
        step_fn=step_fn,
        k0=k0,
        pilot_length=2,
        target_low=0.3,
        target_high=0.5,
        max_rounds=3,
        factor=1.1,
        name="reject",
    )

    _assert_scalar_positive(k_out)
    assert float(k_out.numpy()) < float(k0.numpy())


def test_tune_k_grows_k_when_acceptance_above_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(1.0, tf.float64)  # always accept

    k_out = cl_tuning.tune_k(
        theta0=theta0,
        step_fn=step_fn,
        k0=k0,
        pilot_length=2,
        target_low=0.3,
        target_high=0.5,
        max_rounds=3,
        factor=1.1,
        name="accept",
    )

    _assert_scalar_positive(k_out)
    assert float(k_out.numpy()) > float(k0.numpy())


def test_tune_k_keeps_k_unchanged_when_acceptance_in_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(0.4, tf.float64)  # in-band

    k_out = cl_tuning.tune_k(
        theta0=theta0,
        step_fn=step_fn,
        k0=k0,
        pilot_length=5,
        target_low=0.3,
        target_high=0.5,
        max_rounds=10,
        factor=1.1,
        name="inband",
    )

    assert float(k_out.numpy()) == float(k0.numpy())


def test_tune_k_preserves_theta_shape_scalar_and_vector():
    k0 = tf.constant(1.0, tf.float64)

    for shape_case in ["scalar", "vector"]:
        theta0 = (
            tf.constant(0.0, tf.float64)
            if shape_case == "scalar"
            else tf.constant([0.0, 1.0, -1.0], tf.float64)
        )

        def step_fn(theta, k):
            return theta, tf.constant(0.4, tf.float64)  # in-band

        k_out = cl_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=3,
            target_low=0.3,
            target_high=0.5,
            max_rounds=5,
            factor=1.1,
            name=f"shape_{shape_case}",
        )
        _assert_scalar_positive(k_out)


# -----------------------------------------------------------------------------
# tune_shrinkage wiring tests
# -----------------------------------------------------------------------------
def test_tune_shrinkage_validate_input_rejects_missing_or_wrong_types():
    data = _tiny_cl_data_np()
    shrink = _build_shrink(data)
    _set_tuning_params(shrink)

    # Missing attribute
    delattr(shrink, "qjt")
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)

    # Wrong type for state variable (must be tf.Variable)
    shrink = _build_shrink(data)
    _set_tuning_params(shrink)
    shrink.alpha = shrink.alpha.read_value()  # type: ignore[assignment]
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)

    # Invalid factor_rw
    shrink = _build_shrink(data)
    _set_tuning_params(shrink, factor_rw=1.0)
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)


def test_tune_shrinkage_returns_three_positive_finite_scalars():
    data = _tiny_cl_data_np()
    shrink = _build_shrink(data)
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    with _patched_updates(accept_all=True):
        k_alpha, k_E_bar, k_njt = cl_tuning.tune_shrinkage(shrink)

    for k in [k_alpha, k_E_bar, k_njt]:
        _assert_scalar_positive(k)


def test_tune_shrinkage_does_not_mutate_sampler_state():
    data = _tiny_cl_data_np()
    shrink = _build_shrink(data)
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    with _patched_updates(accept_all=False):
        before = _snapshot_state(shrink)
        _ = cl_tuning.tune_shrinkage(shrink)
        after = _snapshot_state(shrink)

    _assert_state_unchanged(before, after)


def test_tune_shrinkage_uses_correct_factor_for_rw_vs_tmh():
    data = _tiny_cl_data_np()
    shrink = _build_shrink(data)
    _set_tuning_params(
        shrink,
        pilot_length=1,
        max_rounds=1,
        target_low=0.3,
        target_high=0.5,
        factor_rw=1.1,
        factor_tmh=1.5,
    )

    calls: list[tuple[str, float]] = []

    def stub_tune_k(
        theta0: tf.Tensor,
        step_fn,
        k0: tf.Tensor,
        pilot_length: int,
        target_low: float,
        target_high: float,
        max_rounds: int,
        factor: float,
        name: str,
    ) -> tf.Tensor:
        calls.append((str(name), float(factor)))
        return tf.convert_to_tensor(k0, dtype=tf.float64)

    with patch.object(cl_tuning, "tune_k", stub_tune_k):
        _ = cl_tuning.tune_shrinkage(shrink)

    names = [n for (n, _) in calls]
    assert (
        len(calls) >= 3
    ), f"Expected at least 3 tune_k calls, got {len(calls)} with names={names}"

    atol = 1e-6
    for name, factor in calls:
        if name in ["alpha", "E_bar"]:
            exp = float(shrink.factor_rw)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_rw={shrink.factor_rw} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )

        if name in ["njt"]:
            exp = float(shrink.factor_tmh)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_tmh={shrink.factor_tmh} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )
