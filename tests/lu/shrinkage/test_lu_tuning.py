"""
Unit tests for lu.shrinkage.lu_tuning.

Constraints for this test module
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared test utilities are imported from `lu_conftest` (a normal Python module).
- Patching uses unittest.mock.patch (not the pytest monkeypatch fixture).

Notes on updated Lu code
- LuShrinkageEstimator now requires `posterior_config=LuPosteriorConfig(...)`.
- Tuning controls (pilot_length, ridge, targets, factors) are read-only properties
  backed by `shrink._fit_config: LuShrinkageFitConfig`. Tests must set this
  before calling tune_shrinkage().
- tune_k no longer performs explicit input validation; tests should not expect
  invalid-argument rejection from tune_k().
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import lu.shrinkage.lu_tuning as lu_tuning
from lu.shrinkage.lu_posterior import LuPosteriorConfig
from lu.shrinkage.lu_shrinkage import LuShrinkageEstimator, LuShrinkageFitConfig
from lu_conftest import assert_all_finite_tf, tiny_market_data


# -----------------------------------------------------------------------------
# Helpers (test-only)
# -----------------------------------------------------------------------------
def _assert_scalar_positive(x: tf.Tensor) -> None:
    x_t = tf.convert_to_tensor(x, dtype=tf.float64)
    assert x_t.shape == ()
    assert_all_finite_tf(x_t)
    assert float(x_t.numpy()) > 0.0


def _snapshot_state(shrink: LuShrinkageEstimator) -> dict:
    return {
        "beta_p": float(shrink.beta_p.read_value().numpy()),
        "beta_w": float(shrink.beta_w.read_value().numpy()),
        "r": float(shrink.r.read_value().numpy()),
        "E_bar": shrink.E_bar.read_value().numpy().copy(),
        "njt": shrink.njt.read_value().numpy().copy(),
        "gamma": shrink.gamma.read_value().numpy().copy(),
        "phi": shrink.phi.read_value().numpy().copy(),
    }


def _assert_state_unchanged(before: dict, after: dict) -> None:
    assert before["beta_p"] == after["beta_p"]
    assert before["beta_w"] == after["beta_w"]
    assert before["r"] == after["r"]

    assert np.array_equal(before["E_bar"], after["E_bar"])
    assert np.array_equal(before["njt"], after["njt"])
    assert np.array_equal(before["gamma"], after["gamma"])
    assert np.array_equal(before["phi"], after["phi"])


def _make_posterior_config(n_draws: int, seed: int) -> LuPosteriorConfig:
    return LuPosteriorConfig(
        n_draws=int(n_draws),
        seed=int(seed),
        dtype=tf.float64,
        eps=1e-12,
        beta_p_mean=-1.0,
        beta_p_var=1.0,
        beta_w_mean=0.3,
        beta_w_var=1.0,
        r_mean=0.0,
        r_var=1.0,
        E_bar_mean=0.0,
        E_bar_var=1.0,
        T0_sq=1e-2,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
    )


def _build_shrink(
    tiny_data: dict, n_draws: int = 25, seed: int = 123
) -> LuShrinkageEstimator:
    return LuShrinkageEstimator(
        pjt=tiny_data["pjt"],
        wjt=tiny_data["wjt"],
        qjt=tiny_data["qjt"],
        q0t=tiny_data["q0t"],
        posterior_config=_make_posterior_config(n_draws=n_draws, seed=seed),
    )


def _set_fit_config(
    shrink: LuShrinkageEstimator,
    pilot_length: int = 1,
    ridge: float = 1e-6,
    target_low: float = 0.3,
    target_high: float = 0.5,
    max_rounds: int = 1,
    factor_rw: float = 1.1,
    factor_tmh: float = 1.5,
) -> None:
    # Test-only direct assignment: tune_shrinkage reads via properties that
    # dereference shrink._fit_config.
    shrink._fit_config = LuShrinkageFitConfig(
        n_iter=1,
        pilot_length=int(pilot_length),
        ridge=float(ridge),
        target_low=float(target_low),
        target_high=float(target_high),
        max_rounds=int(max_rounds),
        factor_rw=float(factor_rw),
        factor_tmh=float(factor_tmh),
    )


@contextmanager
def _patched_updates(accept_all: bool):
    """
    Patch update_* functions in lu_tuning so that:
    - proposals do not change the pilot state (identity updates)
    - acceptance is deterministic (all accept or all reject)
    """
    accept_bool = tf.constant(bool(accept_all), dtype=tf.bool)

    def stub_update_r(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        k_r: tf.Tensor,
    ):
        return r, accept_bool

    def stub_update_beta(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        k_beta: tf.Tensor,
        ridge: tf.Tensor,
    ):
        return beta_p, beta_w, accept_bool

    def stub_update_E_bar(
        posterior,
        rng: tf.random.Generator,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
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
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
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

    with patch.object(lu_tuning, "update_r", stub_update_r):
        with patch.object(lu_tuning, "update_beta", stub_update_beta):
            with patch.object(lu_tuning, "update_E_bar", stub_update_E_bar):
                with patch.object(lu_tuning, "update_njt", stub_update_njt):
                    yield


# -----------------------------------------------------------------------------
# tune_k unit tests
# -----------------------------------------------------------------------------
def test_tune_k_shrinks_k_when_acceptance_below_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(0.0, tf.float64)  # always reject

    k_out = lu_tuning.tune_k(
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

    k_out = lu_tuning.tune_k(
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

    k_out = lu_tuning.tune_k(
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

        k_out = lu_tuning.tune_k(
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
# tune_shrinkage integration / wiring tests
# -----------------------------------------------------------------------------
def test_tune_shrinkage_validate_input_rejects_missing_or_wrong_types():
    data = tiny_market_data()

    # Missing attribute
    shrink = _build_shrink(data)
    _set_fit_config(shrink)
    delattr(shrink, "qjt")
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)

    # Wrong type for state variable (must be tf.Variable with .read_value())
    shrink = _build_shrink(data)
    _set_fit_config(shrink)
    shrink.beta_p = shrink.beta_p.read_value()  # type: ignore[assignment]
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)


def test_tune_shrinkage_returns_four_positive_finite_scalars():
    data = tiny_market_data()
    shrink = _build_shrink(data)
    _set_fit_config(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    with _patched_updates(accept_all=True):
        k_r, k_E_bar, k_beta, k_njt = lu_tuning.tune_shrinkage(shrink)

    for k in [k_r, k_E_bar, k_beta, k_njt]:
        _assert_scalar_positive(k)


def test_tune_shrinkage_does_not_mutate_sampler_state():
    data = tiny_market_data()
    shrink = _build_shrink(data)
    _set_fit_config(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    with _patched_updates(accept_all=False):
        before = _snapshot_state(shrink)
        _ = lu_tuning.tune_shrinkage(shrink)
        after = _snapshot_state(shrink)

    _assert_state_unchanged(before, after)


def test_tune_shrinkage_uses_correct_factor_for_rw_vs_tmh():
    data = tiny_market_data()
    shrink = _build_shrink(data)
    _set_fit_config(
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

    with patch.object(lu_tuning, "tune_k", stub_tune_k):
        _ = lu_tuning.tune_shrinkage(shrink)

    names = [n for (n, _) in calls]
    assert (
        len(calls) >= 4
    ), f"Expected at least 4 tune_k calls, got {len(calls)} with names={names}"

    atol = 1e-6
    for name, factor in calls:
        if name in ["r", "E_bar"]:
            exp = float(shrink.factor_rw)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_rw={shrink.factor_rw} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )

        if name in ["beta", "njt"]:
            exp = float(shrink.factor_tmh)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_tmh={shrink.factor_tmh} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )
