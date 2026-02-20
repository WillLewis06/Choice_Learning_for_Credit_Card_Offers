# tests/bonus2/test_bonus2_estimator.py
"""
Unit tests for `bonus2.bonus2_estimator.Bonus2Estimator`.

Estimator tests focus on:
- validation behavior for fit inputs,
- correct scalar-fill initialization of theta_init,
- bookkeeping of `saved` and `accept` counters via patched iteration step,
- acceptance-rate denominator equals n_saved (not block size),
- chain-state reset between successive fits.

These tests intentionally avoid exercising the full posterior / likelihood.
"""

from __future__ import annotations

import types
from typing import Callable

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import patch

import bonus2_conftest as bc

from bonus2.bonus2_estimator import Bonus2Estimator, Z_KEYS


def _make_estimator(
    seed: int = 123,
    dims: dict[str, int] | None = None,
    init_theta_overrides: dict[str, float] | None = None,
) -> Bonus2Estimator:
    d = bc.tiny_dims() if dims is None else dict(dims)

    # IMPORTANT: use a network with no empty neighbor lists to satisfy current
    # input validation behavior for neighbors_m.
    panel = bc.panel_np(
        dims=d,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="ring",
        weekend_pattern="0101",
    )

    init_theta = bc.init_theta_scalars(init_theta_overrides)
    sigmas = bc.sigmas_z()
    step_sizes = bc.step_size_z()

    return Bonus2Estimator(
        y_mit=panel["y_mit"],
        delta_mj=panel["delta_mj"],
        is_weekend_t=panel["is_weekend_t"],
        season_sin_kt=panel["season_sin_kt"],
        season_cos_kt=panel["season_cos_kt"],
        neighbors_m=panel["neighbors_m"],
        lookback=int(panel["lookback"]),
        decay=float(panel["decay"]),
        init_theta=init_theta,
        sigmas=sigmas,
        step_size_z=step_sizes,
        seed=int(seed),
    )


def _patch_iteration_step(
    est: Bonus2Estimator,
    step_fn: Callable[[Bonus2Estimator, tf.Tensor], None],
) -> None:
    """
    Replace the compiled `est._mcmc_iteration_step` with a pure-Python bound method.

    The patched method must accept keyword argument `it`, because `fit()` calls:
      self._mcmc_iteration_step(it=it_tf)
    """
    est._mcmc_iteration_step = types.MethodType(step_fn, est)


def _noop_report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
    return None


def test_fit_rejects_nonpositive_n_iter() -> None:
    est = _make_estimator()
    with pytest.raises(ValueError, match=r"n_iter: expected >= 1, got 0"):
        est.fit(n_iter=0)


def test_theta_init_matches_init_theta_scalar_fills() -> None:
    init = {
        "beta_intercept": 0.10,
        "beta_habit": 0.20,
        "beta_peer": 0.30,
        "beta_weekend_weekday": -0.05,
        "beta_weekend_weekend": 0.07,
        "a_m": 0.01,
        "b_m": -0.02,
    }
    est = _make_estimator(init_theta_overrides=init)

    theta0 = est.theta_init
    assert set(theta0.keys()) == {
        "beta_intercept_j",
        "beta_habit_j",
        "beta_peer_j",
        "beta_weekend_jw",
        "a_m",
        "b_m",
    }

    J = est.J
    M = est.M
    K = est.K

    assert theta0["beta_intercept_j"].shape == (J,)
    assert theta0["beta_habit_j"].shape == (J,)
    assert theta0["beta_peer_j"].shape == (J,)
    assert theta0["beta_weekend_jw"].shape == (J, 2)
    assert theta0["a_m"].shape == (M, K)
    assert theta0["b_m"].shape == (M, K)

    np.testing.assert_allclose(
        theta0["beta_intercept_j"], init["beta_intercept"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        theta0["beta_habit_j"], init["beta_habit"], rtol=0, atol=0
    )
    np.testing.assert_allclose(theta0["beta_peer_j"], init["beta_peer"], rtol=0, atol=0)

    np.testing.assert_allclose(
        theta0["beta_weekend_jw"][:, 0], init["beta_weekend_weekday"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        theta0["beta_weekend_jw"][:, 1], init["beta_weekend_weekend"], rtol=0, atol=0
    )

    np.testing.assert_allclose(theta0["a_m"], init["a_m"], rtol=0, atol=0)
    np.testing.assert_allclose(theta0["b_m"], init["b_m"], rtol=0, atol=0)


def test_fit_and_get_results_shapes_and_accept_rates_all_accept() -> None:
    est = _make_estimator(init_theta_overrides={"beta_intercept": 0.0})

    def fake_step(self: Bonus2Estimator, it: tf.Tensor) -> None:
        one_i32 = tf.constant(1, tf.int32)
        inc = tf.constant(0.01, tf.float64)
        for z_key in Z_KEYS:
            self.z[z_key].assign_add(
                tf.ones_like(self.z[z_key], dtype=tf.float64) * inc
            )
            self.accept[z_key].assign_add(one_i32)
        self.saved.assign_add(1)

    _patch_iteration_step(est, fake_step)

    with patch(
        "bonus2.bonus2_estimator.report_iteration_progress",
        new=_noop_report_iteration_progress,
    ):
        est.fit(n_iter=3)

    out = est.get_results()
    assert set(out.keys()) == {"theta_init", "theta_hat", "n_saved", "accept"}

    assert out["n_saved"] == 3
    assert set(out["accept"].keys()) == set(Z_KEYS)
    for k in Z_KEYS:
        assert out["accept"][k] == 1.0

    theta_hat = out["theta_hat"]
    assert set(theta_hat.keys()) == {
        "beta_intercept_j",
        "beta_habit_j",
        "beta_peer_j",
        "beta_weekend_jw",
        "a_m",
        "b_m",
    }

    J = est.J
    M = est.M
    K = est.K
    assert theta_hat["beta_intercept_j"].shape == (J,)
    assert theta_hat["beta_habit_j"].shape == (J,)
    assert theta_hat["beta_peer_j"].shape == (J,)
    assert theta_hat["beta_weekend_jw"].shape == (J, 2)
    assert theta_hat["a_m"].shape == (M, K)
    assert theta_hat["b_m"].shape == (M, K)

    # z moved by 0.03 after 3 sweeps.
    np.testing.assert_allclose(theta_hat["beta_intercept_j"], 0.03, rtol=0, atol=1e-12)


def test_accept_rate_denominator_is_n_saved_not_block_size() -> None:
    est = _make_estimator()

    target_key = "z_beta_intercept_j"

    def fake_step(self: Bonus2Estimator, it: tf.Tensor) -> None:
        it_int = int(it.numpy())
        if it_int % 2 == 0:
            self.accept[target_key].assign_add(tf.constant(1, tf.int32))
        self.saved.assign_add(1)

    _patch_iteration_step(est, fake_step)

    with patch(
        "bonus2.bonus2_estimator.report_iteration_progress",
        new=_noop_report_iteration_progress,
    ):
        est.fit(n_iter=4)

    out = est.get_results()
    assert out["n_saved"] == 4

    # Exactly 2 accepts out of 4 sweeps -> 0.5.
    assert out["accept"][target_key] == 0.5

    for k in Z_KEYS:
        if k != target_key:
            assert out["accept"][k] == 0.0


def test_fit_resets_accept_and_saved() -> None:
    est = _make_estimator()

    def fake_step(self: Bonus2Estimator, it: tf.Tensor) -> None:
        for z_key in Z_KEYS:
            self.accept[z_key].assign_add(tf.constant(1, tf.int32))
        self.saved.assign_add(1)

    _patch_iteration_step(est, fake_step)

    with patch(
        "bonus2.bonus2_estimator.report_iteration_progress",
        new=_noop_report_iteration_progress,
    ):
        est.fit(n_iter=3)

    assert int(est.saved.numpy()) == 3
    for k in Z_KEYS:
        assert int(est.accept[k].numpy()) == 3

    with patch(
        "bonus2.bonus2_estimator.report_iteration_progress",
        new=_noop_report_iteration_progress,
    ):
        est.fit(n_iter=2)

    assert int(est.saved.numpy()) == 2
    for k in Z_KEYS:
        assert int(est.accept[k].numpy()) == 2
