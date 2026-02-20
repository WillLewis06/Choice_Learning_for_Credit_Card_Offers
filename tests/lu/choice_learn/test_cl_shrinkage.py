"""
Unit tests for the choice-learn shrinkage estimator.

Constraints for this test module
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared assertions come from `lu_conftest` (a normal Python module).
- Tests that call fit() patch:
  - tuning (to avoid slow pilot loops), and
  - diagnostics progress printing (to keep test output clean).

Choice-learn + Lu shrinkage model (systematic utility):
  delta[t, j] = alpha * delta_cl[t, j] + E_bar[t] + njt[t, j]
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import lu.choice_learn.cl_diagnostics as cl_diagnostics_mod
import lu.choice_learn.cl_shrinkage as cl_shrinkage_mod
from lu.choice_learn.cl_diagnostics import ChoiceLearnShrinkageDiagnostics
from lu.choice_learn.cl_posterior import LuPosteriorTF
from lu.choice_learn.cl_shrinkage import ChoiceLearnShrinkageEstimator

from lu_conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
    assert_in_open_unit_interval_tf,
)

DTYPE = tf.float64
ATOL = 1e-10


# -----------------------------------------------------------------------------
# Shared posterior hyperparameters used for tests
# -----------------------------------------------------------------------------
_POSTERIOR_CONFIG = {
    "alpha_mean": 1.0,
    "alpha_var": 1.0,
    "E_bar_mean": 0.0,
    "E_bar_var": 1.0,
    "T0_sq": 0.01,
    "T1_sq": 1.0,
    "a_phi": 2.0,
    "b_phi": 2.0,
}


# -----------------------------------------------------------------------------
# Local constructors / helpers (no pytest fixtures)
# -----------------------------------------------------------------------------
def _tiny_cl_data(seed: int = 0) -> dict:
    """
    Tiny (T=2, J=3) choice-learn problem with internally-consistent counts.

    We generate (qjt, q0t) from the same mapping used by cl_posterior.LuPosteriorTF:
      delta[t] = alpha * delta_cl[t] + E_bar[t] + njt[t]
      (sjt[t], s0[t]) = softmax([0, delta[t]])
      qjt[t] = N * sjt[t],  q0t[t] = N * s0[t]
    """
    rng = np.random.default_rng(seed)
    T, J = 2, 3

    delta_cl_np = rng.normal(size=(T, J)).astype(np.float64)
    delta_cl = tf.constant(delta_cl_np, dtype=DTYPE)

    alpha_true = tf.constant(1.8, DTYPE)
    E_bar_true = tf.constant([0.35, -0.25], DTYPE)

    # Deterministic small njt that varies by product within market.
    dc_c = delta_cl - tf.reduce_mean(delta_cl, axis=1, keepdims=True)
    njt_true = 0.12 * dc_c

    posterior = LuPosteriorTF(_POSTERIOR_CONFIG)

    sjt_list, s0_list = [], []
    for t in range(T):
        delta_t = posterior._mean_utility(
            delta_cl=delta_cl[t],
            alpha=alpha_true,
            E_bar=E_bar_true[t],
            n=njt_true[t],
        )
        log_pj, log_p0 = posterior._log_choice_probs(delta=delta_t)
        sjt_list.append(tf.exp(log_pj))
        s0_list.append(tf.exp(log_p0))

    sjt_true = tf.stack(sjt_list, axis=0)  # (T,J)
    s0_true = tf.stack(s0_list, axis=0)  # (T,)

    N = tf.constant(5000.0, DTYPE)
    qjt = N * sjt_true
    q0t = N * s0_true

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
    }


def _default_init_state(T: int, J: int, posterior_cfg: dict) -> dict:
    """Build a fully-specified init_state mapping satisfying validate_init_config."""
    alpha0 = tf.constant(posterior_cfg["alpha_mean"], dtype=DTYPE)
    E_bar0 = tf.fill([T], tf.constant(posterior_cfg["E_bar_mean"], dtype=DTYPE))
    njt0 = tf.zeros([T, J], dtype=DTYPE)
    gamma0 = tf.zeros([T, J], dtype=DTYPE)

    a = float(posterior_cfg["a_phi"])
    b = float(posterior_cfg["b_phi"])
    phi0_val = tf.constant(a / (a + b), dtype=DTYPE)  # in (0,1) for a,b>0
    phi0 = tf.fill([T], phi0_val)

    return {"alpha": alpha0, "E_bar": E_bar0, "njt": njt0, "gamma": gamma0, "phi": phi0}


def _make_estimator(
    data: dict,
    seed: int = 123,
    posterior_cfg: dict | None = None,
    init_state: dict | None = None,
) -> ChoiceLearnShrinkageEstimator:
    posterior_cfg = _POSTERIOR_CONFIG if posterior_cfg is None else posterior_cfg
    T, J = int(data["T"]), int(data["J"])
    if init_state is None:
        init_state = _default_init_state(T=T, J=J, posterior_cfg=posterior_cfg)

    config = {
        "seed": seed,
        "posterior": posterior_cfg,
        "init_state": init_state,
    }

    return ChoiceLearnShrinkageEstimator(
        delta_cl=data["delta_cl"],
        qjt=data["qjt"],
        q0t=data["q0t"],
        config=config,
    )


def _assert_state_shapes(est: ChoiceLearnShrinkageEstimator, T: int, J: int) -> None:
    assert est.alpha.shape == ()
    assert tuple(est.E_bar.shape) == (T,)
    assert tuple(est.njt.shape) == (T, J)
    assert tuple(est.gamma.shape) == (T, J)
    assert tuple(est.phi.shape) == (T,)


def _assert_state_finite(est: ChoiceLearnShrinkageEstimator) -> None:
    assert_all_finite_tf(est.alpha, est.E_bar, est.njt, est.gamma, est.phi)


def _assert_results_schema(res: dict, T: int, J: int) -> None:
    for k in [
        "alpha_hat",
        "E_hat",
        "E_bar_hat",
        "njt_hat",
        "phi_hat",
        "gamma_hat",
        "n_saved",
    ]:
        assert k in res

    assert np.shape(res["E_bar_hat"]) == (T,)
    assert np.shape(res["phi_hat"]) == (T,)
    assert np.shape(res["njt_hat"]) == (T, J)
    assert np.shape(res["gamma_hat"]) == (T, J)
    assert np.shape(res["E_hat"]) == (T, J)


def _stub_tune_shrinkage(_tune_view):
    """
    Test stub to avoid expensive tuning loops.

    Important:
    - k_njt must be strictly positive because the TMH step constructs a
      covariance / cholesky factor; k_njt=0 can cause an InvalidArgumentError.
    - Use very small positive values to keep proposals near-deterministic.
    """
    k_alpha = tf.constant(1e-8, dtype=DTYPE)
    k_E_bar = tf.constant(1e-8, dtype=DTYPE)
    k_njt = tf.constant(1e-6, dtype=DTYPE)
    return k_alpha, k_E_bar, k_njt


@tf.function(reduce_retracing=True)
def _silent_report_iteration_progress(
    it: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    phi: tf.Tensor,
) -> None:
    # Must be traceable: called inside ChoiceLearnShrinkageDiagnostics.step (tf.function).
    tf.no_op()


@contextmanager
def _patched_tuning_and_progress():
    """
    Patch:
    - cl_shrinkage_mod.tune_shrinkage: avoid pilot tuning loops in tests
    - cl_diagnostics_mod.report_iteration_progress: avoid tf.print output
    """
    with patch.object(cl_shrinkage_mod, "tune_shrinkage", _stub_tune_shrinkage):
        with patch.object(
            cl_diagnostics_mod,
            "report_iteration_progress",
            _silent_report_iteration_progress,
        ):
            yield


# -----------------------------------------------------------------------------
# Input-level validation tests (public entrypoints)
# -----------------------------------------------------------------------------
def test_init_raises_on_shape_or_rank_mismatch():
    data = _tiny_cl_data()
    delta_cl = data["delta_cl"]
    qjt = data["qjt"]
    q0t = data["q0t"]
    T, J = data["T"], data["J"]

    base = dict(delta_cl=delta_cl, qjt=qjt, q0t=q0t)

    # Sanity: base construction should not raise.
    _ = _make_estimator(dict(data), seed=0)

    bad_cases = [
        dict(delta_cl=delta_cl[:, : J - 1]),  # (T, J-1)
        dict(qjt=qjt[:, : J - 1]),  # (T, J-1)
        dict(delta_cl=tf.concat([delta_cl, delta_cl[:1]], axis=0)),  # (T+1,J)
        dict(qjt=qjt[:1, :]),  # (T-1,J)
        dict(q0t=q0t[: T - 1]),  # (T-1,)
        dict(q0t=tf.reshape(q0t, [T, 1])),  # rank mismatch
        dict(delta_cl=delta_cl[0]),  # (J,)
        dict(qjt=qjt[0]),  # (J,)
    ]

    for overrides in bad_cases:
        kwargs = dict(base)
        kwargs.update(overrides)

        config = {
            "seed": 0,
            "posterior": _POSTERIOR_CONFIG,
            "init_state": _default_init_state(
                T=T, J=J, posterior_cfg=_POSTERIOR_CONFIG
            ),
        }
        with pytest.raises(Exception):
            ChoiceLearnShrinkageEstimator(config=config, **kwargs)


def test_fit_raises_on_invalid_arguments():
    """
    fit() should fail fast on invalid arguments via validate_fit_config.

    This test intentionally does not patch tuning/progress: validation happens
    before tuning is called.
    """
    data = _tiny_cl_data()
    est = _make_estimator(data)

    base = dict(
        n_iter=2,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
        k_alpha0=0.2,
        k_E_bar0=0.2,
        k_njt0=0.2,
        tune_seed=0,
    )

    bad_overrides = [
        dict(n_iter=0),
        dict(n_iter=-1),
        dict(pilot_length=0),
        dict(pilot_length=-1),
        dict(pilot_length=3),  # pilot_length > n_iter
        dict(ridge=-1e-6),
        dict(target_low=-0.1),
        dict(target_high=1.1),
        dict(target_low=0.6, target_high=0.5),
        dict(max_rounds=0),
        dict(max_rounds=-1),
        dict(factor_rw=1.0),
        dict(factor_rw=0.9),
        dict(factor_tmh=1.0),
        dict(factor_tmh=0.5),
        dict(k_alpha0=0.0),
        dict(k_E_bar0=0.0),
        dict(k_njt0=0.0),
        dict(tune_seed=1.2),
    ]

    for overrides in bad_overrides:
        cfg = dict(base)
        cfg.update(overrides)
        with pytest.raises(Exception):
            est.fit(cfg)

    for missing_key in list(base.keys()):
        cfg = dict(base)
        cfg.pop(missing_key)
        with pytest.raises(Exception):
            est.fit(cfg)


# -----------------------------------------------------------------------------
# Behavioral tests
# -----------------------------------------------------------------------------
def test_init_state_shapes_and_values():
    data = _tiny_cl_data()
    T, J = data["T"], data["J"]

    init_state = _default_init_state(T=T, J=J, posterior_cfg=_POSTERIOR_CONFIG)
    est = _make_estimator(data, init_state=init_state)

    assert est.T == T
    assert est.J == J
    _assert_state_shapes(est, T, J)

    assert np.allclose(
        est.alpha.numpy(), init_state["alpha"].numpy(), atol=0.0, rtol=0.0
    )
    assert np.allclose(
        est.E_bar.numpy(), init_state["E_bar"].numpy(), atol=0.0, rtol=0.0
    )
    assert np.allclose(est.njt.numpy(), init_state["njt"].numpy(), atol=0.0, rtol=0.0)
    assert np.allclose(
        est.gamma.numpy(), init_state["gamma"].numpy(), atol=0.0, rtol=0.0
    )
    assert np.allclose(est.phi.numpy(), init_state["phi"].numpy(), atol=0.0, rtol=0.0)

    assert_binary_01_tf(est.gamma)
    assert_in_open_unit_interval_tf(est.phi)


def test_mcmc_iteration_step_updates_state_and_increments_saved():
    """
    ChoiceLearnShrinkageEstimator._mcmc_iteration_step updates state only.
    Diagnostics accumulation (and saved count) is handled outside the tf.function.

    This test calls diag.step() explicitly to verify the saved counter increments.
    """
    data = _tiny_cl_data()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    with _patched_tuning_and_progress():
        diag = ChoiceLearnShrinkageDiagnostics(T=T, J=J)

        saved0, *_ = diag.get_sums()
        saved0_i = int(saved0.numpy())

        k = tf.constant(0.2, dtype=DTYPE)
        ridge = tf.constant(1e-6, dtype=DTYPE)

        est._mcmc_iteration_step(
            k_alpha=k,
            k_E_bar=k,
            k_njt=k,
            ridge=ridge,
        )
        diag.step(est, tf.constant(0, dtype=tf.int32))

        _assert_state_shapes(est, T, J)
        _assert_state_finite(est)
        assert_binary_01_tf(est.gamma)
        assert_in_open_unit_interval_tf(est.phi)

        saved1, *_ = diag.get_sums()
        saved1_i = int(saved1.numpy())
        assert saved1_i == saved0_i + 1


def test_run_mcmc_loop_saved_equals_n_iter():
    data = _tiny_cl_data()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    with _patched_tuning_and_progress():
        diag = ChoiceLearnShrinkageDiagnostics(T=T, J=J)
        k = tf.constant(0.2, dtype=DTYPE)
        ridge = tf.constant(1e-6, dtype=DTYPE)

        n_iter = 3
        est._run_mcmc_loop(
            n_iter=n_iter,
            k_alpha=k,
            k_E_bar=k,
            k_njt=k,
            ridge=ridge,
            diag=diag,
        )

        saved, *_ = diag.get_sums()
        assert int(saved.numpy()) == n_iter


def test_fit_runs_with_mocked_tuning():
    data = _tiny_cl_data()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    fit_config = dict(
        n_iter=2,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
        k_alpha0=0.2,
        k_E_bar0=0.2,
        k_njt0=0.2,
        tune_seed=0,
    )

    with _patched_tuning_and_progress():
        est.fit(fit_config)

    assert est._diag is not None
    saved, *_ = est._diag.get_sums()
    assert int(saved.numpy()) == fit_config["n_iter"]

    _assert_state_shapes(est, T, J)
    _assert_state_finite(est)
    assert_binary_01_tf(est.gamma)
    assert_in_open_unit_interval_tf(est.phi)


def test_fit_round_trip_share_misfit_is_stable_under_tiny_step_sizes():
    """
    Round-trip: generate (qjt,q0t) from known parameters using the same mapping
    used in the posterior, then run fit() with patched tuning that returns
    tiny positive step sizes.

    Expectation:
    - fit() should run.
    - share misfit should remain close (not necessarily identical) because the
      proposal scales are extremely small.
    """
    data = _tiny_cl_data(seed=0)
    T, J = data["T"], data["J"]

    posterior = LuPosteriorTF(_POSTERIOR_CONFIG)

    def _choice_probs_batch(alpha, E_bar, njt):
        sjt_list, s0_list = [], []
        for t in range(T):
            delta_t = posterior._mean_utility(
                delta_cl=data["delta_cl"][t],
                alpha=alpha,
                E_bar=E_bar[t],
                n=njt[t],
            )
            log_pj, log_p0 = posterior._log_choice_probs(delta=delta_t)
            sjt_list.append(tf.exp(log_pj))
            s0_list.append(tf.exp(log_p0))
        return tf.stack(sjt_list, axis=0), tf.stack(s0_list, axis=0)

    qjt_np = data["qjt"].numpy()
    q0t_np = data["q0t"].numpy()
    Nt = q0t_np + np.sum(qjt_np, axis=1)
    s_obs = qjt_np / Nt[:, None]
    s0_obs = q0t_np / Nt

    def _misfit(alpha, E_bar, njt) -> float:
        sjt_hat, s0_hat = _choice_probs_batch(alpha, E_bar, njt)
        sj = sjt_hat.numpy()
        s0 = s0_hat.numpy()
        return float(np.sum(np.abs(sj - s_obs)) + np.sum(np.abs(s0 - s0_obs)))

    est = _make_estimator(data, seed=123)

    misfit0 = _misfit(
        est.alpha.read_value(), est.E_bar.read_value(), est.njt.read_value()
    )

    fit_config = dict(
        n_iter=10,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
        k_alpha0=0.2,
        k_E_bar0=0.2,
        k_njt0=0.2,
        tune_seed=0,
    )

    with _patched_tuning_and_progress():
        est.fit(fit_config)

    misfit1 = _misfit(
        est.alpha.read_value(), est.E_bar.read_value(), est.njt.read_value()
    )

    assert np.isfinite(misfit0)
    assert np.isfinite(misfit1)

    # With extremely small proposal scales, the misfit should not move much.
    assert abs(misfit1 - misfit0) < 1e-3


def test_get_results_shapes_and_E_identity():
    data = _tiny_cl_data()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    fit_config = dict(
        n_iter=2,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
        k_alpha0=0.2,
        k_E_bar0=0.2,
        k_njt0=0.2,
        tune_seed=0,
    )

    with _patched_tuning_and_progress():
        est.fit(fit_config)

    res = est.get_results()
    _assert_results_schema(res, T, J)

    E_hat = np.asarray(res["E_hat"])
    E_bar_hat = np.asarray(res["E_bar_hat"])
    njt_hat = np.asarray(res["njt_hat"])

    assert np.all(np.isfinite(E_hat))
    assert np.all(np.isfinite(E_bar_hat))
    assert np.all(np.isfinite(njt_hat))
    assert np.isfinite(res["alpha_hat"])

    assert np.allclose(E_hat, E_bar_hat[:, None] + njt_hat, atol=ATOL, rtol=0.0)


# -----------------------------------------------------------------------------
# Permutation invariance (deterministic): posterior log-density
# -----------------------------------------------------------------------------
def test_logpost_is_equivariant_to_product_permutation():
    """
    If we permute the product index j (columns) in all product-indexed inputs
    (delta_cl, qjt) and permute the product-indexed latent states (njt, gamma)
    the same way, then the per-market log posterior vector should be unchanged.
    """
    data = _tiny_cl_data()
    T, J = data["T"], data["J"]

    delta_cl = tf.identity(data["delta_cl"])  # (T,J)
    qjt = tf.identity(data["qjt"])  # (T,J)
    q0t = tf.identity(data["q0t"])  # (T,)

    alpha = tf.constant(1.2, dtype=DTYPE)
    E_bar = tf.cast(tf.linspace(-0.05, 0.05, T), DTYPE)  # (T,)

    dc_c = delta_cl - tf.reduce_mean(delta_cl, axis=1, keepdims=True)
    njt = 0.03 * dc_c  # (T,J)

    gamma = tf.cast(njt > 0.0, DTYPE)  # (T,J)
    phi = tf.fill([T], tf.constant(0.4, dtype=DTYPE))  # (T,)

    posterior = LuPosteriorTF(_POSTERIOR_CONFIG)

    lp0 = posterior.logpost_vec(
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        alpha=alpha,
        E_bar=E_bar,
        njt=njt,
        gamma=gamma,
        phi=phi,
    ).numpy()

    perm = (
        tf.constant([2, 0, 1], dtype=tf.int32)
        if J == 3
        else tf.range(J, dtype=tf.int32)
    )

    delta_cl_p = tf.gather(delta_cl, perm, axis=1)
    qjt_p = tf.gather(qjt, perm, axis=1)
    njt_p = tf.gather(njt, perm, axis=1)
    gamma_p = tf.gather(gamma, perm, axis=1)

    lp1 = posterior.logpost_vec(
        qjt=qjt_p,
        q0t=q0t,
        delta_cl=delta_cl_p,
        alpha=alpha,
        E_bar=E_bar,
        njt=njt_p,
        gamma=gamma_p,
        phi=phi,
    ).numpy()

    assert lp0.shape == (T,)
    assert lp1.shape == (T,)
    assert np.all(np.isfinite(lp0))
    assert np.all(np.isfinite(lp1))
    assert np.allclose(lp0, lp1, rtol=1e-10, atol=1e-10)
