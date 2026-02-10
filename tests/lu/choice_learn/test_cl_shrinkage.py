# tests/lu/choice_learn/test_cl_shrinkage.py
import numpy as np
import pytest
import tensorflow as tf

import lu.choice_learn.cl_shrinkage as cl_shrinkage_mod
import lu.choice_learn.cl_diagnostics as cl_diagnostics_mod
from lu.choice_learn.cl_diagnostics import ChoiceLearnShrinkageDiagnostics
from lu.choice_learn.cl_posterior import LuPosteriorTF
from lu.choice_learn.cl_shrinkage import ChoiceLearnShrinkageEstimator

# Shared test helpers from tests/conftest.py
from conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
    assert_in_open_unit_interval_tf,
)


# -----------------------------------------------------------------------------
# Local fixtures (self-contained choice-learn tiny problem)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_cl_data():
    """
    Tiny (T=2, J=3) choice-learn problem with internally-consistent counts.

    We generate (qjt, q0t) from the same logit-with-outside mapping used by
    cl_posterior.LuPosteriorTF:
        delta_t = alpha * delta_cl_t + E_bar_t + njt_t
        (sjt_t, s0t) = softmax([0, delta_t])
    """
    rng = np.random.default_rng(0)
    T, J = 2, 3

    delta_cl = rng.normal(size=(T, J)).astype(np.float64)

    # "True" parameters for synthetic counts (not used by the estimator directly)
    alpha_true = tf.constant(1.8, tf.float64)
    E_bar_true = tf.constant([0.35, -0.25], tf.float64)
    njt_true = tf.constant([[0.20, -0.10, 0.00], [-0.08, 0.14, 0.06]], tf.float64)

    posterior = LuPosteriorTF(dtype=tf.float64)

    def _choice_probs_batch(delta_cl_tj, alpha, E_bar, njt):
        delta_cl_tj = tf.convert_to_tensor(delta_cl_tj, tf.float64)
        sjt_list, s0_list = [], []
        for t in range(T):
            delta_t = posterior._mean_utility_jt(
                delta_cl_t=delta_cl_tj[t],
                alpha=alpha,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )
            sjt_t, s0t = posterior._choice_probs_t(delta_t=delta_t)
            sjt_list.append(sjt_t)
            s0_list.append(s0t)
        return tf.stack(sjt_list, axis=0), tf.stack(s0_list, axis=0)

    sjt_true, s0_true = _choice_probs_batch(delta_cl, alpha_true, E_bar_true, njt_true)

    # Convert shares to counts with large N (kept as float64; likelihood uses them as weights)
    N = tf.constant(5000.0, tf.float64)
    qjt = (N * sjt_true).numpy()
    q0t = (N * s0_true).numpy()

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
    }


@pytest.fixture
def estimator(tiny_cl_data):
    return ChoiceLearnShrinkageEstimator(
        delta_cl=tiny_cl_data["delta_cl"],
        qjt=tiny_cl_data["qjt"],
        q0t=tiny_cl_data["q0t"],
        seed=123,
    )


# -----------------------------------------------------------------------------
# Test-only helpers
# -----------------------------------------------------------------------------
def _assert_state_shapes(est: ChoiceLearnShrinkageEstimator, T: int, J: int) -> None:
    assert est.alpha.shape == ()
    assert tuple(est.E_bar.shape) == (T,)
    assert tuple(est.njt.shape) == (T, J)
    assert tuple(est.gamma.shape) == (T, J)
    assert tuple(est.phi.shape) == (T,)


def _assert_state_finite(est: ChoiceLearnShrinkageEstimator) -> None:
    assert_all_finite_tf(
        est.alpha,
        est.E_bar,
        est.njt,
        est.gamma,
        est.phi,
    )


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


def _patch_tuning_and_debug(monkeypatch) -> None:
    """
    Patch tuning and progress printing so tests focus on validation and wiring.
    """
    monkeypatch.setattr(
        cl_diagnostics_mod, "report_iteration_progress", lambda *args, **kwargs: None
    )

    def _stub_tune_shrinkage(_shrink):
        # Slightly larger than 0.1 to encourage movement in the round-trip test.
        k = tf.constant(0.2, dtype=tf.float64)
        return k, k, k

    monkeypatch.setattr(cl_shrinkage_mod, "tune_shrinkage", _stub_tune_shrinkage)


# -----------------------------------------------------------------------------
# Input-level validation tests (public entrypoints)
# -----------------------------------------------------------------------------
def test_init_raises_on_shape_or_rank_mismatch(tiny_cl_data):
    delta_cl = tiny_cl_data["delta_cl"]
    qjt = tiny_cl_data["qjt"]
    q0t = tiny_cl_data["q0t"]
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    base = dict(delta_cl=delta_cl, qjt=qjt, q0t=q0t, seed=0)

    bad_cases = [
        dict(delta_cl=delta_cl[:, : J - 1]),
        dict(qjt=qjt[:, : J - 1]),
        dict(delta_cl=np.vstack([delta_cl, delta_cl[:1]])),  # (T+1,J)
        dict(qjt=qjt[:1, :]),  # (T-1,J)
        dict(q0t=q0t[: T - 1]),  # (T-1,)
        dict(q0t=q0t.reshape(T, 1)),  # (T,1)
        dict(delta_cl=delta_cl[0]),  # (J,)
        dict(qjt=qjt[0]),  # (J,)
    ]

    for overrides in bad_cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(Exception):
            ChoiceLearnShrinkageEstimator(**kwargs)


def test_fit_raises_on_invalid_arguments(estimator, monkeypatch):
    _patch_tuning_and_debug(monkeypatch)

    base = dict(
        n_iter=2,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
    )

    bad_overrides = [
        dict(n_iter=0),
        dict(n_iter=-1),
        dict(pilot_length=0),
        dict(pilot_length=-1),
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
    ]

    for overrides in bad_overrides:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(Exception):
            estimator.fit(**kwargs)


# -----------------------------------------------------------------------------
# Behavioral tests
# -----------------------------------------------------------------------------
def test_init_state_shapes_and_defaults(estimator, tiny_cl_data):
    est = estimator
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    assert est.T == T
    assert est.J == J
    _assert_state_shapes(est, T, J)

    # Defaults explicitly set in __init__
    assert float(est.alpha.numpy()) == 1.0
    assert np.allclose(est.njt.numpy(), 0.0)
    assert np.allclose(est.gamma.numpy(), 0.0)

    # phi initialized to Beta prior mean a/(a+b)
    phi0 = (est.posterior.a_phi / (est.posterior.a_phi + est.posterior.b_phi)).numpy()
    assert np.allclose(est.phi.numpy(), float(phi0))
    assert_in_open_unit_interval_tf(est.phi)

    # E_bar initialized to posterior.E_bar_mean
    e0 = est.posterior.E_bar_mean.numpy()
    assert np.allclose(est.E_bar.numpy(), float(e0))


def test_mcmc_iteration_step_updates_state(estimator, tiny_cl_data):
    est = estimator
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    k = tf.constant(0.2, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    est._mcmc_iteration_step(
        it=tf.constant(0, dtype=tf.int32),
        k_alpha=k,
        k_E_bar=k,
        k_njt=k,
        ridge=ridge,
    )

    _assert_state_shapes(est, T, J)
    _assert_state_finite(est)
    assert_binary_01_tf(est.gamma)
    assert_in_open_unit_interval_tf(est.phi)


def test_run_mcmc_loop_saved_equals_n_iter(estimator, tiny_cl_data, monkeypatch):
    est = estimator
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    _patch_tuning_and_debug(monkeypatch)

    diag = ChoiceLearnShrinkageDiagnostics(T=T, J=J)
    k = tf.constant(0.2, dtype=tf.float64)

    n_iter = 3
    est._run_mcmc_loop(
        n_iter=n_iter,
        k_alpha=k,
        k_E_bar=k,
        k_njt=k,
        ridge=1e-6,
        diag=diag,
    )

    saved, *_ = diag.get_sums()
    assert int(saved.numpy()) == n_iter


def test_fit_runs_with_mocked_tuning(estimator, tiny_cl_data, monkeypatch):
    est = estimator
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    _patch_tuning_and_debug(monkeypatch)

    n_iter = 2
    est.fit(
        n_iter=n_iter,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
    )

    assert est._diag is not None
    saved, *_ = est._diag.get_sums()
    assert int(saved.numpy()) == n_iter

    _assert_state_shapes(est, T, J)
    _assert_state_finite(est)
    assert_binary_01_tf(est.gamma)
    assert_in_open_unit_interval_tf(est.phi)


def test_fit_round_trip_improves_share_fit(monkeypatch, tiny_cl_data):
    """
    Round-trip (simple): generate qjt,q0t from known parameters using the same
    choice-probability mapping used in the posterior, then fit and check share
    misfit decreases.
    """
    _patch_tuning_and_debug(monkeypatch)

    T, J = tiny_cl_data["T"], tiny_cl_data["J"]
    delta_cl = tf.constant(tiny_cl_data["delta_cl"], tf.float64)  # (T,J)

    # Rebuild the same synthetic counts deterministically (so this test is standalone)
    alpha_true = tf.constant(1.8, tf.float64)
    E_bar_true = tf.constant([0.35, -0.25], tf.float64)
    njt_true = tf.constant([[0.20, -0.10, 0.00], [-0.08, 0.14, 0.06]], tf.float64)

    posterior = LuPosteriorTF(dtype=tf.float64)

    def _choice_probs_batch(alpha, E_bar, njt):
        sjt_list, s0_list = [], []
        for t in range(T):
            delta_t = posterior._mean_utility_jt(
                delta_cl_t=delta_cl[t],
                alpha=alpha,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )
            sjt_t, s0t = posterior._choice_probs_t(delta_t=delta_t)
            sjt_list.append(sjt_t)
            s0_list.append(s0t)
        return tf.stack(sjt_list, axis=0), tf.stack(s0_list, axis=0)

    sjt_true, s0_true = _choice_probs_batch(alpha_true, E_bar_true, njt_true)

    N = tf.constant(5000.0, tf.float64)
    qjt = (N * sjt_true).numpy()
    q0t = (N * s0_true).numpy()

    Nt = q0t + np.sum(qjt, axis=1)
    s_obs = qjt / Nt[:, None]
    s0_obs = q0t / Nt

    def _misfit(est: ChoiceLearnShrinkageEstimator) -> float:
        sjt_hat, s0_hat = _choice_probs_batch(
            est.alpha.read_value(),
            est.E_bar.read_value(),
            est.njt.read_value(),
        )
        sjt_hat = sjt_hat.numpy()
        s0_hat = s0_hat.numpy()
        return float(np.sum(np.abs(sjt_hat - s_obs)) + np.sum(np.abs(s0_hat - s0_obs)))

    est = ChoiceLearnShrinkageEstimator(
        delta_cl=delta_cl.numpy(),
        qjt=qjt,
        q0t=q0t,
        seed=123,
    )

    misfit0 = _misfit(est)

    est.fit(
        n_iter=40,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
    )

    misfit1 = _misfit(est)
    assert misfit1 < 0.99 * misfit0


def test_get_results_shapes_and_E_identity(estimator, tiny_cl_data, monkeypatch):
    est = estimator
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    _patch_tuning_and_debug(monkeypatch)

    est.fit(
        n_iter=2,
        pilot_length=2,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=2,
        factor_rw=1.1,
        factor_tmh=1.5,
    )

    res = est.get_results()
    _assert_results_schema(res, T, J)

    E_hat = np.asarray(res["E_hat"])
    E_bar_hat = np.asarray(res["E_bar_hat"])
    njt_hat = np.asarray(res["njt_hat"])

    assert np.all(np.isfinite(E_hat))
    assert np.all(np.isfinite(E_bar_hat))
    assert np.all(np.isfinite(njt_hat))
    assert np.isfinite(res["alpha_hat"])

    assert np.allclose(E_hat, E_bar_hat[:, None] + njt_hat)


# -----------------------------------------------------------------------------
# Permutation invariance (deterministic): posterior log-density
# -----------------------------------------------------------------------------
def test_logpost_is_equivariant_to_product_permutation(tiny_cl_data):
    """
    If we permute the product index j (columns) in all product-indexed inputs
    (delta_cl, qjt) and permute the product-indexed latent states (njt, gamma)
    the same way, then the per-market log posterior vector should be unchanged
    (up to floating-point tolerance).
    """
    T, J = tiny_cl_data["T"], tiny_cl_data["J"]

    delta_cl = tf.constant(tiny_cl_data["delta_cl"], dtype=tf.float64)  # (T,J)
    qjt = tf.constant(tiny_cl_data["qjt"], dtype=tf.float64)  # (T,J)
    q0t = tf.constant(tiny_cl_data["q0t"], dtype=tf.float64)  # (T,)

    alpha = tf.constant(1.2, dtype=tf.float64)
    E_bar = tf.cast(tf.linspace(-0.05, 0.05, T), tf.float64)  # (T,)

    dc_c = delta_cl - tf.reduce_mean(delta_cl, axis=1, keepdims=True)
    njt = 0.03 * dc_c  # (T,J)

    gamma = tf.cast(njt > 0.0, tf.float64)  # (T,J)
    phi = tf.fill([T], tf.constant(0.4, dtype=tf.float64))  # (T,)

    posterior = LuPosteriorTF(dtype=tf.float64)

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

    if J == 3:
        perm = tf.constant([2, 0, 1], dtype=tf.int32)
    else:
        perm = tf.constant(list(range(1, J)) + [0], dtype=tf.int32)

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
