"""
Unit tests for the LuShrinkageEstimator (Lu shrinkage MCMC sampler).

Constraints for this test module
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared test utilities and tiny-problem builders are imported from `lu_conftest`
  (a normal Python helper module, not a pytest conftest plugin).
- Tests that call `fit()` patch:
  - tuning (to keep tests fast and deterministic), and
  - debug file I/O (to avoid writing to disk during tests).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import lu.shrinkage.lu_shrinkage as lu_shrinkage_mod
from lu.shrinkage.lu_diagnostics import LuShrinkageDiagnostics
from lu.shrinkage.lu_posterior import LuPosteriorTF
from lu.shrinkage.lu_shrinkage import LuShrinkageEstimator

from lu_conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
    assert_in_open_unit_interval_tf,
    tiny_market_np,
)

DTYPE = tf.float64
ATOL = 1e-10


# -----------------------------------------------------------------------------
# Local constructors / helpers (no pytest fixtures)
# -----------------------------------------------------------------------------
def _make_estimator(
    data: dict,
    n_draws: int = 20,
    seed: int = 123,
) -> LuShrinkageEstimator:
    """Construct a LuShrinkageEstimator on the provided market panel."""
    return LuShrinkageEstimator(
        pjt=data["pjt"],
        wjt=data["wjt"],
        qjt=data["qjt"],
        q0t=data["q0t"],
        n_draws=n_draws,
        seed=seed,
    )


def _assert_state_shapes(est: LuShrinkageEstimator, T: int, J: int) -> None:
    """Basic shape contract for the sampler state."""
    assert est.beta_p.shape == ()
    assert est.beta_w.shape == ()
    assert est.r.shape == ()
    assert tuple(est.E_bar.shape) == (T,)
    assert tuple(est.njt.shape) == (T, J)
    assert tuple(est.gamma.shape) == (T, J)
    assert tuple(est.phi.shape) == (T,)


def _assert_state_finite(est: LuShrinkageEstimator) -> None:
    """All state components should be finite."""
    assert_all_finite_tf(
        est.beta_p,
        est.beta_w,
        est.r,
        est.E_bar,
        est.njt,
        est.gamma,
        est.phi,
    )


def _assert_results_schema(res: dict, T: int, J: int) -> None:
    """Minimal schema contract for get_results()."""
    for k in ["beta_p_hat", "beta_w_hat", "sigma_hat", "n_saved"]:
        assert k in res
    for k in ["E_hat", "E_bar_hat", "njt_hat", "phi_hat", "gamma_hat"]:
        assert k in res

    assert np.shape(res["E_bar_hat"]) == (T,)
    assert np.shape(res["phi_hat"]) == (T,)
    assert np.shape(res["njt_hat"]) == (T, J)
    assert np.shape(res["gamma_hat"]) == (T, J)
    assert np.shape(res["E_hat"]) == (T, J)


def _noop_debug_save_k(self, k_r, k_E_bar, k_beta, k_njt) -> None:
    """Disable debug file writes in tests."""
    return None


def _stub_tune_shrinkage(_shrink: LuShrinkageEstimator):
    """Return fixed proposal scales so fit() is fast and deterministic in tests."""
    k = tf.constant(0.1, dtype=DTYPE)
    return k, k, k, k


@contextmanager
def _patched_tuning_and_debug():
    """
    Patch:
    - LuShrinkageEstimator._debug_save_k (disk I/O)
    - lu_shrinkage_mod.tune_shrinkage (pilot tuning loop)
    """
    with patch.object(LuShrinkageEstimator, "_debug_save_k", _noop_debug_save_k):
        with patch.object(lu_shrinkage_mod, "tune_shrinkage", _stub_tune_shrinkage):
            yield


# -----------------------------------------------------------------------------
# Input-level validation tests (public entrypoints)
# -----------------------------------------------------------------------------
def test_init_raises_on_shape_or_rank_mismatch():
    """
    __init__ converts inputs to tf.float64 before validation, so dtype/type errors
    are not meaningfully testable at the entrypoint. What remains is shape/rank
    consistency across (T,J) arrays and q0t (T,).
    """
    data = tiny_market_np()
    pjt = data["pjt"]
    wjt = data["wjt"]
    qjt = data["qjt"]
    q0t = data["q0t"]
    T, J = data["T"], data["J"]

    base = dict(pjt=pjt, wjt=wjt, qjt=qjt, q0t=q0t, n_draws=10, seed=0)

    bad_cases = [
        dict(pjt=pjt[:, : J - 1]),
        dict(wjt=wjt[:, : J - 1]),
        dict(qjt=qjt[:, : J - 1]),
        dict(pjt=np.vstack([pjt, pjt[:1]])),  # (T+1,J)
        dict(qjt=qjt[:1, :]),  # (T-1,J)
        dict(q0t=q0t[: T - 1]),  # (T-1,)
        dict(q0t=q0t.reshape(T, 1)),  # (T,1)
        dict(pjt=pjt[0]),  # (J,)
        dict(wjt=wjt[0]),  # (J,)
        dict(qjt=qjt[0]),  # (J,)
    ]

    for overrides in bad_cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(Exception):
            LuShrinkageEstimator(**kwargs)


def test_fit_raises_on_invalid_arguments():
    """
    fit() should fail fast on invalid arguments via fit_validate_input.

    This test intentionally does not patch tuning or debug I/O because validation
    happens before those paths are reached.
    """
    data = tiny_market_np()
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
            est.fit(**kwargs)


# -----------------------------------------------------------------------------
# Behavioral tests
# -----------------------------------------------------------------------------
def test_init_state_shapes_and_defaults():
    data = tiny_market_np()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    assert est.T == T
    assert est.J == J
    _assert_state_shapes(est, T, J)

    # Defaults explicitly set in __init__
    assert float(est.beta_p.numpy()) == 0.0
    assert float(est.beta_w.numpy()) == 0.0
    assert float(est.r.numpy()) == 0.0

    assert np.allclose(est.njt.numpy(), 0.0)
    assert np.allclose(est.gamma.numpy(), 0.0)

    # phi initialized to Beta prior mean a/(a+b)
    phi0 = (est.posterior.a_phi / (est.posterior.a_phi + est.posterior.b_phi)).numpy()
    assert np.allclose(est.phi.numpy(), float(phi0))
    assert_in_open_unit_interval_tf(est.phi)

    # E_bar initialized to posterior.E_bar_mean
    e0 = est.posterior.E_bar_mean.numpy()
    assert np.allclose(est.E_bar.numpy(), float(e0))


def test_mcmc_iteration_step_requires_diag_set():
    data = tiny_market_np()
    est = _make_estimator(data)

    k = tf.constant(0.1, dtype=DTYPE)
    ridge = tf.constant(1e-6, dtype=DTYPE)

    with pytest.raises(Exception):
        est._mcmc_iteration_step(
            it=tf.constant(0, dtype=tf.int32),
            k_beta=k,
            k_njt=k,
            k_r=k,
            k_E_bar=k,
            ridge=ridge,
        )


def test_mcmc_iteration_step_updates_state_and_increments_saved():
    data = tiny_market_np()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    diag = LuShrinkageDiagnostics(T=T, J=J)
    est._diag = diag

    k = tf.constant(0.1, dtype=DTYPE)
    ridge = tf.constant(1e-6, dtype=DTYPE)

    saved0, *_ = diag.get_sums()
    saved0_i = int(saved0.numpy())

    est._mcmc_iteration_step(
        it=tf.constant(0, dtype=tf.int32),
        k_beta=k,
        k_njt=k,
        k_r=k,
        k_E_bar=k,
        ridge=ridge,
    )

    _assert_state_shapes(est, T, J)
    _assert_state_finite(est)
    assert_binary_01_tf(est.gamma)
    assert_in_open_unit_interval_tf(est.phi)

    saved1, *_ = diag.get_sums()
    saved1_i = int(saved1.numpy())
    assert saved1_i == saved0_i + 1


def test_run_mcmc_loop_saved_equals_n_iter():
    data = tiny_market_np()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    diag = LuShrinkageDiagnostics(T=T, J=J)
    k = tf.constant(0.1, dtype=DTYPE)

    n_iter = 3
    est._run_mcmc_loop(
        n_iter=n_iter,
        k_beta=k,
        k_njt=k,
        k_r=k,
        k_E_bar=k,
        ridge=1e-6,
        diag=diag,
    )

    saved, *_ = diag.get_sums()
    assert int(saved.numpy()) == n_iter


def test_fit_runs_with_mocked_tuning():
    data = tiny_market_np()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    with _patched_tuning_and_debug():
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


def test_fit_round_trip_improves_share_fit():
    """
    Round-trip (simple): generate qjt,q0t from known parameters using the same
    per-market choice-probability mapping used in the posterior, then fit and
    check share misfit decreases.
    """
    data = tiny_market_np()

    with _patched_tuning_and_debug():
        pjt = tf.constant(data["pjt"], dtype=DTYPE)  # (T,J)
        wjt = tf.constant(data["wjt"], dtype=DTYPE)  # (T,J)
        T, J = data["T"], data["J"]

        beta_p_true = tf.constant(-1.2, dtype=DTYPE)
        beta_w_true = tf.constant(0.6, dtype=DTYPE)
        r_true = tf.constant(0.0, dtype=DTYPE)  # sigma=1

        E_bar_true = tf.linspace(
            tf.constant(0.3, dtype=DTYPE), tf.constant(-0.2, dtype=DTYPE), T
        )

        p_c = pjt - tf.reduce_mean(pjt, axis=1, keepdims=True)
        w_c = wjt - tf.reduce_mean(wjt, axis=1, keepdims=True)
        njt_true = 0.15 * p_c - 0.10 * w_c

        posterior_sim = LuPosteriorTF(n_draws=25, seed=123, dtype=DTYPE)

        def _choice_probs_batch(beta_p, beta_w, r, E_bar, njt):
            sjt_list = []
            s0_list = []
            for t in range(T):
                delta_t = posterior_sim._mean_utility_jt(
                    pjt_t=pjt[t],
                    wjt_t=wjt[t],
                    beta_p=beta_p,
                    beta_w=beta_w,
                    E_bar_t=E_bar[t],
                    njt_t=njt[t],
                )
                sjt_t, s0t = posterior_sim._choice_probs_t(
                    pjt_t=pjt[t], delta_t=delta_t, r=r
                )
                sjt_list.append(sjt_t)
                s0_list.append(s0t)
            return tf.stack(sjt_list, axis=0), tf.stack(s0_list, axis=0)

        sjt_true, s0_true = _choice_probs_batch(
            beta_p_true, beta_w_true, r_true, E_bar_true, njt_true
        )

        N = tf.constant(5000.0, dtype=DTYPE)
        qjt = (N * sjt_true).numpy()
        q0t = (N * s0_true).numpy()

        Nt = q0t + np.sum(qjt, axis=1)
        s_obs = qjt / Nt[:, None]
        s0_obs = q0t / Nt

        def _misfit(estimator: LuShrinkageEstimator) -> float:
            sjt_hat, s0_hat = _choice_probs_batch(
                estimator.beta_p.read_value(),
                estimator.beta_w.read_value(),
                estimator.r.read_value(),
                estimator.E_bar.read_value(),
                estimator.njt.read_value(),
            )
            sj = sjt_hat.numpy()
            s0 = s0_hat.numpy()
            return float(np.sum(np.abs(sj - s_obs)) + np.sum(np.abs(s0 - s0_obs)))

        est = LuShrinkageEstimator(
            pjt=data["pjt"],
            wjt=data["wjt"],
            qjt=qjt,
            q0t=q0t,
            n_draws=20,
            seed=123,
        )

        misfit0 = _misfit(est)

        est.fit(
            n_iter=30,
            pilot_length=2,
            ridge=1e-6,
            target_low=0.3,
            target_high=0.5,
            max_rounds=2,
            factor_rw=1.1,
            factor_tmh=1.5,
        )

        misfit1 = _misfit(est)

    assert misfit1 < 0.98 * misfit0


def test_get_results_shapes_and_E_identity():
    data = tiny_market_np()
    est = _make_estimator(data)
    T, J = data["T"], data["J"]

    with _patched_tuning_and_debug():
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
    assert np.isfinite(res["sigma_hat"])

    assert np.allclose(E_hat, E_bar_hat[:, None] + njt_hat, atol=ATOL, rtol=0.0)


# -----------------------------------------------------------------------------
# Permutation invariance (deterministic): posterior log-density
# -----------------------------------------------------------------------------
def test_logpost_is_equivariant_to_product_permutation():
    """
    Deterministic equivariance check at the posterior level.

    If we permute the product index j (columns) in all product-indexed inputs
    (pjt, wjt, qjt) and permute the product-indexed latent states (njt, gamma)
    the same way, then the per-market log posterior vector should be unchanged
    (up to floating-point tolerance).
    """
    data = tiny_market_np()
    T, J = data["T"], data["J"]

    pjt = tf.constant(data["pjt"], dtype=DTYPE)  # (T,J)
    wjt = tf.constant(data["wjt"], dtype=DTYPE)  # (T,J)
    qjt = tf.constant(data["qjt"], dtype=DTYPE)  # (T,J)
    q0t = tf.constant(data["q0t"], dtype=DTYPE)  # (T,)

    beta_p = tf.constant(0.15, dtype=DTYPE)
    beta_w = tf.constant(-0.08, dtype=DTYPE)
    r = tf.constant(0.30, dtype=DTYPE)  # log(sigma)

    E_bar = tf.cast(tf.linspace(-0.05, 0.05, T), DTYPE)  # (T,)

    p_c = pjt - tf.reduce_mean(pjt, axis=1, keepdims=True)
    w_c = wjt - tf.reduce_mean(wjt, axis=1, keepdims=True)
    njt = 0.01 * p_c + 0.02 * w_c  # (T,J)

    gamma = tf.cast(njt > 0.0, DTYPE)  # (T,J)
    phi = tf.fill([T], tf.constant(0.4, dtype=DTYPE))  # (T,)

    posterior = LuPosteriorTF(n_draws=25, seed=0, dtype=DTYPE)

    lp0 = posterior.logpost_vec(
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
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

    pjt_p = tf.gather(pjt, perm, axis=1)
    wjt_p = tf.gather(wjt, perm, axis=1)
    qjt_p = tf.gather(qjt, perm, axis=1)
    njt_p = tf.gather(njt, perm, axis=1)
    gamma_p = tf.gather(gamma, perm, axis=1)

    lp1 = posterior.logpost_vec(
        qjt=qjt_p,
        q0t=q0t,
        pjt=pjt_p,
        wjt=wjt_p,
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
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
