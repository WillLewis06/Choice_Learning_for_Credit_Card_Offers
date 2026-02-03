# tests/test_lu_shrinkage.py
import numpy as np
import pytest
import tensorflow as tf

from market_shock_estimators.lu_diagnostics import LuShrinkageDiagnostics
import market_shock_estimators.lu_shrinkage as lu_shrinkage_mod
from market_shock_estimators.lu_shrinkage import LuShrinkageEstimator


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_market_data():
    """
    Minimal (T,J) market data that avoids obvious degeneracies.
    """
    T, J = 2, 3

    # Prices and characteristics: float
    pjt = np.array([[1.0, 1.2, 0.8], [1.1, 0.9, 1.3]], dtype=np.float64)
    wjt = np.array([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=np.float64)

    # Quantities: nonnegative (treated as counts in the likelihood)
    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return {"pjt": pjt, "wjt": wjt, "qjt": qjt, "q0t": q0t, "T": T, "J": J}


@pytest.fixture
def estimator(tiny_market_data):
    est = LuShrinkageEstimator(
        pjt=tiny_market_data["pjt"],
        wjt=tiny_market_data["wjt"],
        qjt=tiny_market_data["qjt"],
        q0t=tiny_market_data["q0t"],
        n_draws=20,
        seed=123,
    )
    return est


# -----------------------------------------------------------------------------
# Test-only helpers
# -----------------------------------------------------------------------------
def _assert_state_shapes(est: LuShrinkageEstimator, T: int, J: int) -> None:
    assert est.beta_p.shape == ()
    assert est.beta_w.shape == ()
    assert est.r.shape == ()
    assert tuple(est.E_bar.shape) == (T,)
    assert tuple(est.njt.shape) == (T, J)
    assert tuple(est.gamma.shape) == (T, J)
    assert tuple(est.phi.shape) == (T,)


def _assert_state_finite(est: LuShrinkageEstimator) -> None:
    def _finite(x):
        x = tf.convert_to_tensor(x)
        return bool(tf.reduce_all(tf.math.is_finite(x)).numpy())

    assert _finite(est.beta_p)
    assert _finite(est.beta_w)
    assert _finite(est.r)
    assert _finite(est.E_bar)
    assert _finite(est.njt)
    assert _finite(est.gamma)
    assert _finite(est.phi)


def _assert_results_schema(res: dict, T: int, J: int) -> None:
    # Scalars
    for k in ["beta_p_hat", "beta_w_hat", "sigma_hat", "n_saved"]:
        assert k in res
    # Arrays
    for k in ["E_hat", "E_bar_hat", "njt_hat", "phi_hat", "gamma_hat"]:
        assert k in res

    assert np.shape(res["E_bar_hat"]) == (T,)
    assert np.shape(res["phi_hat"]) == (T,)
    assert np.shape(res["njt_hat"]) == (T, J)
    assert np.shape(res["gamma_hat"]) == (T, J)
    assert np.shape(res["E_hat"]) == (T, J)


def _patch_tuning_and_debug(monkeypatch):
    """
    Patch tuning and debug I/O so tests focus on validation and core wiring.
    """
    monkeypatch.setattr(
        LuShrinkageEstimator, "_debug_save_k", lambda *args, **kwargs: None
    )

    def _stub_tune_shrinkage(_shrink):
        k = tf.constant(0.1, dtype=tf.float64)
        return k, k, k, k

    monkeypatch.setattr(lu_shrinkage_mod, "tune_shrinkage", _stub_tune_shrinkage)


# -----------------------------------------------------------------------------
# Input-level validation tests (public entrypoints)
# -----------------------------------------------------------------------------
def test_init_raises_on_shape_or_rank_mismatch(tiny_market_data):
    """
    __init__ converts inputs to tf.float64 before validation, so dtype/type errors
    are not meaningfully testable at the entrypoint. What remains is shape/rank
    consistency across (T,J) arrays and q0t (T,).
    """
    pjt = tiny_market_data["pjt"]
    wjt = tiny_market_data["wjt"]
    qjt = tiny_market_data["qjt"]
    q0t = tiny_market_data["q0t"]
    T, J = tiny_market_data["T"], tiny_market_data["J"]

    base = dict(pjt=pjt, wjt=wjt, qjt=qjt, q0t=q0t, n_draws=10, seed=0)

    bad_cases = [
        ("pjt_wrong_J", dict(pjt=pjt[:, : J - 1])),
        ("wjt_wrong_J", dict(wjt=wjt[:, : J - 1])),
        ("qjt_wrong_J", dict(qjt=qjt[:, : J - 1])),
        ("pjt_wrong_T", dict(pjt=np.vstack([pjt, pjt[:1]]))),  # (T+1,J)
        ("qjt_wrong_T", dict(qjt=qjt[:1, :])),  # (T-1,J)
        ("q0t_wrong_len", dict(q0t=q0t[: T - 1])),  # (T-1,)
        ("q0t_rank2", dict(q0t=q0t.reshape(T, 1))),  # (T,1)
        ("pjt_rank1", dict(pjt=pjt[0])),  # (J,)
        ("wjt_rank1", dict(wjt=wjt[0])),  # (J,)
        ("qjt_rank1", dict(qjt=qjt[0])),  # (J,)
    ]

    for _, overrides in bad_cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(Exception):
            LuShrinkageEstimator(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
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
    ],
)
def test_fit_raises_on_invalid_arguments(estimator, monkeypatch, overrides):
    """
    fit() should fail fast on invalid arguments via fit_validate_input.
    """
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
    base.update(overrides)

    with pytest.raises(Exception):
        estimator.fit(**base)


# -----------------------------------------------------------------------------
# Existing behavioral tests
# -----------------------------------------------------------------------------
def test_init_state_shapes_and_defaults(estimator, tiny_market_data):
    est = estimator
    T, J = tiny_market_data["T"], tiny_market_data["J"]

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

    # E_bar initialized to posterior.E_bar_mean
    e0 = est.posterior.E_bar_mean.numpy()
    assert np.allclose(est.E_bar.numpy(), float(e0))


def test_mcmc_iteration_step_requires_diag_set(estimator):
    est = estimator
    # Calling compiled step without setting est._diag should fail,
    # because the step calls `self._diag.step(...)`.
    k = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    with pytest.raises(Exception):
        est._mcmc_iteration_step(
            it=tf.constant(0, dtype=tf.int32),
            k_beta=k,
            k_njt=k,
            k_r=k,
            k_E_bar=k,
            ridge=ridge,
        )


def test_mcmc_iteration_step_updates_state_and_increments_saved(
    estimator, tiny_market_data
):
    est = estimator
    T, J = tiny_market_data["T"], tiny_market_data["J"]

    diag = LuShrinkageDiagnostics(T=T, J=J)
    est._diag = diag

    k = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    saved0, *_ = diag.get_sums()
    saved0 = int(saved0.numpy())

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

    saved1, *_ = diag.get_sums()
    saved1 = int(saved1.numpy())
    assert saved1 == saved0 + 1


def test_run_mcmc_loop_saved_equals_n_iter(estimator, tiny_market_data):
    est = estimator
    T, J = tiny_market_data["T"], tiny_market_data["J"]

    diag = LuShrinkageDiagnostics(T=T, J=J)
    k = tf.constant(0.1, dtype=tf.float64)

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


def test_fit_runs_with_mocked_tuning(estimator, tiny_market_data, monkeypatch):
    est = estimator
    T, J = tiny_market_data["T"], tiny_market_data["J"]

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


def test_fit_round_trip_improves_share_fit(monkeypatch, tiny_market_data):
    """
    Round-trip (simple): generate qjt,q0t from known parameters using the same
    per-market choice-probability mapping used in the posterior, then fit and
    check share misfit decreases.
    """
    import numpy as np
    import tensorflow as tf

    _patch_tuning_and_debug(monkeypatch)

    pjt = tf.constant(tiny_market_data["pjt"], dtype=tf.float64)  # (T,J)
    wjt = tf.constant(tiny_market_data["wjt"], dtype=tf.float64)  # (T,J)
    T, J = tiny_market_data["T"], tiny_market_data["J"]

    # "True" parameters for generating internally-consistent shares
    beta_p_true = tf.constant(-1.2, dtype=tf.float64)
    beta_w_true = tf.constant(0.6, dtype=tf.float64)
    r_true = tf.constant(0.0, dtype=tf.float64)  # sigma=1
    E_bar_true = tf.constant([0.3, -0.2], dtype=tf.float64)  # (T,)
    njt_true = tf.constant([[0.2, -0.1, 0.0], [-0.1, 0.15, 0.05]], dtype=tf.float64)

    def _batched_choice_probs(posterior, pjt, wjt, beta_p, beta_w, r, E_bar, njt):
        sjt_list = []
        s0_list = []
        for t in range(int(pjt.shape[0])):
            delta_t = posterior._mean_utility_jt(
                pjt_t=pjt[t],
                wjt_t=wjt[t],
                beta_p=beta_p,
                beta_w=beta_w,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )
            sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt[t], delta_t=delta_t, r=r)
            sjt_list.append(sjt_t)  # (J,)
            s0_list.append(s0t)  # scalar
        return tf.stack(sjt_list, axis=0), tf.stack(s0_list, axis=0)  # (T,J), (T,)

    # Temporary estimator only to access a correctly-configured posterior (draws etc.)
    tmp = LuShrinkageEstimator(
        pjt=tiny_market_data["pjt"],
        wjt=tiny_market_data["wjt"],
        qjt=tiny_market_data["qjt"],
        q0t=tiny_market_data["q0t"],
        n_draws=20,
        seed=123,
    )

    # Shares implied by the true parameters
    sjt_true, s0_true = _batched_choice_probs(
        posterior=tmp.posterior,
        pjt=pjt,
        wjt=wjt,
        beta_p=beta_p_true,
        beta_w=beta_w_true,
        r=r_true,
        E_bar=E_bar_true,
        njt=njt_true,
    )

    # Convert shares to counts with large N
    N = tf.constant(5000.0, dtype=tf.float64)
    qjt = (N * sjt_true).numpy()
    q0t = (N * s0_true).numpy()

    # Observed shares from counts (target)
    Nt = q0t + np.sum(qjt, axis=1)
    s_obs = qjt / Nt[:, None]
    s0_obs = q0t / Nt

    def _misfit(est: LuShrinkageEstimator) -> float:
        sjt_hat, s0_hat = _batched_choice_probs(
            posterior=est.posterior,
            pjt=pjt,
            wjt=wjt,
            beta_p=est.beta_p.read_value(),
            beta_w=est.beta_w.read_value(),
            r=est.r.read_value(),
            E_bar=est.E_bar.read_value(),
            njt=est.njt.read_value(),
        )
        sjt_hat = sjt_hat.numpy()
        s0_hat = s0_hat.numpy()
        return float(np.sum(np.abs(sjt_hat - s_obs)) + np.sum(np.abs(s0_hat - s0_obs)))

    est = LuShrinkageEstimator(
        pjt=tiny_market_data["pjt"],
        wjt=tiny_market_data["wjt"],
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


def test_get_results_shapes_and_E_identity(estimator, tiny_market_data, monkeypatch):
    est = estimator
    T, J = tiny_market_data["T"], tiny_market_data["J"]

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
    assert np.isfinite(res["sigma_hat"])

    assert np.allclose(E_hat, E_bar_hat[:, None] + njt_hat)
