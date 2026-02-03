# tests/test_lu_tuning.py
import numpy as np
import pytest
import tensorflow as tf

import market_shock_estimators.lu_tuning as lu_tuning
from market_shock_estimators.lu_shrinkage import LuShrinkageEstimator


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _assert_all_finite(*xs) -> None:
    for x in xs:
        x = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x))
        assert bool(ok.numpy()), "Found non-finite values."


def _assert_scalar_positive(x: tf.Tensor) -> None:
    x = tf.convert_to_tensor(x)
    assert x.shape == ()
    _assert_all_finite(x)
    assert float(x.numpy()) > 0.0


def _snapshot_state(shrink: LuShrinkageEstimator) -> dict:
    # Copy all sampler state variables that must NOT be mutated by tuning.
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

    assert np.allclose(before["E_bar"], after["E_bar"], atol=0.0, rtol=0.0)
    assert np.allclose(before["njt"], after["njt"], atol=0.0, rtol=0.0)

    # gamma is binary float; should be exactly equal if no mutation occurred
    assert np.array_equal(before["gamma"], after["gamma"])
    assert np.allclose(before["phi"], after["phi"], atol=0.0, rtol=0.0)


def _make_shrink(
    pjt_np: np.ndarray,
    wjt_np: np.ndarray,
    qjt_np: np.ndarray,
    q0t_np: np.ndarray,
    *,
    n_draws: int = 25,
    seed: int = 123,
    pilot_length: int = 1,
    ridge: float = 1e-6,
    target_low: float = 0.3,
    target_high: float = 0.5,
    max_rounds: int = 1,
    factor_rw: float = 1.1,
    factor_tmh: float = 1.5,
) -> LuShrinkageEstimator:
    shrink = LuShrinkageEstimator(
        pjt=pjt_np,
        wjt=wjt_np,
        qjt=qjt_np,
        q0t=q0t_np,
        n_draws=n_draws,
        seed=seed,
    )

    # Tuning controls required by tune_shrinkage_validate_input
    shrink.pilot_length = pilot_length
    shrink.ridge = ridge
    shrink.target_low = target_low
    shrink.target_high = target_high
    shrink.max_rounds = max_rounds
    shrink.factor_rw = factor_rw
    shrink.factor_tmh = factor_tmh

    return shrink


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def dummy_shrinkage_estimator():
    # Tiny, valid problem: T=2, J=3
    pjt = np.array([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=np.float64)
    wjt = np.array([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=np.float64)
    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return _make_shrink(pjt, wjt, qjt, q0t)


# -----------------------------------------------------------------------------
# tune_k unit tests
# -----------------------------------------------------------------------------
def test_tune_k_validate_input_rejects_invalid_args():
    theta0 = tf.constant(0.0, tf.float64)

    def step_fn(theta, k):
        return theta, tf.constant(0.0, tf.float64)

    k0 = tf.constant(1.0, tf.float64)

    # pilot_length <= 0
    with pytest.raises(Exception):
        lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=0,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name="x",
        )

    # max_rounds <= 0
    with pytest.raises(Exception):
        lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=0,
            factor=1.1,
            name="x",
        )

    # invalid probability band (low > high)
    with pytest.raises(Exception):
        lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=1,
            target_low=0.6,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name="x",
        )

    # factor <= 1
    with pytest.raises(Exception):
        lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.0,
            name="x",
        )

    # name must be str
    with pytest.raises(Exception):
        lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name=123,  # type: ignore[arg-type]
        )


def test_tune_k_shrinks_k_when_acceptance_below_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)
    pilot_length = 2
    max_rounds = 3
    factor = 1.1

    def step_fn(theta, k):
        # Always reject
        return theta, tf.constant(0.0, tf.float64)

    k_out = lu_tuning.tune_k(
        theta0=theta0,
        step_fn=step_fn,
        k0=k0,
        pilot_length=pilot_length,
        target_low=0.3,
        target_high=0.5,
        max_rounds=max_rounds,
        factor=factor,
        name="reject",
    )

    expected = 1.0 / (factor**max_rounds)
    assert abs(float(k_out.numpy()) - expected) < 1e-12


def test_tune_k_grows_k_when_acceptance_above_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)
    pilot_length = 2
    max_rounds = 3
    factor = 1.1

    def step_fn(theta, k):
        # Always accept
        return theta, tf.constant(1.0, tf.float64)

    k_out = lu_tuning.tune_k(
        theta0=theta0,
        step_fn=step_fn,
        k0=k0,
        pilot_length=pilot_length,
        target_low=0.3,
        target_high=0.5,
        max_rounds=max_rounds,
        factor=factor,
        name="accept",
    )

    expected = 1.0 * (factor**max_rounds)
    assert abs(float(k_out.numpy()) - expected) < 1e-12


def test_tune_k_keeps_k_unchanged_when_acceptance_in_band():
    theta0 = tf.constant(0.0, tf.float64)
    k0 = tf.constant(1.0, tf.float64)

    def step_fn(theta, k):
        # acc_inc constant in-band
        return theta, tf.constant(0.4, tf.float64)

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


@pytest.mark.parametrize("shape_case", ["scalar", "vector"])
def test_tune_k_preserves_theta_shape_scalar_and_vector(shape_case):
    k0 = tf.constant(1.0, tf.float64)

    if shape_case == "scalar":
        theta0 = tf.constant(0.0, tf.float64)

        def step_fn(theta, k):
            return theta, tf.constant(0.4, tf.float64)

        k_out = lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=3,
            target_low=0.3,
            target_high=0.5,
            max_rounds=5,
            factor=1.1,
            name="shape_scalar",
        )
        _assert_scalar_positive(k_out)

    else:
        theta0 = tf.constant([0.0, 1.0, -1.0], tf.float64)

        def step_fn(theta, k):
            return theta, tf.constant(0.4, tf.float64)

        k_out = lu_tuning.tune_k(
            theta0=theta0,
            step_fn=step_fn,
            k0=k0,
            pilot_length=3,
            target_low=0.3,
            target_high=0.5,
            max_rounds=5,
            factor=1.1,
            name="shape_vector",
        )
        _assert_scalar_positive(k_out)


# -----------------------------------------------------------------------------
# tune_shrinkage integration / wiring tests
# -----------------------------------------------------------------------------
def test_tune_shrinkage_validate_input_rejects_missing_or_wrong_types(
    dummy_shrinkage_estimator,
):
    # Cache arrays BEFORE any mutation; do not mutate the fixture instance.
    pjt_np = dummy_shrinkage_estimator.pjt.numpy()
    wjt_np = dummy_shrinkage_estimator.wjt.numpy()
    qjt_np = dummy_shrinkage_estimator.qjt.numpy()
    q0t_np = dummy_shrinkage_estimator.q0t.numpy()

    # Missing attribute
    shrink = _make_shrink(pjt_np, wjt_np, qjt_np, q0t_np)
    delattr(shrink, "qjt")
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)

    # Wrong type for state variable (must be tf.Variable)
    shrink = _make_shrink(pjt_np, wjt_np, qjt_np, q0t_np)
    shrink.beta_p = shrink.beta_p.read_value()  # type: ignore[assignment]
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)

    # Invalid factor_rw
    shrink = _make_shrink(pjt_np, wjt_np, qjt_np, q0t_np, factor_rw=1.0)
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)


def test_tune_shrinkage_returns_four_positive_finite_scalars(
    monkeypatch, dummy_shrinkage_estimator
):
    # Make tuning deterministic and cheap: stub update functions to return theta unchanged.
    # Set band wide so k is unchanged (action=ok) and tuner terminates in round 0.
    shrink = dummy_shrinkage_estimator
    shrink.target_low = 0.0
    shrink.target_high = 1.0
    shrink.max_rounds = 1
    shrink.pilot_length = 1

    def stub_update_r(**kwargs):
        return kwargs["r"], tf.constant(True)

    def stub_update_beta(**kwargs):
        return kwargs["beta_p"], kwargs["beta_w"], tf.constant(True)

    def stub_update_E_bar(**kwargs):
        E_bar = kwargs["E_bar"]
        accepted = tf.ones(tf.shape(E_bar), dtype=tf.bool)
        return E_bar, accepted

    def stub_update_njt(**kwargs):
        njt = kwargs["njt"]
        acc_sum = tf.cast(tf.shape(njt)[0], tf.float64)
        return njt, acc_sum

    monkeypatch.setattr(lu_tuning, "update_r", stub_update_r)
    monkeypatch.setattr(lu_tuning, "update_beta", stub_update_beta)
    monkeypatch.setattr(lu_tuning, "update_E_bar", stub_update_E_bar)
    monkeypatch.setattr(lu_tuning, "update_njt", stub_update_njt)

    k_r, k_E_bar, k_beta, k_njt = lu_tuning.tune_shrinkage(shrink)

    for k in [k_r, k_E_bar, k_beta, k_njt]:
        _assert_scalar_positive(k)


def test_tune_shrinkage_does_not_mutate_sampler_state(
    monkeypatch, dummy_shrinkage_estimator
):
    shrink = dummy_shrinkage_estimator
    shrink.target_low = 0.0
    shrink.target_high = 1.0
    shrink.max_rounds = 1
    shrink.pilot_length = 1

    def stub_update_r(**kwargs):
        return kwargs["r"], tf.constant(False)

    def stub_update_beta(**kwargs):
        return kwargs["beta_p"], kwargs["beta_w"], tf.constant(False)

    def stub_update_E_bar(**kwargs):
        E_bar = kwargs["E_bar"]
        accepted = tf.zeros(tf.shape(E_bar), dtype=tf.bool)
        return E_bar, accepted

    def stub_update_njt(**kwargs):
        njt = kwargs["njt"]
        return njt, tf.constant(0.0, tf.float64)

    monkeypatch.setattr(lu_tuning, "update_r", stub_update_r)
    monkeypatch.setattr(lu_tuning, "update_beta", stub_update_beta)
    monkeypatch.setattr(lu_tuning, "update_E_bar", stub_update_E_bar)
    monkeypatch.setattr(lu_tuning, "update_njt", stub_update_njt)

    before = _snapshot_state(shrink)
    _ = lu_tuning.tune_shrinkage(shrink)
    after = _snapshot_state(shrink)

    _assert_state_unchanged(before, after)


def test_tune_shrinkage_uses_correct_factor_for_rw_vs_tmh(
    monkeypatch, dummy_shrinkage_estimator
):
    # Make acceptance deterministically out-of-band so k must update once.
    shrink = dummy_shrinkage_estimator
    shrink.pilot_length = 1
    shrink.max_rounds = 1
    shrink.target_low = 0.3
    shrink.target_high = 0.5
    shrink.factor_rw = 1.1
    shrink.factor_tmh = 1.5

    # Always accept => acc_rate = 1.0 => "grow" => k *= factor
    def stub_update_r(**kwargs):
        return kwargs["r"], tf.constant(True)

    def stub_update_beta(**kwargs):
        return kwargs["beta_p"], kwargs["beta_w"], tf.constant(True)

    def stub_update_E_bar(**kwargs):
        E_bar = kwargs["E_bar"]
        accepted = tf.ones(tf.shape(E_bar), dtype=tf.bool)
        return E_bar, accepted

    def stub_update_njt(**kwargs):
        njt = kwargs["njt"]
        acc_sum = tf.cast(tf.shape(njt)[0], tf.float64)
        return njt, acc_sum

    monkeypatch.setattr(lu_tuning, "update_r", stub_update_r)
    monkeypatch.setattr(lu_tuning, "update_beta", stub_update_beta)
    monkeypatch.setattr(lu_tuning, "update_E_bar", stub_update_E_bar)
    monkeypatch.setattr(lu_tuning, "update_njt", stub_update_njt)

    # Expected initial ks from lu_tuning's internal rule.
    k_r0 = float(lu_tuning._lu_k0(tf.constant(1.0, tf.float64)).numpy())
    k_beta0 = float(lu_tuning._lu_k0(tf.constant(2.0, tf.float64)).numpy())
    k_E_bar0 = float(lu_tuning._lu_k0(tf.constant(1.0, tf.float64)).numpy())
    k_njt0 = float(lu_tuning._lu_k0(tf.constant(int(shrink.J), tf.float64)).numpy())

    k_r, k_E_bar, k_beta, k_njt = lu_tuning.tune_shrinkage(shrink)

    assert abs(float(k_r.numpy()) - (k_r0 * shrink.factor_rw)) < 1e-12
    assert abs(float(k_E_bar.numpy()) - (k_E_bar0 * shrink.factor_rw)) < 1e-12
    assert abs(float(k_beta.numpy()) - (k_beta0 * shrink.factor_tmh)) < 1e-12
    assert abs(float(k_njt.numpy()) - (k_njt0 * shrink.factor_tmh)) < 1e-12
