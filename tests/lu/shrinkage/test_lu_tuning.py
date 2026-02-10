# tests/test_lu_tuning.py
import numpy as np
import pytest
import tensorflow as tf

from conftest import assert_all_finite_tf
import lu.shrinkage.lu_tuning as lu_tuning
from lu.shrinkage.lu_shrinkage import LuShrinkageEstimator


# -----------------------------------------------------------------------------
# Helpers (test-only)
# -----------------------------------------------------------------------------
def _assert_scalar_positive(x: tf.Tensor) -> None:
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    assert x.shape == ()
    assert_all_finite_tf(x)
    assert float(x.numpy()) > 0.0


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

    assert np.allclose(before["E_bar"], after["E_bar"], atol=0.0, rtol=0.0)
    assert np.allclose(before["njt"], after["njt"], atol=0.0, rtol=0.0)
    assert np.array_equal(before["gamma"], after["gamma"])
    assert np.allclose(before["phi"], after["phi"], atol=0.0, rtol=0.0)


def _build_shrink(
    tiny_market_data: dict, *, n_draws: int = 25, seed: int = 123
) -> LuShrinkageEstimator:
    return LuShrinkageEstimator(
        pjt=tiny_market_data["pjt"],
        wjt=tiny_market_data["wjt"],
        qjt=tiny_market_data["qjt"],
        q0t=tiny_market_data["q0t"],
        n_draws=n_draws,
        seed=seed,
    )


def _set_tuning_params(
    shrink: LuShrinkageEstimator,
    *,
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


def _patch_updates(monkeypatch, *, accept_all: bool) -> None:
    """
    Patch update_* functions in lu_tuning so that:
    - proposals do not change the state (identity updates)
    - acceptance is deterministic (all accept or all reject)
    """
    accept_bool = tf.constant(bool(accept_all), dtype=tf.bool)

    def stub_update_r(**kwargs):
        return kwargs["r"], accept_bool

    def stub_update_beta(**kwargs):
        return kwargs["beta_p"], kwargs["beta_w"], accept_bool

    def stub_update_E_bar(**kwargs):
        E_bar = kwargs["E_bar"]
        accepted = tf.fill(tf.shape(E_bar), accept_bool)
        return E_bar, accepted

    def stub_update_njt(**kwargs):
        njt = kwargs["njt"]
        T = tf.cast(tf.shape(njt)[0], tf.float64)
        acc_sum = T if accept_all else tf.constant(0.0, tf.float64)
        return njt, acc_sum

    monkeypatch.setattr(lu_tuning, "update_r", stub_update_r)
    monkeypatch.setattr(lu_tuning, "update_beta", stub_update_beta)
    monkeypatch.setattr(lu_tuning, "update_E_bar", stub_update_E_bar)
    monkeypatch.setattr(lu_tuning, "update_njt", stub_update_njt)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def shrinkage_estimator(tiny_market_data):
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink)
    return shrink


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
            lu_tuning.tune_k(theta0=theta0, step_fn=step_fn, k0=k0, **kw)


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
def test_tune_shrinkage_validate_input_rejects_missing_or_wrong_types(tiny_market_data):
    # Missing attribute
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink)
    delattr(shrink, "qjt")
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)

    # Wrong type for state variable (must be tf.Variable)
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink)
    shrink.beta_p = shrink.beta_p.read_value()  # type: ignore[assignment]
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)

    # Invalid factor_rw
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink, factor_rw=1.0)
    with pytest.raises(Exception):
        lu_tuning.tune_shrinkage(shrink)


def test_tune_shrinkage_returns_four_positive_finite_scalars(
    monkeypatch, shrinkage_estimator
):
    shrink = shrinkage_estimator
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    _patch_updates(monkeypatch, accept_all=True)

    k_r, k_E_bar, k_beta, k_njt = lu_tuning.tune_shrinkage(shrink)
    for k in [k_r, k_E_bar, k_beta, k_njt]:
        _assert_scalar_positive(k)


def test_tune_shrinkage_does_not_mutate_sampler_state(monkeypatch, shrinkage_estimator):
    shrink = shrinkage_estimator
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    _patch_updates(monkeypatch, accept_all=False)

    before = _snapshot_state(shrink)
    _ = lu_tuning.tune_shrinkage(shrink)
    after = _snapshot_state(shrink)

    _assert_state_unchanged(before, after)


def test_tune_shrinkage_uses_correct_factor_for_rw_vs_tmh(
    monkeypatch, shrinkage_estimator
):
    shrink = shrinkage_estimator
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

    def stub_tune_k(**kwargs):
        name = str(kwargs.get("name", ""))
        factor = float(tf.convert_to_tensor(kwargs["factor"]).numpy())
        calls.append((name, factor))
        return tf.convert_to_tensor(kwargs["k0"], dtype=tf.float64)

    monkeypatch.setattr(lu_tuning, "tune_k", stub_tune_k)

    _ = lu_tuning.tune_shrinkage(shrink)

    names = [n for (n, _) in calls]
    assert (
        len(calls) >= 4
    ), f"Expected at least 4 tune_k calls, got {len(calls)} with names={names}"

    rw_names = [
        n
        for (n, f) in calls
        if ("E_bar" in n) or ("step_r" in n) or (n == "r") or n.endswith("_r")
    ]
    tmh_names = [n for (n, f) in calls if ("beta" in n) or ("njt" in n)]

    assert len(rw_names) >= 2, f"Could not identify both RW-tuned names from {names}"
    assert len(tmh_names) >= 2, f"Could not identify both TMH-tuned names from {names}"

    atol = 1e-6
    for name, factor in calls:
        if (
            ("E_bar" in name)
            or ("step_r" in name)
            or (name == "r")
            or name.endswith("_r")
        ):
            exp = float(shrink.factor_rw)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_rw={shrink.factor_rw} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )

        if ("beta" in name) or ("njt" in name):
            exp = float(shrink.factor_tmh)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_tmh={shrink.factor_tmh} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )
