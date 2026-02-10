# tests/test_cl_tuning.py
import numpy as np
import pytest
import tensorflow as tf

from conftest import assert_all_finite_tf
import lu.choice_learn.cl_tuning as cl_tuning
from lu.choice_learn.cl_shrinkage import ChoiceLearnShrinkageEstimator


# -----------------------------------------------------------------------------
# Helpers (test-only)
# -----------------------------------------------------------------------------
def _assert_scalar_positive(x: tf.Tensor) -> None:
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    assert x.shape == ()
    assert_all_finite_tf(x)
    assert float(x.numpy()) > 0.0


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
    assert np.allclose(before["E_bar"], after["E_bar"], atol=0.0, rtol=0.0)
    assert np.allclose(before["njt"], after["njt"], atol=0.0, rtol=0.0)
    assert np.array_equal(before["gamma"], after["gamma"])
    assert np.allclose(before["phi"], after["phi"], atol=0.0, rtol=0.0)


def _build_shrink(
    tiny_market_data: dict, *, seed: int = 123
) -> ChoiceLearnShrinkageEstimator:
    """
    Build a tiny ChoiceLearnShrinkageEstimator.

    Expects the ChoiceLearn conftest schema:
      - delta_cl: (T,J)
      - qjt: (T,J)
      - q0t: (T,)
    """
    delta_cl = np.asarray(tiny_market_data["delta_cl"], dtype=np.float64)
    qjt = np.asarray(tiny_market_data["qjt"], dtype=np.float64)
    q0t = np.asarray(tiny_market_data["q0t"], dtype=np.float64)

    return ChoiceLearnShrinkageEstimator(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
        seed=seed,
    )


def _set_tuning_params(
    shrink: ChoiceLearnShrinkageEstimator,
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
    Patch update_* functions in cl_tuning so that:
    - proposals do not change the state (identity updates)
    - acceptance is deterministic (all accept or all reject)
    """
    accept_bool = tf.constant(bool(accept_all), dtype=tf.bool)

    def stub_update_alpha(**kwargs):
        return kwargs["alpha"], accept_bool

    def stub_update_E_bar(**kwargs):
        E_bar = kwargs["E_bar"]
        accepted = tf.fill(tf.shape(E_bar), accept_bool)
        return E_bar, accepted

    def stub_update_njt(**kwargs):
        njt = kwargs["njt"]
        T = tf.cast(tf.shape(njt)[0], tf.float64)
        acc_sum = T if accept_all else tf.constant(0.0, tf.float64)
        return njt, acc_sum

    monkeypatch.setattr(cl_tuning, "update_alpha", stub_update_alpha)
    monkeypatch.setattr(cl_tuning, "update_E_bar", stub_update_E_bar)
    monkeypatch.setattr(cl_tuning, "update_njt", stub_update_njt)


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
        dict(
            pilot_length=1,
            target_low=0.3,
            target_high=0.5,
            max_rounds=1,
            factor=1.1,
            name=123,  # type: ignore[arg-type]
        ),
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
# tune_shrinkage integration / wiring tests
# -----------------------------------------------------------------------------
def test_tune_shrinkage_validate_input_rejects_missing_or_wrong_types(tiny_market_data):
    # Missing attribute
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink)
    delattr(shrink, "qjt")
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)

    # Wrong type for state variable (must be tf.Variable)
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink)
    shrink.alpha = shrink.alpha.read_value()  # type: ignore[assignment]
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)

    # Invalid factor_rw
    shrink = _build_shrink(tiny_market_data)
    _set_tuning_params(shrink, factor_rw=1.0)
    with pytest.raises(Exception):
        cl_tuning.tune_shrinkage(shrink)


def test_tune_shrinkage_returns_three_positive_finite_scalars(
    monkeypatch, shrinkage_estimator
):
    shrink = shrinkage_estimator
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    _patch_updates(monkeypatch, accept_all=True)

    k_alpha, k_E_bar, k_njt = cl_tuning.tune_shrinkage(shrink)
    for k in [k_alpha, k_E_bar, k_njt]:
        _assert_scalar_positive(k)


def test_tune_shrinkage_does_not_mutate_sampler_state(monkeypatch, shrinkage_estimator):
    shrink = shrinkage_estimator
    _set_tuning_params(
        shrink, target_low=0.0, target_high=1.0, max_rounds=1, pilot_length=1
    )

    _patch_updates(monkeypatch, accept_all=False)

    before = _snapshot_state(shrink)
    _ = cl_tuning.tune_shrinkage(shrink)
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

    monkeypatch.setattr(cl_tuning, "tune_k", stub_tune_k)

    _ = cl_tuning.tune_shrinkage(shrink)

    names = [n for (n, _) in calls]
    assert (
        len(calls) >= 3
    ), f"Expected at least 3 tune_k calls, got {len(calls)}: {names}"

    atol = 1e-6
    for name, factor in calls:
        if name in ["alpha", "E_bar"]:
            exp = float(shrink.factor_rw)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_rw={shrink.factor_rw} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )
        if name == "njt":
            exp = float(shrink.factor_tmh)
            assert abs(factor - exp) <= atol, (
                f"{name} used factor={factor}, expected factor_tmh={shrink.factor_tmh} "
                f"(abs diff={abs(factor - exp):.3e}, atol={atol:.3e})"
            )
