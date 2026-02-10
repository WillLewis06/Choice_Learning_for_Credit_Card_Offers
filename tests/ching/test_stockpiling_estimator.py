# tests/ching/test_stockpiling_estimator.py
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import ching.stockpiling_estimator as est_mod
from ching.stockpiling_estimator import StockpilingEstimator


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def test_init_converts_inputs_and_initial_state(
    estimator_tiny: StockpilingEstimator,
    tiny_dims: dict[str, int],
) -> None:
    est = estimator_tiny
    M, N, T, S, I_max = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["T"],
        tiny_dims["S"],
        tiny_dims["I_max"],
    )
    I = I_max + 1

    assert est.M == M
    assert est.N == N

    assert tuple(est.a_imt.shape) == (M, N, T)
    assert est.a_imt.dtype == tf.int32

    assert tuple(est.p_state_mt.shape) == (M, T)
    assert est.p_state_mt.dtype == tf.int32

    assert tuple(est.u_m.shape) == (M,)
    assert est.u_m.dtype == tf.float64

    assert tuple(est.price_vals.shape) == (S,)
    assert est.price_vals.dtype == tf.float64

    assert tuple(est.P_price.shape) == (S, S)
    assert est.P_price.dtype == tf.float64

    assert tuple(est.pi_I0.shape) == (I,)
    assert est.pi_I0.dtype == tf.float64

    assert est.waste_cost.dtype == tf.float64
    assert est.eps.dtype == tf.float64
    assert est.tol.dtype == tf.float64
    assert est.max_iter.dtype == tf.int32

    # Inventory maps exist (detailed correctness is tested in posterior tests).
    maps = est.maps
    assert isinstance(maps, (tuple, list)) and len(maps) == 4
    D_down, D_up, stockout_mask, at_cap_mask = maps
    assert tuple(D_down.shape) == (I, I)
    assert tuple(D_up.shape) == (I, I)
    assert tuple(stockout_mask.shape) == (I,)
    assert tuple(at_cap_mask.shape) == (I,)

    # z blocks initialized to 0
    assert set(est.z.keys()) == {
        "z_beta",
        "z_alpha",
        "z_v",
        "z_fc",
        "z_lambda",
        "z_u_scale",
    }
    for k in ["z_beta", "z_alpha", "z_v", "z_fc", "z_lambda"]:
        assert tuple(est.z[k].shape) == (M, N)
        np.testing.assert_allclose(est.z[k].numpy(), 0.0)
    assert tuple(est.z["z_u_scale"].shape) == (M,)
    np.testing.assert_allclose(est.z["z_u_scale"].numpy(), 0.0)

    # Acceptance counters initialized to 0
    assert int(est.accept_beta.numpy()) == 0
    assert int(est.accept_alpha.numpy()) == 0
    assert int(est.accept_v.numpy()) == 0
    assert int(est.accept_fc.numpy()) == 0
    assert int(est.accept_lambda.numpy()) == 0
    assert int(est.accept_u_scale.numpy()) == 0

    # No diagnostics until fit
    assert est._diag is None


def test_get_results_before_fit_raises(estimator_tiny: StockpilingEstimator) -> None:
    with pytest.raises(ValueError, match=r"get_results\(\) called before fit\(\)\."):
        _ = estimator_tiny.get_results()


@pytest.mark.parametrize(
    "bad_case",
    [
        "pi_I0_not_normalized",
        "sigmas_missing_key",
    ],
)
def test_init_rejects_invalid_inputs(
    bad_case: str,
    panel_np: dict[str, np.ndarray],
    u_m_np: np.ndarray,
    price_process: dict[str, np.ndarray],
    pi_I0_uniform: np.ndarray,
    tiny_dims: dict[str, int],
    tiny_dp_config: dict[str, object],
    sigmas: dict[str, float],
) -> None:
    pi_I0 = pi_I0_uniform.copy()
    sigmas_local = dict(sigmas)

    if bad_case == "pi_I0_not_normalized":
        pi_I0 = 2.0 * pi_I0  # sum != 1
    elif bad_case == "sigmas_missing_key":
        sigmas_local.pop("z_u_scale")
    else:
        raise ValueError("Unknown bad_case")

    with pytest.raises(ValueError):
        _ = StockpilingEstimator(
            a_imt=panel_np["a_imt"],
            p_state_mt=panel_np["p_state_mt"],
            u_m=u_m_np,
            price_vals=price_process["price_vals"],
            P_price=price_process["P_price"],
            I_max=int(tiny_dims["I_max"]),
            pi_I0=pi_I0,
            waste_cost=float(tiny_dp_config["waste_cost"]),
            eps=float(tiny_dp_config["eps"]),
            tol=float(tiny_dp_config["tol"]),
            max_iter=int(tiny_dp_config["max_iter"]),
            sigmas=sigmas_local,
            seed=123,
        )


@pytest.mark.parametrize(
    "n_iter,k_beta",
    [
        (0, 0.1),
        (1, 0.0),
        (1, -0.1),
        (-1, 0.1),
    ],
)
def test_fit_rejects_invalid_fit_inputs(
    estimator_tiny: StockpilingEstimator,
    n_iter: int,
    k_beta: float,
) -> None:
    with pytest.raises(ValueError):
        estimator_tiny.fit(
            n_iter=n_iter,
            k_beta=k_beta,
            k_alpha=0.1,
            k_v=0.1,
            k_fc=0.1,
            k_lambda=0.1,
            k_u_scale=0.1,
        )


def test_fit_updates_z_accept_counts_and_posterior_means_with_deterministic_mh(
    estimator_tiny: StockpilingEstimator,
    tiny_dims: dict[str, int],
    proposal_scales_tiny: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    M, N = tiny_dims["M"], tiny_dims["N"]

    def fake_rw_mh_step(z_current, logp_fn, k, rng):
        # Deterministic: always add 1 and always accept.
        z_new = tf.cast(z_current, tf.float64) + tf.ones_like(
            z_current, dtype=tf.float64
        )
        accepted = tf.ones_like(z_current, dtype=tf.bool)
        return z_new, accepted

    monkeypatch.setattr(est_mod, "rw_mh_step", fake_rw_mh_step, raising=True)

    est = estimator_tiny
    est.fit(n_iter=2, **proposal_scales_tiny)

    # z increments twice
    for k in ["z_beta", "z_alpha", "z_v", "z_fc", "z_lambda"]:
        np.testing.assert_allclose(est.z[k].numpy(), 2.0)
    np.testing.assert_allclose(est.z["z_u_scale"].numpy(), 2.0)

    res = est.get_results()
    assert res["n_saved"] == 2

    # Acceptance counts (elementwise): n_iter * (#elements)
    counts = res["accept"]["counts"]
    assert counts["beta"] == 2 * M * N
    assert counts["alpha"] == 2 * M * N
    assert counts["v"] == 2 * M * N
    assert counts["fc"] == 2 * M * N
    assert counts["lambda_c"] == 2 * M * N
    assert counts["u_scale"] == 2 * M

    rates = res["accept"]["rates"]
    assert rates["beta"] == 1.0
    assert rates["alpha"] == 1.0
    assert rates["v"] == 1.0
    assert rates["fc"] == 1.0
    assert rates["lambda_c"] == 1.0
    assert rates["u_scale"] == 1.0

    # Posterior means: average of transform(z=1) and transform(z=2)
    beta_hat_expected = 0.5 * (_sigmoid(1.0) + _sigmoid(2.0))
    exp_hat_expected = 0.5 * (np.exp(1.0) + np.exp(2.0))

    theta_hat = res["theta_hat"]

    # Types: numpy arrays returned, python scalar for n_saved, python floats for rates
    assert isinstance(theta_hat["beta"], np.ndarray)
    assert isinstance(theta_hat["u_scale"], np.ndarray)
    assert isinstance(res["n_saved"], int)
    assert isinstance(rates["beta"], float)

    np.testing.assert_allclose(
        theta_hat["beta"], beta_hat_expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        theta_hat["lambda_c"], beta_hat_expected, rtol=0.0, atol=1e-12
    )

    np.testing.assert_allclose(
        theta_hat["alpha"], exp_hat_expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(theta_hat["v"], exp_hat_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(theta_hat["fc"], exp_hat_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        theta_hat["u_scale"], exp_hat_expected, rtol=0.0, atol=1e-12
    )


def test_fit_resets_acceptance_counters_each_call(
    estimator_tiny: StockpilingEstimator,
    tiny_dims: dict[str, int],
    proposal_scales_tiny: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    M, N = tiny_dims["M"], tiny_dims["N"]

    def fake_rw_mh_step(z_current, logp_fn, k, rng):
        z_new = tf.cast(z_current, tf.float64) + tf.ones_like(
            z_current, dtype=tf.float64
        )
        accepted = tf.ones_like(z_current, dtype=tf.bool)
        return z_new, accepted

    monkeypatch.setattr(est_mod, "rw_mh_step", fake_rw_mh_step, raising=True)

    est = estimator_tiny

    est.fit(n_iter=1, **proposal_scales_tiny)
    res1 = est.get_results()
    assert res1["n_saved"] == 1
    assert res1["accept"]["counts"]["beta"] == 1 * M * N
    assert res1["accept"]["counts"]["u_scale"] == 1 * M

    est.fit(n_iter=1, **proposal_scales_tiny)
    res2 = est.get_results()
    assert res2["n_saved"] == 1
    assert res2["accept"]["counts"]["beta"] == 1 * M * N
    assert res2["accept"]["counts"]["u_scale"] == 1 * M

    # Since z is not reset, second fit's posterior mean corresponds to z==2 only.
    beta_expected = _sigmoid(2.0)
    exp_expected = np.exp(2.0)
    np.testing.assert_allclose(
        res2["theta_hat"]["beta"], beta_expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        res2["theta_hat"]["alpha"], exp_expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        res2["theta_hat"]["u_scale"], exp_expected, rtol=0.0, atol=1e-12
    )


def test_logp_closures_return_shape_matching_each_block(
    estimator_tiny: StockpilingEstimator,
    proposal_scales_tiny: dict[str, float],
    tiny_dims: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Patch posterior view functions to cheap shape-correct stubs, and patch rw_mh_step to
    assert logp_fn(z_candidate) has the same shape as z_candidate (for each block).
    """
    M, N = tiny_dims["M"], tiny_dims["N"]

    def fake_logpost_z_beta_mn(*, z, **kwargs):
        return tf.zeros_like(z["z_beta"])

    def fake_logpost_z_alpha_mn(*, z, **kwargs):
        return tf.zeros_like(z["z_alpha"])

    def fake_logpost_z_v_mn(*, z, **kwargs):
        return tf.zeros_like(z["z_v"])

    def fake_logpost_z_fc_mn(*, z, **kwargs):
        return tf.zeros_like(z["z_fc"])

    def fake_logpost_z_lambda_mn(*, z, **kwargs):
        return tf.zeros_like(z["z_lambda"])

    def fake_logpost_u_scale_m(*, z, **kwargs):
        return tf.zeros_like(z["z_u_scale"])

    monkeypatch.setattr(
        est_mod, "logpost_z_beta_mn", fake_logpost_z_beta_mn, raising=True
    )
    monkeypatch.setattr(
        est_mod, "logpost_z_alpha_mn", fake_logpost_z_alpha_mn, raising=True
    )
    monkeypatch.setattr(est_mod, "logpost_z_v_mn", fake_logpost_z_v_mn, raising=True)
    monkeypatch.setattr(est_mod, "logpost_z_fc_mn", fake_logpost_z_fc_mn, raising=True)
    monkeypatch.setattr(
        est_mod, "logpost_z_lambda_mn", fake_logpost_z_lambda_mn, raising=True
    )
    monkeypatch.setattr(
        est_mod, "logpost_u_scale_m", fake_logpost_u_scale_m, raising=True
    )

    def shape_check_rw_mh_step(z_current, logp_fn, k, rng):
        logp = logp_fn(tf.convert_to_tensor(z_current, dtype=tf.float64))
        tf.debugging.assert_equal(tf.shape(logp), tf.shape(z_current))
        accepted = tf.zeros_like(z_current, dtype=tf.bool)
        return tf.convert_to_tensor(z_current, dtype=tf.float64), accepted

    monkeypatch.setattr(est_mod, "rw_mh_step", shape_check_rw_mh_step, raising=True)

    est = estimator_tiny
    est.fit(n_iter=1, **proposal_scales_tiny)

    # No accepts
    res = est.get_results()
    assert res["n_saved"] == 1
    assert res["accept"]["counts"]["beta"] == 0
    assert res["accept"]["counts"]["u_scale"] == 0

    # With z still at 0, posterior mean is prior-mode transform.
    np.testing.assert_allclose(res["theta_hat"]["beta"], 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res["theta_hat"]["alpha"], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res["theta_hat"]["u_scale"], 1.0, rtol=0.0, atol=0.0)

    # Shapes
    assert res["theta_hat"]["beta"].shape == (M, N)
    assert res["theta_hat"]["u_scale"].shape == (M,)


def test_init_and_fit_do_not_mutate_numpy_inputs(
    panel_np: dict[str, np.ndarray],
    u_m_np: np.ndarray,
    price_process: dict[str, np.ndarray],
    pi_I0_uniform: np.ndarray,
    tiny_dims: dict[str, int],
    tiny_dp_config: dict[str, object],
    sigmas: dict[str, float],
    proposal_scales_tiny: dict[str, float],
    silence_progress: None,
    tf_seed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a0 = panel_np["a_imt"].copy()
    s0 = panel_np["p_state_mt"].copy()
    u0 = u_m_np.copy()
    pv0 = price_process["price_vals"].copy()
    P0 = price_process["P_price"].copy()
    pi0 = pi_I0_uniform.copy()

    # Patch MH to avoid posterior cost
    def fake_rw_mh_step(z_current, logp_fn, k, rng):
        z_new = tf.cast(z_current, tf.float64)
        accepted = tf.zeros_like(z_current, dtype=tf.bool)
        return z_new, accepted

    monkeypatch.setattr(est_mod, "rw_mh_step", fake_rw_mh_step, raising=True)

    est = StockpilingEstimator(
        a_imt=panel_np["a_imt"],
        p_state_mt=panel_np["p_state_mt"],
        u_m=u_m_np,
        price_vals=price_process["price_vals"],
        P_price=price_process["P_price"],
        I_max=int(tiny_dims["I_max"]),
        pi_I0=pi_I0_uniform,
        waste_cost=float(tiny_dp_config["waste_cost"]),
        eps=float(tiny_dp_config["eps"]),
        tol=float(tiny_dp_config["tol"]),
        max_iter=int(tiny_dp_config["max_iter"]),
        sigmas=sigmas,
        seed=123,
    )
    est.fit(n_iter=1, **proposal_scales_tiny)

    np.testing.assert_array_equal(panel_np["a_imt"], a0)
    np.testing.assert_array_equal(panel_np["p_state_mt"], s0)
    np.testing.assert_array_equal(u_m_np, u0)
    np.testing.assert_array_equal(price_process["price_vals"], pv0)
    np.testing.assert_array_equal(price_process["P_price"], P0)
    np.testing.assert_array_equal(pi_I0_uniform, pi0)


@pytest.mark.slow
def test_fit_runs_with_real_posterior_one_iteration_smoke(
    estimator_tiny: StockpilingEstimator,
    proposal_scales_tiny: dict[str, float],
) -> None:
    """
    Smoke test: ensure the unpatched estimator can run one iteration end-to-end.
    Marked slow because it executes the posterior DP/filter.
    """
    estimator_tiny.fit(n_iter=1, **proposal_scales_tiny)
    res = estimator_tiny.get_results()

    assert res["n_saved"] == 1
    for k, v in res["theta_hat"].items():
        assert isinstance(v, np.ndarray)
        assert np.isfinite(v).all()
