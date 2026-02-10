# tests/ching/test_stockpiling_estimator.py
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

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

    # Inventory maps exist (posterior tests cover detailed correctness).
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

    # No diagnostics until fit()
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
        pi_I0 = 2.0 * pi_I0
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


def test_fit_updates_z_accept_counts_and_posterior_means_with_deterministic_step(
    estimator_tiny: StockpilingEstimator,
    tiny_dims: dict[str, int],
    proposal_scales_tiny: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Deterministic, fast test of estimator accounting and get_results:
    patch the per-iteration step to avoid TF tracing / posterior cost.
    """
    est = estimator_tiny
    M, N = tiny_dims["M"], tiny_dims["N"]

    def fake_iteration_step(
        self: StockpilingEstimator, it, k_beta, k_alpha, k_v, k_fc, k_lambda, k_u_scale
    ) -> None:
        ones_mn = tf.ones_like(self.z["z_beta"], dtype=tf.float64)
        ones_m = tf.ones_like(self.z["z_u_scale"], dtype=tf.float64)

        # Increment each z block by +1
        self.z["z_beta"].assign_add(ones_mn)
        self.z["z_alpha"].assign_add(ones_mn)
        self.z["z_v"].assign_add(ones_mn)
        self.z["z_fc"].assign_add(ones_mn)
        self.z["z_lambda"].assign_add(ones_mn)
        self.z["z_u_scale"].assign_add(ones_m)

        # Accept everything (elementwise counts)
        add_mn = tf.convert_to_tensor(self.M * self.N, dtype=tf.int32)
        add_m = tf.convert_to_tensor(self.M, dtype=tf.int32)
        self.accept_beta.assign_add(add_mn)
        self.accept_alpha.assign_add(add_mn)
        self.accept_v.assign_add(add_mn)
        self.accept_fc.assign_add(add_mn)
        self.accept_lambda.assign_add(add_mn)
        self.accept_u_scale.assign_add(add_m)

        # Accumulate posterior means (no printing)
        assert self._diag is not None
        self._diag._accumulate_draw(self)

    monkeypatch.setattr(
        est,
        "_mcmc_iteration_step",
        fake_iteration_step.__get__(est, StockpilingEstimator),
        raising=True,
    )

    est.fit(n_iter=2, **proposal_scales_tiny)

    # z increments twice
    for k in ["z_beta", "z_alpha", "z_v", "z_fc", "z_lambda"]:
        np.testing.assert_allclose(est.z[k].numpy(), 2.0)
    np.testing.assert_allclose(est.z["z_u_scale"].numpy(), 2.0)

    res = est.get_results()
    assert res["n_saved"] == 2

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
    est = estimator_tiny
    M, N = tiny_dims["M"], tiny_dims["N"]

    def fake_iteration_step(
        self: StockpilingEstimator, it, k_beta, k_alpha, k_v, k_fc, k_lambda, k_u_scale
    ) -> None:
        ones_mn = tf.ones_like(self.z["z_beta"], dtype=tf.float64)
        ones_m = tf.ones_like(self.z["z_u_scale"], dtype=tf.float64)

        self.z["z_beta"].assign_add(ones_mn)
        self.z["z_alpha"].assign_add(ones_mn)
        self.z["z_v"].assign_add(ones_mn)
        self.z["z_fc"].assign_add(ones_mn)
        self.z["z_lambda"].assign_add(ones_mn)
        self.z["z_u_scale"].assign_add(ones_m)

        add_mn = tf.convert_to_tensor(self.M * self.N, dtype=tf.int32)
        add_m = tf.convert_to_tensor(self.M, dtype=tf.int32)
        self.accept_beta.assign_add(add_mn)
        self.accept_alpha.assign_add(add_mn)
        self.accept_v.assign_add(add_mn)
        self.accept_fc.assign_add(add_mn)
        self.accept_lambda.assign_add(add_mn)
        self.accept_u_scale.assign_add(add_m)

        assert self._diag is not None
        self._diag._accumulate_draw(self)

    monkeypatch.setattr(
        est,
        "_mcmc_iteration_step",
        fake_iteration_step.__get__(est, StockpilingEstimator),
        raising=True,
    )

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

    # z is not reset; first fit makes z==1, second fit makes z==2 for the saved draw
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
    test_seed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a0 = panel_np["a_imt"].copy()
    s0 = panel_np["p_state_mt"].copy()
    u0 = u_m_np.copy()
    pv0 = price_process["price_vals"].copy()
    P0 = price_process["P_price"].copy()
    pi0 = pi_I0_uniform.copy()

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
        seed=int(test_seed),
    )

    # Patch the iteration step to avoid posterior cost.
    def fake_iteration_step(
        self: StockpilingEstimator, it, k_beta, k_alpha, k_v, k_fc, k_lambda, k_u_scale
    ) -> None:
        assert self._diag is not None
        self._diag._accumulate_draw(self)

    monkeypatch.setattr(
        est,
        "_mcmc_iteration_step",
        fake_iteration_step.__get__(est, StockpilingEstimator),
        raising=True,
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
    Marked slow because it executes posterior DP/filter inside tf.function.
    """
    estimator_tiny.fit(n_iter=1, **proposal_scales_tiny)
    res = estimator_tiny.get_results()

    assert res["n_saved"] == 1
    for v in res["theta_hat"].values():
        assert isinstance(v, np.ndarray)
        assert np.isfinite(v).all()
