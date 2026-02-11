# tests/ching/test_stockpiling_estimator.py
"""
Unit tests for the Phase-3 (Ching-style) stockpiling estimator.

Design goals for these tests:
  - No pytest fixtures: tests build their own tiny inputs.
  - Fast + deterministic: most tests patch the per-iteration MCMC step.
  - Readable: focus on public API contract (init/fit/get_results) and bookkeeping.

The expensive dynamic-programming posterior is exercised only in a small smoke test.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
import tensorflow as tf

import ching_conftest as ct
from ching.stockpiling_estimator import StockpilingEstimator


def _sigmoid(x: float) -> float:
    """Scalar sigmoid used to form deterministic expectations."""
    return 1.0 / (1.0 + np.exp(-x))


def _proposal_scales_tiny() -> dict[str, float]:
    """
    Proposal step sizes for fit(...).

    Kept local to avoid depending on any test-helper implementation details.
    """
    return {
        "k_beta": 0.1,
        "k_alpha": 0.1,
        "k_v": 0.1,
        "k_fc": 0.1,
        "k_lambda": 0.1,
        "k_u_scale": 0.1,
    }


def _tiny_inputs() -> tuple[
    dict[str, int],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, Any],
    dict[str, float],
]:
    """
    Build a deterministic tiny test environment.

    Returns:
      dims:     {"M","N","T","S","I_max"}
      panel:    {"a_imt","p_state_mt"} as numpy arrays
      u_m:      (M,) numpy float64
      prices:   {"price_vals","P_price"} as numpy arrays
      pi_i0:    (I_max+1,) numpy float64
      cfg:      {"waste_cost","eps","tol","max_iter"}
      sigmas:   prior scales dict for z_* blocks
    """
    dims = ct.tiny_dims()
    panel = ct.panel_np(dims)
    u_m = ct.u_m_np(dims)
    prices = ct.price_process(dims)
    pi_i0 = ct.pi_I0_uniform(dims)
    cfg = ct.tiny_dp_config()
    sigmas = ct.sigmas()
    return dims, panel, u_m, prices, pi_i0, cfg, sigmas


def _make_estimator(seed: int = 0) -> StockpilingEstimator:
    """Construct a StockpilingEstimator from the tiny environment."""
    dims, panel, u_m, prices, pi_i0, cfg, sigmas = _tiny_inputs()
    i_max = int(dims["I_max"])
    return StockpilingEstimator(
        a_imt=panel["a_imt"],
        p_state_mt=panel["p_state_mt"],
        u_m=u_m,
        price_vals=prices["price_vals"],
        P_price=prices["P_price"],
        I_max=i_max,
        pi_I0=pi_i0,
        waste_cost=float(cfg["waste_cost"]),
        eps=float(cfg["eps"]),
        tol=float(cfg["tol"]),
        max_iter=int(cfg["max_iter"]),
        sigmas=dict(sigmas),
        seed=int(seed),
    )


def _patch_iteration_step(
    est: StockpilingEstimator, step_fn: Callable[..., None]
) -> Callable[[], None]:
    """
    Patch est._mcmc_iteration_step in-place and return an undo() closure.

    This avoids relying on pytest's monkeypatch fixture.
    """
    original = est._mcmc_iteration_step
    est._mcmc_iteration_step = step_fn.__get__(est, StockpilingEstimator)  # type: ignore[assignment]

    def undo() -> None:
        est._mcmc_iteration_step = original  # type: ignore[assignment]

    return undo


def test_init_converts_inputs_and_initial_state() -> None:
    """
    Estimator should:
      - convert numpy inputs to TF tensors/variables with expected shapes/dtypes
      - allocate inventory maps, z blocks, accept counters, and running sums
    """
    est = _make_estimator(seed=1)
    dims, _, _, _, _, _, _ = _tiny_inputs()

    m = int(dims["M"])
    n = int(dims["N"])
    t = int(dims["T"])
    s = int(dims["S"])
    i_max = int(dims["I_max"])
    i = i_max + 1

    assert est.M == m
    assert est.N == n

    assert tuple(est.a_imt.shape) == (m, n, t)
    assert est.a_imt.dtype == tf.int32

    assert tuple(est.p_state_mt.shape) == (m, t)
    assert est.p_state_mt.dtype == tf.int32

    assert tuple(est.u_m.shape) == (m,)
    assert est.u_m.dtype == tf.float64

    assert tuple(est.price_vals.shape) == (s,)
    assert est.price_vals.dtype == tf.float64

    assert tuple(est.P_price.shape) == (s, s)
    assert est.P_price.dtype == tf.float64

    assert tuple(est.pi_I0.shape) == (i,)
    assert est.pi_I0.dtype == tf.float64

    # Inventory maps exist (posterior tests cover detailed correctness).
    maps = est.maps
    assert isinstance(maps, (tuple, list)) and len(maps) == 4
    idx_down, idx_up, stockout_mask, at_cap_mask = maps
    assert tuple(idx_down.shape) == (i,)
    assert tuple(idx_up.shape) == (i,)
    assert tuple(stockout_mask.shape) == (i,)
    assert tuple(at_cap_mask.shape) == (i,)

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
        assert tuple(est.z[k].shape) == (m, n)
        np.testing.assert_allclose(est.z[k].numpy(), 0.0)
    assert tuple(est.z["z_u_scale"].shape) == (m,)
    np.testing.assert_allclose(est.z["z_u_scale"].numpy(), 0.0)

    # Acceptance counters initialized to 0
    assert set(est.accept.keys()) == {"beta", "alpha", "v", "fc", "lambda_c", "u_scale"}
    for v in est.accept.values():
        assert int(v.numpy()) == 0

    # Running sums and saved counter initialized to 0
    assert int(est.saved.numpy()) == 0
    for v in est.sums.values():
        np.testing.assert_allclose(v.numpy(), 0.0)


def test_get_results_before_fit_raises() -> None:
    """get_results() requires at least one saved draw."""
    est = _make_estimator(seed=2)
    with pytest.raises(RuntimeError, match=r"n_saved == 0"):
        _ = est.get_results()


@pytest.mark.parametrize("bad_case", ["pi_i0_not_normalized", "sigmas_missing_key"])
def test_init_rejects_invalid_inputs(bad_case: str) -> None:
    """Init validation should reject obvious structural errors."""
    dims, panel, u_m, prices, pi_i0, cfg, sigmas = _tiny_inputs()
    i_max = int(dims["I_max"])

    pi_local = pi_i0.copy()
    sigmas_local = dict(sigmas)

    if bad_case == "pi_i0_not_normalized":
        pi_local = 2.0 * pi_local
    elif bad_case == "sigmas_missing_key":
        sigmas_local.pop("z_u_scale", None)
    else:
        raise ValueError("unknown bad_case")

    with pytest.raises(ValueError):
        _ = StockpilingEstimator(
            a_imt=panel["a_imt"],
            p_state_mt=panel["p_state_mt"],
            u_m=u_m,
            price_vals=prices["price_vals"],
            P_price=prices["P_price"],
            I_max=i_max,
            pi_I0=pi_local,
            waste_cost=float(cfg["waste_cost"]),
            eps=float(cfg["eps"]),
            tol=float(cfg["tol"]),
            max_iter=int(cfg["max_iter"]),
            sigmas=sigmas_local,
            seed=0,
        )


@pytest.mark.parametrize(
    "n_iter,k_beta",
    [
        (0, 0.1),  # n_iter must be >= 1
        (1, 0.0),  # proposal scales must be > 0
        (1, -0.1),
    ],
)
def test_fit_rejects_invalid_fit_inputs(n_iter: int, k_beta: float) -> None:
    """fit(...) validation should reject non-positive iteration counts / step sizes."""
    est = _make_estimator(seed=3)
    with pytest.raises(ValueError):
        est.fit(
            n_iter=n_iter,
            k_beta=k_beta,
            k_alpha=0.1,
            k_v=0.1,
            k_fc=0.1,
            k_lambda=0.1,
            k_u_scale=0.1,
        )


def test_fit_updates_z_accept_counts_and_posterior_means_with_deterministic_step() -> (
    None
):
    """
    Deterministic accounting test.

    Patch the per-iteration step so that:
      - each z block increments by +1 every iteration
      - all proposals are accepted
      - posterior sums are updated exactly like the real code
    """
    est = _make_estimator(seed=4)
    dims, _, _, _, _, _, _ = _tiny_inputs()
    m = int(dims["M"])
    n = int(dims["N"])

    def fake_iteration_step(
        self: StockpilingEstimator,
        it: tf.Tensor,
        k_beta: tf.Tensor,
        k_alpha: tf.Tensor,
        k_v: tf.Tensor,
        k_fc: tf.Tensor,
        k_lambda: tf.Tensor,
        k_u_scale: tf.Tensor,
    ) -> None:
        ones_mn = tf.ones_like(self.z["z_beta"], dtype=tf.float64)
        ones_m = tf.ones_like(self.z["z_u_scale"], dtype=tf.float64)

        # Increment each latent block.
        self.z["z_beta"].assign_add(ones_mn)
        self.z["z_alpha"].assign_add(ones_mn)
        self.z["z_v"].assign_add(ones_mn)
        self.z["z_fc"].assign_add(ones_mn)
        self.z["z_lambda"].assign_add(ones_mn)
        self.z["z_u_scale"].assign_add(ones_m)

        # Accept everything (counts accepted entries).
        add_mn = tf.convert_to_tensor(self.M * self.N, dtype=tf.int32)
        add_m = tf.convert_to_tensor(self.M, dtype=tf.int32)
        self.accept["beta"].assign_add(add_mn)
        self.accept["alpha"].assign_add(add_mn)
        self.accept["v"].assign_add(add_mn)
        self.accept["fc"].assign_add(add_mn)
        self.accept["lambda_c"].assign_add(add_mn)
        self.accept["u_scale"].assign_add(add_m)

        # Accumulate posterior means (same transforms as the real step).
        self.saved.assign_add(1)
        beta = tf.math.sigmoid(self.z["z_beta"])
        alpha = tf.exp(self.z["z_alpha"])
        v = tf.exp(self.z["z_v"])
        fc = tf.exp(self.z["z_fc"])
        lambda_c = tf.math.sigmoid(self.z["z_lambda"])
        u_scale = tf.exp(self.z["z_u_scale"])

        self.sums["beta"].assign_add(beta)
        self.sums["alpha"].assign_add(alpha)
        self.sums["v"].assign_add(v)
        self.sums["fc"].assign_add(fc)
        self.sums["lambda_c"].assign_add(lambda_c)
        self.sums["u_scale"].assign_add(u_scale)

    undo = _patch_iteration_step(est, fake_iteration_step)
    try:
        est.fit(n_iter=2, **_proposal_scales_tiny())
        res = est.get_results()
    finally:
        undo()

    # z increments twice: z==2
    for k in ["z_beta", "z_alpha", "z_v", "z_fc", "z_lambda"]:
        np.testing.assert_allclose(est.z[k].numpy(), 2.0)
    np.testing.assert_allclose(est.z["z_u_scale"].numpy(), 2.0)

    assert res["n_saved"] == 2

    counts = res["accept"]["counts"]
    assert counts["beta"] == 2 * m * n
    assert counts["alpha"] == 2 * m * n
    assert counts["v"] == 2 * m * n
    assert counts["fc"] == 2 * m * n
    assert counts["lambda_c"] == 2 * m * n
    assert counts["u_scale"] == 2 * m

    rates = res["accept"]["rates"]
    assert rates["beta"] == 1.0
    assert rates["alpha"] == 1.0
    assert rates["v"] == 1.0
    assert rates["fc"] == 1.0
    assert rates["lambda_c"] == 1.0
    assert rates["u_scale"] == 1.0

    # Posterior means: average of transform(z=1) and transform(z=2).
    beta_expected = 0.5 * (_sigmoid(1.0) + _sigmoid(2.0))
    exp_expected = 0.5 * (np.exp(1.0) + np.exp(2.0))

    theta_hat = res["theta_hat"]
    np.testing.assert_allclose(theta_hat["beta"], beta_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        theta_hat["lambda_c"], beta_expected, rtol=0.0, atol=1e-12
    )

    np.testing.assert_allclose(theta_hat["alpha"], exp_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(theta_hat["v"], exp_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(theta_hat["fc"], exp_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(theta_hat["u_scale"], exp_expected, rtol=0.0, atol=1e-12)


def test_fit_resets_acceptance_counters_and_sums_each_call() -> None:
    """
    fit(...) resets accept/sums/saved each time, but does not reset z.

    This matters because an orchestration layer might call fit multiple times
    (e.g., for tuning step sizes) and expects per-run acceptance rates.
    """
    est = _make_estimator(seed=5)
    dims, _, _, _, _, _, _ = _tiny_inputs()
    m = int(dims["M"])
    n = int(dims["N"])

    def fake_iteration_step(
        self: StockpilingEstimator,
        it: tf.Tensor,
        k_beta: tf.Tensor,
        k_alpha: tf.Tensor,
        k_v: tf.Tensor,
        k_fc: tf.Tensor,
        k_lambda: tf.Tensor,
        k_u_scale: tf.Tensor,
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
        self.accept["beta"].assign_add(add_mn)
        self.accept["alpha"].assign_add(add_mn)
        self.accept["v"].assign_add(add_mn)
        self.accept["fc"].assign_add(add_mn)
        self.accept["lambda_c"].assign_add(add_mn)
        self.accept["u_scale"].assign_add(add_m)

        self.saved.assign_add(1)
        self.sums["beta"].assign_add(tf.math.sigmoid(self.z["z_beta"]))
        self.sums["alpha"].assign_add(tf.exp(self.z["z_alpha"]))
        self.sums["v"].assign_add(tf.exp(self.z["z_v"]))
        self.sums["fc"].assign_add(tf.exp(self.z["z_fc"]))
        self.sums["lambda_c"].assign_add(tf.math.sigmoid(self.z["z_lambda"]))
        self.sums["u_scale"].assign_add(tf.exp(self.z["z_u_scale"]))

    undo = _patch_iteration_step(est, fake_iteration_step)
    try:
        est.fit(n_iter=1, **_proposal_scales_tiny())
        res1 = est.get_results()

        est.fit(n_iter=1, **_proposal_scales_tiny())
        res2 = est.get_results()
    finally:
        undo()

    assert res1["n_saved"] == 1
    assert res1["accept"]["counts"]["beta"] == 1 * m * n
    assert res1["accept"]["counts"]["u_scale"] == 1 * m

    assert res2["n_saved"] == 1
    assert res2["accept"]["counts"]["beta"] == 1 * m * n
    assert res2["accept"]["counts"]["u_scale"] == 1 * m

    # z is not reset; the second run's only saved draw corresponds to z==2.
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


def test_init_and_fit_do_not_mutate_numpy_inputs() -> None:
    """
    The estimator should not modify user-provided numpy inputs in-place.

    This test also patches the iteration step to avoid running the full posterior.
    """
    dims, panel, u_m, prices, pi_i0, cfg, sigmas = _tiny_inputs()

    a0 = panel["a_imt"].copy()
    s0 = panel["p_state_mt"].copy()
    u0 = u_m.copy()
    pv0 = prices["price_vals"].copy()
    p0 = prices["P_price"].copy()
    pi0 = pi_i0.copy()

    est = StockpilingEstimator(
        a_imt=panel["a_imt"],
        p_state_mt=panel["p_state_mt"],
        u_m=u_m,
        price_vals=prices["price_vals"],
        P_price=prices["P_price"],
        I_max=int(dims["I_max"]),
        pi_I0=pi_i0,
        waste_cost=float(cfg["waste_cost"]),
        eps=float(cfg["eps"]),
        tol=float(cfg["tol"]),
        max_iter=int(cfg["max_iter"]),
        sigmas=dict(sigmas),
        seed=123,
    )

    def fake_iteration_step(
        self: StockpilingEstimator,
        it: tf.Tensor,
        k_beta: tf.Tensor,
        k_alpha: tf.Tensor,
        k_v: tf.Tensor,
        k_fc: tf.Tensor,
        k_lambda: tf.Tensor,
        k_u_scale: tf.Tensor,
    ) -> None:
        # Minimal update: just create one saved draw so get_results is defined.
        self.saved.assign_add(1)
        self.sums["beta"].assign_add(tf.zeros_like(self.sums["beta"]))
        self.sums["alpha"].assign_add(tf.zeros_like(self.sums["alpha"]))
        self.sums["v"].assign_add(tf.zeros_like(self.sums["v"]))
        self.sums["fc"].assign_add(tf.zeros_like(self.sums["fc"]))
        self.sums["lambda_c"].assign_add(tf.zeros_like(self.sums["lambda_c"]))
        self.sums["u_scale"].assign_add(tf.zeros_like(self.sums["u_scale"]))

    undo = _patch_iteration_step(est, fake_iteration_step)
    try:
        est.fit(n_iter=1, **_proposal_scales_tiny())
        _ = est.get_results()
    finally:
        undo()

    np.testing.assert_array_equal(panel["a_imt"], a0)
    np.testing.assert_array_equal(panel["p_state_mt"], s0)
    np.testing.assert_array_equal(u_m, u0)
    np.testing.assert_array_equal(prices["price_vals"], pv0)
    np.testing.assert_array_equal(prices["P_price"], p0)
    np.testing.assert_array_equal(pi_i0, pi0)


def test_fit_runs_with_real_posterior_one_iteration_smoke() -> None:
    """
    Smoke test: run one real iteration (unpatched) to ensure end-to-end wiring works.

    This will execute value iteration and filtering inside tf.function, so keep dims tiny.
    """
    est = _make_estimator(seed=6)
    est.fit(n_iter=1, **_proposal_scales_tiny())
    res = est.get_results()

    assert res["n_saved"] == 1
    for v in res["theta_hat"].values():
        assert isinstance(v, np.ndarray)
        assert np.isfinite(v).all()
