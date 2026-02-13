# tests/ching/test_stockpiling_estimator.py
"""
Unit tests for `ching.stockpiling_estimator` (multi-product Phase-3).

These tests cover ONLY estimator-layer responsibilities:
- fit()/get_results() bookkeeping (saved draws, posterior-mean accumulation)
- acceptance-rate denominators by block sizes:
    (M,J) for beta/alpha/v/fc
    (M,N) for lambda
    (M,)  for u_scale
- step-size assignment into estimator.k
- input-guard behavior (n_iter > 0, get_results requires saved draws)

The core DP / CCP / filtering logic is tested elsewhere (model/posterior tests).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

# Ensure local helper module is importable when tests/ching is not on PYTHONPATH.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reduce TensorFlow C++ logging (must be set before importing TF).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import ching_conftest as cc  # noqa: E402
import tensorflow as tf  # noqa: E402

from ching.stockpiling_estimator import StockpilingEstimator  # noqa: E402


_ATOL = 1e-12


def _make_estimator(seed: int = 123) -> StockpilingEstimator:
    """
    Build a small estimator instance using deterministic test builders.

    Expects ching_conftest helpers to provide multi-product shapes:
      a_mnjt:      (M,N,J,T)
      p_state_mjt: (M,J,T)
      u_mj:        (M,J)
      price_vals_mj: (M,J,S)
      P_price_mj:    (M,J,S,S)
      pi_I0:         (I_max+1,)
      sigmas: keys {z_beta,z_alpha,z_v,z_fc,z_lambda,z_u_scale}
    """
    dims = cc.tiny_dims()
    dp = cc.tiny_dp_config()

    panel = cc.panel_np(dims)
    u_mj = cc.u_mj_np(dims)
    price = cc.price_process(dims)
    pi_I0 = cc.pi_I0_uniform(dims)
    sigmas = cc.sigmas()

    return StockpilingEstimator(
        a_mnjt=panel["a_mnjt"],
        p_state_mjt=panel["p_state_mjt"],
        u_mj=u_mj,
        price_vals_mj=price["price_vals_mj"],
        P_price_mj=price["P_price_mj"],
        I_max=int(dims["I_max"]),
        pi_I0=pi_I0,
        waste_cost=float(dp["waste_cost"]),
        eps=float(dp["eps"]),
        tol=float(dp["tol"]),
        max_iter=int(dp["max_iter"]),
        sigmas=sigmas,
        seed=int(seed),
    )


def _patch_iteration_step(est: StockpilingEstimator, fn) -> None:
    """
    Replace the compiled `_mcmc_iteration_step` with a Python implementation.

    This avoids exercising the expensive DP+filter likelihood in unit tests that
    validate estimator bookkeeping only.
    """
    est._mcmc_iteration_step = types.MethodType(fn, est)


def test_fit_rejects_nonpositive_n_iter() -> None:
    est = _make_estimator()
    k = {
        "beta": 0.1,
        "alpha": 0.1,
        "v": 0.1,
        "fc": 0.1,
        "lambda": 0.1,
        "u_scale": 0.1,
    }

    try:
        est.fit(n_iter=0, k=k)
    except ValueError as e:
        assert "n_iter must be > 0" in str(e)
    else:
        raise AssertionError("Expected ValueError for n_iter=0")


def test_get_results_raises_when_no_saved() -> None:
    est = _make_estimator()
    try:
        est.get_results()
    except RuntimeError as e:
        assert "n_saved == 0" in str(e)
    else:
        raise AssertionError(
            "Expected RuntimeError when calling get_results() before fit()"
        )


def test_fit_and_get_results_shapes_and_accept_rates_all_accept() -> None:
    est = _make_estimator()

    def fake_step(self: StockpilingEstimator, it: tf.Tensor) -> None:
        # Accept everything in every block, every sweep.
        for key in self.accept.keys():
            self.accept[key].assign_add(int(self._block_sizes[key]))

        # Save this draw.
        self.saved.assign_add(1)

        # Accumulate a constant "theta draw" so posterior means are deterministic.
        self.sums["beta"].assign_add(tf.ones([self.M, self.J], tf.float64) * 0.6)
        self.sums["alpha"].assign_add(tf.ones([self.M, self.J], tf.float64) * 2.0)
        self.sums["v"].assign_add(tf.ones([self.M, self.J], tf.float64) * 0.7)
        self.sums["fc"].assign_add(tf.ones([self.M, self.J], tf.float64) * 0.4)
        self.sums["lambda"].assign_add(tf.ones([self.M, self.N], tf.float64) * 0.3)
        self.sums["u_scale"].assign_add(tf.ones([self.M], tf.float64) * 1.5)

    _patch_iteration_step(est, fake_step)

    k = {
        "beta": 0.11,
        "alpha": 0.12,
        "v": 0.13,
        "fc": 0.14,
        "lambda": 0.15,
        "u_scale": 0.16,
    }
    est.fit(n_iter=3, k=k)

    out = est.get_results()
    theta_hat = out["theta_hat"]
    accept = out["accept"]

    assert out["n_saved"] == 3

    # Shapes
    assert theta_hat["beta"].shape == (est.M, est.J)
    assert theta_hat["alpha"].shape == (est.M, est.J)
    assert theta_hat["v"].shape == (est.M, est.J)
    assert theta_hat["fc"].shape == (est.M, est.J)
    assert theta_hat["lambda"].shape == (est.M, est.N)
    assert theta_hat["u_scale"].shape == (est.M,)

    # Values (posterior means) - allow tiny float rounding
    np.testing.assert_allclose(theta_hat["beta"], 0.6, rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(theta_hat["alpha"], 2.0, rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(theta_hat["v"], 0.7, rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(theta_hat["fc"], 0.4, rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(theta_hat["lambda"], 0.3, rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(theta_hat["u_scale"], 1.5, rtol=0.0, atol=_ATOL)

    # Acceptance rates: 1.0 if we accept every element each sweep.
    for key, rate in accept.items():
        assert np.isfinite(rate)
        np.testing.assert_allclose(rate, 1.0, rtol=0.0, atol=_ATOL)

    # Step sizes were assigned into estimator.k
    np.testing.assert_allclose(est.k["beta"].numpy(), k["beta"], rtol=0.0, atol=_ATOL)
    np.testing.assert_allclose(
        est.k["u_scale"].numpy(), k["u_scale"], rtol=0.0, atol=_ATOL
    )


def test_accept_rate_denominator_uses_block_sizes() -> None:
    est = _make_estimator()

    def fake_step(self: StockpilingEstimator, it: tf.Tensor) -> None:
        # Accept half of beta elements per sweep; accept nothing else.
        beta_block = int(self._block_sizes["beta"])
        self.accept["beta"].assign_add(int(beta_block // 2))

        # Save draw and add minimal sums so get_results can compute theta_hat.
        self.saved.assign_add(1)
        self.sums["beta"].assign_add(tf.ones([self.M, self.J], tf.float64) * 0.5)
        self.sums["alpha"].assign_add(tf.zeros([self.M, self.J], tf.float64))
        self.sums["v"].assign_add(tf.zeros([self.M, self.J], tf.float64))
        self.sums["fc"].assign_add(tf.zeros([self.M, self.J], tf.float64))
        self.sums["lambda"].assign_add(tf.zeros([self.M, self.N], tf.float64))
        self.sums["u_scale"].assign_add(tf.zeros([self.M], tf.float64))

    _patch_iteration_step(est, fake_step)

    k = {
        "beta": 0.2,
        "alpha": 0.2,
        "v": 0.2,
        "fc": 0.2,
        "lambda": 0.2,
        "u_scale": 0.2,
    }
    est.fit(n_iter=4, k=k)

    out = est.get_results()
    accept = out["accept"]

    beta_block = int(est._block_sizes["beta"])
    expected_beta_rate = (beta_block // 2) / max(1, beta_block)
    np.testing.assert_allclose(accept["beta"], expected_beta_rate, rtol=0.0, atol=_ATOL)

    # Other blocks untouched in this fake step should have 0 acceptance.
    for key in ["alpha", "v", "fc", "lambda", "u_scale"]:
        np.testing.assert_allclose(accept[key], 0.0, rtol=0.0, atol=_ATOL)
