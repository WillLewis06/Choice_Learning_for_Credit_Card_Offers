# tests/ching/test_stockpiling_estimator.py
"""
Unit tests for `ching.stockpiling_estimator` (Phase-3, multi-product).

These tests focus on estimator-layer behavior only:
- input validation passthrough (fit inputs)
- correct return structure, shapes, and acceptance-rate aggregation
- correct propagation of z updates across iterations
- predict_probabilities calls the posterior prediction function with the right tensors

We patch RW-MH update kernels and iteration diagnostics to avoid expensive DP/likelihood work.
"""

from __future__ import annotations

import os
from unittest.mock import Mock

import numpy as np
import pytest

# Reduce TensorFlow C++ logging (must be set before importing TF).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

import ching.stockpiling_estimator as est_mod  # noqa: E402
from ching.stockpiling_estimator import StockpilingEstimator  # noqa: E402


_ATOL = 1e-12


def _row_stochastic(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    x = rng.uniform(0.1, 1.0, size=shape)
    x = x / x.sum(axis=-1, keepdims=True)
    return x


def _make_tiny_inputs(seed: int = 123) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    M, N, J, T, S = 2, 3, 2, 4, 3
    I_max = 2

    a_mnjt = rng.integers(0, 2, size=(M, N, J, T), dtype=np.int32)
    s_mjt = rng.integers(0, S, size=(M, J, T), dtype=np.int32)

    u_mj = rng.normal(0.0, 1.0, size=(M, J)).astype(np.float64)

    price_vals_mj = rng.uniform(1.0, 5.0, size=(M, J, S)).astype(np.float64)
    P_price_mj = _row_stochastic(rng, (M, J, S, S)).astype(np.float64)

    lambda_mn = rng.uniform(0.1, 0.9, size=(M, N)).astype(np.float64)

    pi_I0 = np.full((I_max + 1,), 1.0 / (I_max + 1), dtype=np.float64)

    sigmas = {
        "z_beta": 1.0,
        "z_alpha": 1.0,
        "z_v": 1.0,
        "z_fc": 1.0,
        "z_u_scale": 1.0,
    }

    return {
        "a_mnjt": a_mnjt,
        "s_mjt": s_mjt,
        "u_mj": u_mj,
        "price_vals_mj": price_vals_mj,
        "P_price_mj": P_price_mj,
        "lambda_mn": lambda_mn,
        "I_max": I_max,
        "pi_I0": pi_I0,
        "waste_cost": 0.2,
        "tol": 1e-8,
        "max_iter": 50,
        "sigmas": sigmas,
        "rng_seed": 999,
    }


def _make_estimator(seed: int = 123) -> StockpilingEstimator:
    inp = _make_tiny_inputs(seed=seed)
    return StockpilingEstimator(
        a_mnjt=inp["a_mnjt"],
        s_mjt=inp["s_mjt"],
        u_mj=inp["u_mj"],
        price_vals_mj=inp["price_vals_mj"],
        P_price_mj=inp["P_price_mj"],
        lambda_mn=inp["lambda_mn"],
        I_max=int(inp["I_max"]),
        pi_I0=inp["pi_I0"],
        waste_cost=float(inp["waste_cost"]),
        tol=float(inp["tol"]),
        max_iter=int(inp["max_iter"]),
        sigmas=inp["sigmas"],
        rng_seed=int(inp["rng_seed"]),
    )


def test_fit_rejects_nonpositive_n_iter() -> None:
    est = _make_estimator()
    k = {"beta": 0.1, "alpha": 0.1, "v": 0.1, "fc": 0.1, "u_scale": 0.1}
    init_theta = {
        "beta": 0.9,
        "alpha": np.array([1.0, 1.0]),
        "v": np.array([1.0, 1.0]),
        "fc": np.array([0.5, 0.5]),
        "u_scale": np.array([1.0, 1.0]),
    }

    with pytest.raises(ValueError, match="n_iter"):
        est.fit(n_iter=0, k=k, init_theta=init_theta)


def test_fit_returns_expected_shapes_accept_and_z_last_with_patched_updates() -> None:
    est = _make_estimator(seed=321)

    # Patch diagnostics to avoid tf.print and to count calls.
    call_its: list[int] = []

    def fake_report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
        call_its.append(int(it.numpy()))

    # Patch update kernels to deterministic "always accept and add constant step".
    def fake_update_z_beta_scalar(
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
        inputs: dict,
        sigma_z_beta: tf.Tensor,
        k_beta: tf.Tensor,
        rng: tf.random.Generator,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z_new = z_beta + tf.constant([0.1], dtype=tf.float64)
        accepted = tf.ones_like(z_beta, dtype=tf.bool)
        return z_new, accepted

    def fake_update_z_alpha_j(
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
        inputs: dict,
        sigma_z_alpha: tf.Tensor,
        k_alpha: tf.Tensor,
        rng: tf.random.Generator,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z_new = z_alpha + tf.constant(0.2, dtype=tf.float64)
        accepted = tf.ones_like(z_alpha, dtype=tf.bool)
        return z_new, accepted

    def fake_update_z_v_j(
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
        inputs: dict,
        sigma_z_v: tf.Tensor,
        k_v: tf.Tensor,
        rng: tf.random.Generator,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z_new = z_v + tf.constant(0.3, dtype=tf.float64)
        accepted = tf.ones_like(z_v, dtype=tf.bool)
        return z_new, accepted

    def fake_update_z_fc_j(
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
        inputs: dict,
        sigma_z_fc: tf.Tensor,
        k_fc: tf.Tensor,
        rng: tf.random.Generator,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z_new = z_fc + tf.constant(0.4, dtype=tf.float64)
        accepted = tf.ones_like(z_fc, dtype=tf.bool)
        return z_new, accepted

    def fake_update_z_u_scale_m(
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
        inputs: dict,
        sigma_z_u_scale: tf.Tensor,
        k_u_scale: tf.Tensor,
        rng: tf.random.Generator,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z_new = z_u_scale + tf.constant(0.5, dtype=tf.float64)
        accepted = tf.ones_like(z_u_scale, dtype=tf.bool)
        return z_new, accepted

    est_mod.report_iteration_progress = fake_report_iteration_progress
    est_mod.update_z_beta_scalar = fake_update_z_beta_scalar
    est_mod.update_z_alpha_j = fake_update_z_alpha_j
    est_mod.update_z_v_j = fake_update_z_v_j
    est_mod.update_z_fc_j = fake_update_z_fc_j
    est_mod.update_z_u_scale_m = fake_update_z_u_scale_m

    k = {"beta": 0.11, "alpha": 0.12, "v": 0.13, "fc": 0.14, "u_scale": 0.15}

    init_theta = {
        "beta": 0.9,
        "alpha": np.array([1.0, 2.0]),
        "v": np.array([0.5, 1.5]),
        "fc": np.array([0.2, 0.4]),
        "u_scale": np.array([1.0, 1.2]),
    }

    out = est.fit(n_iter=3, k=k, init_theta=init_theta)

    # Diagnostics called each iteration
    assert call_its == [0, 1, 2]

    # Return structure
    assert set(out.keys()) == {"theta_mean", "accept", "n_saved", "z_last"}
    assert out["n_saved"] == 3

    theta_mean = out["theta_mean"]
    accept = out["accept"]
    z_last = out["z_last"]

    # Shapes
    assert theta_mean["beta"].shape == ()
    assert theta_mean["alpha"].shape == (est.J,)
    assert theta_mean["v"].shape == (est.J,)
    assert theta_mean["fc"].shape == (est.J,)
    assert theta_mean["u_scale"].shape == (est.M,)

    # Acceptance rates: all ones due to patched kernels
    for key in ["beta", "alpha", "v", "fc", "u_scale"]:
        assert key in accept
        np.testing.assert_allclose(accept[key], 1.0, rtol=0.0, atol=_ATOL)

    # z_last matches initial z + n_iter * delta (in unconstrained space)
    beta0 = float(init_theta["beta"])
    z_beta0 = np.array([np.log(beta0) - np.log(1.0 - beta0)], dtype=np.float64)
    z_alpha0 = np.log(np.asarray(init_theta["alpha"], dtype=np.float64))
    z_v0 = np.log(np.asarray(init_theta["v"], dtype=np.float64))
    z_fc0 = np.log(np.asarray(init_theta["fc"], dtype=np.float64))
    z_u_scale0 = np.log(np.asarray(init_theta["u_scale"], dtype=np.float64))

    np.testing.assert_allclose(
        z_last["z_beta"].numpy(), z_beta0 + 3 * 0.1, rtol=0.0, atol=_ATOL
    )
    np.testing.assert_allclose(
        z_last["z_alpha"].numpy(), z_alpha0 + 3 * 0.2, rtol=0.0, atol=_ATOL
    )
    np.testing.assert_allclose(
        z_last["z_v"].numpy(), z_v0 + 3 * 0.3, rtol=0.0, atol=_ATOL
    )
    np.testing.assert_allclose(
        z_last["z_fc"].numpy(), z_fc0 + 3 * 0.4, rtol=0.0, atol=_ATOL
    )
    np.testing.assert_allclose(
        z_last["z_u_scale"].numpy(), z_u_scale0 + 3 * 0.5, rtol=0.0, atol=_ATOL
    )

    # Estimator fields are populated
    assert est.z is not None
    assert est.theta_mean is not None
    assert est.accept is not None
    assert est.n_saved == 3


def test_predict_probabilities_calls_backend_and_returns_tensor() -> None:
    est = _make_estimator(seed=777)

    backend = Mock()

    def fake_predict_p_buy_mnjt_from_theta(
        theta: dict[str, tf.Tensor], inputs: dict
    ) -> tf.Tensor:
        backend(theta=theta, inputs=inputs)
        M = int(inputs["a_mnjt"].shape[0])
        N = int(inputs["a_mnjt"].shape[1])
        J = int(inputs["a_mnjt"].shape[2])
        T = int(inputs["a_mnjt"].shape[3])
        return tf.fill((M, N, J, T), tf.constant(0.5, dtype=tf.float64))

    est_mod.predict_p_buy_mnjt_from_theta = fake_predict_p_buy_mnjt_from_theta

    theta = {
        "beta": tf.constant(0.9, dtype=tf.float64),
        "alpha": tf.constant([1.0, 2.0], dtype=tf.float64),
        "v": tf.constant([0.5, 1.5], dtype=tf.float64),
        "fc": tf.constant([0.2, 0.4], dtype=tf.float64),
        "u_scale": tf.constant([1.0, 1.2], dtype=tf.float64),
    }

    p = est.predict_probabilities(theta=theta)
    assert p.shape == (est.M, est.N, est.J, est.T)
    assert p.dtype == tf.float64
    np.testing.assert_allclose(p.numpy(), 0.5, rtol=0.0, atol=_ATOL)

    assert backend.call_count == 1
    called = backend.call_args.kwargs
    assert set(called["theta"].keys()) == {"beta", "alpha", "v", "fc", "u_scale"}
