# test_stockpiling_posterior.py
"""
Pytests for ching.stockpiling_posterior (multi-product Phase-3).

These tests assume the updated Phase-3 shapes:
  a_mnjt        (M,N,J,T)
  p_state_mjt   (M,J,T)
  u_mj          (M,J)
  price_vals_mj (M,J,S)
  P_price_mj    (M,J,S,S)

Posterior API:
  logprior_normal(z_block, sigma)
  loglik_mnj(z, inputs) -> (M,N,J)
  logpost_z_* views with shapes:
    beta/alpha/v/fc : (M,J)
    lambda          : (M,N)
    u_scale         : (M,)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import tensorflow as tf

import ching_conftest as cc
from ching import stockpiling_model as sm
from ching import stockpiling_posterior as sp


def _to_tf_float64_dict(d: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    return {
        k: tf.convert_to_tensor(np.asarray(v), dtype=tf.float64) for k, v in d.items()
    }


def _sigmas_tf(sigmas: dict[str, float]) -> dict[str, tf.Tensor]:
    return {
        k: tf.convert_to_tensor(float(v), dtype=tf.float64) for k, v in sigmas.items()
    }


def _build_tf_inputs(
    panel_np: dict[str, np.ndarray],
    u_mj_np: np.ndarray,
    price_proc: dict[str, np.ndarray],
    pi_i0_np: np.ndarray,
    dp_cfg: dict[str, float | int],
    maps: sm.InventoryMaps,
) -> dict[str, tf.Tensor]:
    return {
        "a_mnjt": tf.convert_to_tensor(panel_np["a_mnjt"], dtype=tf.int32),
        "p_state_mjt": tf.convert_to_tensor(panel_np["p_state_mjt"], dtype=tf.int32),
        "u_mj": tf.convert_to_tensor(u_mj_np, dtype=tf.float64),
        "price_vals_mj": tf.convert_to_tensor(
            price_proc["price_vals_mj"], dtype=tf.float64
        ),
        "P_price_mj": tf.convert_to_tensor(price_proc["P_price_mj"], dtype=tf.float64),
        "pi_I0": tf.convert_to_tensor(pi_i0_np, dtype=tf.float64),
        "waste_cost": tf.convert_to_tensor(
            float(dp_cfg["waste_cost"]), dtype=tf.float64
        ),
        "eps": tf.convert_to_tensor(float(dp_cfg["eps"]), dtype=tf.float64),
        "tol": tf.convert_to_tensor(float(dp_cfg["tol"]), dtype=tf.float64),
        "max_iter": tf.convert_to_tensor(int(dp_cfg["max_iter"]), dtype=tf.int32),
        "maps": maps,
    }


def _tiny_env() -> tuple[
    dict[str, int],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
]:
    dims = cc.tiny_dims()
    dp_cfg = cc.tiny_dp_config()

    panel_np = cc.panel_np(dims)
    u_mj_np = cc.u_mj_np(dims)
    price_proc = cc.price_process(dims)
    pi_i0_np = cc.pi_I0_uniform(dims)

    maps = sm.build_inventory_maps(
        tf.convert_to_tensor(int(dims["I_max"]), dtype=tf.int32)
    )

    inputs = _build_tf_inputs(panel_np, u_mj_np, price_proc, pi_i0_np, dp_cfg, maps)
    z = _to_tf_float64_dict(cc.z_blocks_np(dims))
    sigma_z = _sigmas_tf(cc.sigmas())

    return dims, inputs, z, sigma_z


def test_logprior_normal_matches_formula() -> None:
    x = tf.constant([[0.0, 1.5], [-2.0, 3.0]], dtype=tf.float64)
    sigma = tf.constant(2.0, dtype=tf.float64)

    out = sp.logprior_normal(x, sigma).numpy()
    expected = (-0.5 * (x.numpy() / 2.0) ** 2).astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_loglik_mnj_runs_end_to_end_and_is_finite() -> None:
    dims, inputs, z, _ = _tiny_env()

    ll_mnj = sp.loglik_mnj(z=z, inputs=inputs)

    M, N, J = int(dims["M"]), int(dims["N"]), int(dims["J"])
    assert tuple(ll_mnj.shape) == (M, N, J)

    ll_np = ll_mnj.numpy()
    assert np.isfinite(ll_np).all()


def test_logpost_views_combine_reduced_ll_and_prior_with_patch() -> None:
    dims, inputs, z, sigma_z = _tiny_env()
    M, N, J = int(dims["M"]), int(dims["N"]), int(dims["J"])

    ll_fake = tf.reshape(
        tf.linspace(
            tf.constant(-0.2, tf.float64), tf.constant(0.3, tf.float64), M * N * J
        ),
        [M, N, J],
    )

    with patch.object(sp, "loglik_mnj", return_value=ll_fake):
        # (M,J) views: sum over N (axis=1)
        out_beta = sp.logpost_z_beta_mj(z=z, inputs=inputs, sigma_z=sigma_z)
        prior_beta = sp.logprior_normal(z["z_beta"], sigma_z["z_beta"])
        expected_beta = tf.reduce_sum(ll_fake, axis=1) + prior_beta
        np.testing.assert_allclose(
            out_beta.numpy(), expected_beta.numpy(), rtol=0, atol=0
        )

        out_alpha = sp.logpost_z_alpha_mj(z=z, inputs=inputs, sigma_z=sigma_z)
        prior_alpha = sp.logprior_normal(z["z_alpha"], sigma_z["z_alpha"])
        expected_alpha = tf.reduce_sum(ll_fake, axis=1) + prior_alpha
        np.testing.assert_allclose(
            out_alpha.numpy(), expected_alpha.numpy(), rtol=0, atol=0
        )

        # (M,N) view: sum over J (axis=2)
        out_lambda = sp.logpost_z_lambda_mn(z=z, inputs=inputs, sigma_z=sigma_z)
        prior_lambda = sp.logprior_normal(z["z_lambda"], sigma_z["z_lambda"])
        expected_lambda = tf.reduce_sum(ll_fake, axis=2) + prior_lambda
        np.testing.assert_allclose(
            out_lambda.numpy(), expected_lambda.numpy(), rtol=0, atol=0
        )

        # (M,) view: sum over N,J (axes=[1,2])
        out_u_scale = sp.logpost_z_u_scale_m(z=z, inputs=inputs, sigma_z=sigma_z)
        prior_u_scale = sp.logprior_normal(z["z_u_scale"], sigma_z["z_u_scale"])
        expected_u_scale = tf.reduce_sum(ll_fake, axis=[1, 2]) + prior_u_scale
        np.testing.assert_allclose(
            out_u_scale.numpy(), expected_u_scale.numpy(), rtol=0, atol=0
        )
