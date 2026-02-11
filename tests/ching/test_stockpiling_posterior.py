# tests/ching/test_stockpiling_posterior.py
"""
Unit tests for `ching.stockpiling_posterior`.

These tests cover ONLY posterior-layer responsibilities:
- priors on unconstrained z-blocks
- loglik_mn(z, ...) wrapper (delegates to stockpiling_model)
- log-posterior "views" used by RW-MH block updates

Core mechanics (DP, CCPs, inventory filter, maps, transforms) are tested in:
  tests/ching/test_stockpiling_model.py
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import ching_conftest as cc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

import ching.stockpiling_model as sm
import ching.stockpiling_posterior as sp


def _sigmas_tf(sigmas: dict[str, float]) -> dict[str, tf.Tensor]:
    """Convert prior scales to float64 TensorFlow tensors."""
    return {
        k: tf.convert_to_tensor(float(v), dtype=tf.float64) for k, v in sigmas.items()
    }


def _to_tf_float64_dict(arrs: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    """Convert a dict of numpy arrays to float64 TensorFlow tensors."""
    return {k: tf.convert_to_tensor(v, dtype=tf.float64) for k, v in arrs.items()}


def _build_tf_inputs(
    panel_np: dict[str, np.ndarray],
    u_m_np: np.ndarray,
    price_proc: dict[str, np.ndarray],
    pi_i0_np: np.ndarray,
    dp_cfg: dict[str, float | int],
) -> dict[str, tf.Tensor]:
    """
    Build canonical TF inputs for posterior calls.

    Returns a dict containing:
      a_imt, p_state_mt: int32
      u_m, price_vals, P_price, pi_I0: float64
      waste_cost, eps, tol: float64
      max_iter: int32
    """
    return {
        "a_imt": tf.convert_to_tensor(panel_np["a_imt"], dtype=tf.int32),
        "p_state_mt": tf.convert_to_tensor(panel_np["p_state_mt"], dtype=tf.int32),
        "u_m": tf.convert_to_tensor(u_m_np, dtype=tf.float64),
        "price_vals": tf.convert_to_tensor(price_proc["price_vals"], dtype=tf.float64),
        "P_price": tf.convert_to_tensor(price_proc["P_price"], dtype=tf.float64),
        "pi_I0": tf.convert_to_tensor(pi_i0_np, dtype=tf.float64),
        "waste_cost": tf.convert_to_tensor(
            float(dp_cfg["waste_cost"]), dtype=tf.float64
        ),
        "eps": tf.convert_to_tensor(float(dp_cfg["eps"]), dtype=tf.float64),
        "tol": tf.convert_to_tensor(float(dp_cfg["tol"]), dtype=tf.float64),
        "max_iter": tf.convert_to_tensor(int(dp_cfg["max_iter"]), dtype=tf.int32),
    }


def _tiny_env() -> tuple[
    dict[str, int],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
    dict[str, float],
    sm.InventoryMaps,
]:
    """
    Build the small deterministic Phase-3 objects used by all posterior tests.

    Returns:
      tiny_dims:   dict with M,N,T,S,I_max
      tf_inputs:   canonical TF inputs
      z_blocks_tf: unconstrained blocks (float64) at the prior mode (all zeros)
      sigmas:      prior scales (python floats)
      maps:        inventory maps tuple (from stockpiling_model)
    """
    tf.random.set_seed(0)

    tiny_dims = cc.tiny_dims()
    dp_cfg = cc.tiny_dp_config()

    panel_np = cc.panel_np(tiny_dims)
    u_m_np = cc.u_m_np(tiny_dims)
    price_proc = cc.price_process(tiny_dims)
    pi_i0_np = cc.pi_I0_uniform(tiny_dims)

    tf_inputs = _build_tf_inputs(panel_np, u_m_np, price_proc, pi_i0_np, dp_cfg)
    z_blocks_tf = _to_tf_float64_dict(cc.z_blocks_np(tiny_dims))
    sigmas = cc.sigmas()

    maps = sm.build_inventory_maps(
        tf.convert_to_tensor(int(tiny_dims["I_max"]), dtype=tf.int32)
    )
    return tiny_dims, tf_inputs, z_blocks_tf, sigmas, maps


def test_loglik_mn_runs_end_to_end_and_is_finite() -> None:
    """`loglik_mn` should run end-to-end and return finite (M,N) outputs."""
    tiny_dims, tf_inputs, z_blocks_tf, _, maps = _tiny_env()

    ll_mn = sp.loglik_mn(
        z=z_blocks_tf,
        a_imt=tf_inputs["a_imt"],
        p_state_mt=tf_inputs["p_state_mt"],
        u_m=tf_inputs["u_m"],
        price_vals=tf_inputs["price_vals"],
        P_price=tf_inputs["P_price"],
        pi_I0=tf_inputs["pi_I0"],
        waste_cost=tf_inputs["waste_cost"],
        eps=tf_inputs["eps"],
        tol=tf.constant(1.0e-6, tf.float64),
        max_iter=tf.constant(60, tf.int32),
        maps=maps,
    )

    M, N = tiny_dims["M"], tiny_dims["N"]
    assert tuple(ll_mn.shape) == (M, N)
    ll_np = ll_mn.numpy()
    assert np.isfinite(ll_np).all()


def test_logpost_views_combine_ll_and_prior_correctly_with_patch() -> None:
    """
    The log-posterior "view" functions should equal:
      loglik_mn(z) + logprior(z_block)                     (for MN blocks)
      sum_n loglik_mn(z)[m,n] + logprior(z_u_scale[m])     (for market block)
    """
    tiny_dims, tf_inputs, z_blocks_tf, sigmas, maps = _tiny_env()
    M, N = tiny_dims["M"], tiny_dims["N"]
    sigmas_tf = _sigmas_tf(sigmas)

    ll_fake = tf.reshape(
        tf.linspace(tf.constant(-0.2, tf.float64), tf.constant(0.3, tf.float64), M * N),
        [M, N],
    )

    with patch.object(sp, "loglik_mn", return_value=ll_fake):
        out_beta = sp.logpost_z_beta_mn(
            z=z_blocks_tf,
            a_imt=tf_inputs["a_imt"],
            p_state_mt=tf_inputs["p_state_mt"],
            u_m=tf_inputs["u_m"],
            price_vals=tf_inputs["price_vals"],
            P_price=tf_inputs["P_price"],
            pi_I0=tf_inputs["pi_I0"],
            waste_cost=tf_inputs["waste_cost"],
            eps=tf_inputs["eps"],
            tol=tf.constant(1.0e-6, tf.float64),
            max_iter=tf.constant(10, tf.int32),
            maps=maps,
            sigmas=sigmas_tf,
        )

        prior_beta = sp.logprior_normal_mn(z_blocks_tf["z_beta"], sigmas_tf["z_beta"])
        np.testing.assert_allclose(out_beta.numpy(), (ll_fake + prior_beta).numpy())

        out_us = sp.logpost_u_scale_m(
            z=z_blocks_tf,
            a_imt=tf_inputs["a_imt"],
            p_state_mt=tf_inputs["p_state_mt"],
            u_m=tf_inputs["u_m"],
            price_vals=tf_inputs["price_vals"],
            P_price=tf_inputs["P_price"],
            pi_I0=tf_inputs["pi_I0"],
            waste_cost=tf_inputs["waste_cost"],
            eps=tf_inputs["eps"],
            tol=tf.constant(1.0e-6, tf.float64),
            max_iter=tf.constant(10, tf.int32),
            maps=maps,
            sigmas=sigmas_tf,
        )

        prior_us = sp.logprior_normal_m(
            z_blocks_tf["z_u_scale"], sigmas_tf["z_u_scale"]
        )
        np.testing.assert_allclose(
            out_us.numpy(), (tf.reduce_sum(ll_fake, axis=1) + prior_us).numpy()
        )
