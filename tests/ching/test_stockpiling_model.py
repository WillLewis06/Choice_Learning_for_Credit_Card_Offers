# tests/ching/test_stockpiling_model.py
"""
Unit tests for `ching.stockpiling_model`.

These tests validate core Phase-3 mechanics only:
- transformations from unconstrained z-blocks to constrained parameters,
- inventory-map construction,
- flow-utility construction (including boundary penalties),
- price-state slice selection,
- forward filtering of hidden inventory (log-likelihood),
- DP + CCP construction,
- end-to-end likelihood from constrained theta.

The tests avoid pytest fixtures. Instead, they build a tiny synthetic environment via
`ching_conftest` helper functions (called directly like normal Python functions).
"""

from __future__ import annotations

import numpy as np

import ching_conftest as cc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

import ching.stockpiling_model as sm


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
    Build canonical TF inputs for model calls.

    Returns a dict containing:
      a_imt, p_state_mt: int32
      u_m, price_vals, P_price, pi_I0: float64
      waste_cost, eps: float64
      tol: float64
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
    sm.InventoryMaps,
]:
    """
    Build the small deterministic Phase-3 objects used by all tests.

    Returns:
      tiny_dims:   dict with M,N,T,S,I_max
      tf_inputs:   canonical TF inputs (see `_build_tf_inputs`)
      z_blocks_tf: unconstrained blocks (float64) at the prior mode (all zeros)
      maps:        inventory maps tuple
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

    maps = sm.build_inventory_maps(
        tf.convert_to_tensor(int(tiny_dims["I_max"]), dtype=tf.int32)
    )
    return tiny_dims, tf_inputs, z_blocks_tf, maps


def test_unconstrained_to_theta_shapes_and_constraints() -> None:
    """z-block transforms should produce correctly-shaped, constrained parameters."""
    tiny_dims, _, z_blocks_tf, _ = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    M, N = tiny_dims["M"], tiny_dims["N"]

    assert set(theta.keys()) == {"beta", "alpha", "v", "fc", "lambda_c", "u_scale"}

    assert tuple(theta["beta"].shape) == (M, N)
    assert tuple(theta["alpha"].shape) == (M, N)
    assert tuple(theta["v"].shape) == (M, N)
    assert tuple(theta["fc"].shape) == (M, N)
    assert tuple(theta["lambda_c"].shape) == (M, N)
    assert tuple(theta["u_scale"].shape) == (M,)

    # z=0 -> sigmoid=0.5, exp=1
    np.testing.assert_allclose(theta["beta"].numpy(), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["lambda_c"].numpy(), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["alpha"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["v"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["fc"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["u_scale"].numpy(), 1.0, rtol=0.0, atol=0.0)

    beta = theta["beta"].numpy()
    lam = theta["lambda_c"].numpy()
    assert np.all((beta > 0.0) & (beta < 1.0))
    assert np.all((lam > 0.0) & (lam < 1.0))
    assert np.all(theta["alpha"].numpy() > 0.0)
    assert np.all(theta["v"].numpy() > 0.0)
    assert np.all(theta["fc"].numpy() > 0.0)
    assert np.all(theta["u_scale"].numpy() > 0.0)


def test_build_inventory_maps_correctness() -> None:
    """Inventory maps should encode boundary-safe up/down moves and masks."""
    tiny_dims, _, _, maps = _tiny_env()
    idx_down, idx_up, stockout_mask, at_cap_mask = maps

    i_max = int(tiny_dims["I_max"])
    I = i_max + 1

    assert tuple(idx_down.shape) == (I,)
    assert tuple(idx_up.shape) == (I,)
    assert tuple(stockout_mask.shape) == (I,)
    assert tuple(at_cap_mask.shape) == (I,)

    # For i_max=2: down = [0,0,1], up = [1,2,2]
    np.testing.assert_array_equal(
        idx_down.numpy(), np.asarray([0, 0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(idx_up.numpy(), np.asarray([1, 2, 2], dtype=np.int32))

    np.testing.assert_array_equal(stockout_mask.numpy(), np.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(at_cap_mask.numpy(), np.asarray([0.0, 0.0, 1.0]))


def test_make_flow_utilities_penalties() -> None:
    """Flow-utility tensors should have correct shapes and boundary penalties."""
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    _, _, stockout_mask, at_cap_mask = maps
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    u0, u1 = sm.make_flow_utilities(
        u_m=tf_inputs["u_m"],
        price_vals=tf_inputs["price_vals"],
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        stockout_mask=stockout_mask,
        at_cap_mask=at_cap_mask,
    )

    M, N = tiny_dims["M"], tiny_dims["N"]
    S = int(tf_inputs["price_vals"].shape[0])
    i_max = int(tiny_dims["I_max"])
    I = i_max + 1

    assert tuple(u0.shape) == (M, N, S, I)
    assert tuple(u1.shape) == (M, N, S, I)

    u0_np = u0.numpy()
    u1_np = u1.numpy()
    v_np = theta["v"].numpy()
    lam_np = theta["lambda_c"].numpy()
    waste = float(tf_inputs["waste_cost"].numpy())

    expected_u0_i0 = np.broadcast_to(-v_np[:, :, None], (M, N, S))
    np.testing.assert_allclose(u0_np[:, :, :, 0], expected_u0_i0)

    np.testing.assert_allclose(u0_np[:, :, :, 1:], 0.0)

    diff_cap = u1_np[:, :, :, i_max] - u1_np[:, :, :, i_max - 1]
    expected_pen = np.broadcast_to(-waste * (1.0 - lam_np)[:, :, None], (M, N, S))
    np.testing.assert_allclose(diff_cap, expected_pen)


def test_select_pi_by_state_gathers_correct_state() -> None:
    """`select_pi_by_state` must gather the correct price-state slice per market."""
    tiny_dims, _, _, _ = _tiny_env()
    M, N, S, i_max = tiny_dims["M"], tiny_dims["N"], tiny_dims["S"], tiny_dims["I_max"]
    I = i_max + 1
    assert S == 2

    # ccp_buy[m,n,0,i]=0.1, ccp_buy[m,n,1,i]=0.9
    ccp_buy = tf.concat(
        [
            tf.fill([M, N, 1, I], tf.constant(0.1, tf.float64)),
            tf.fill([M, N, 1, I], tf.constant(0.9, tf.float64)),
        ],
        axis=2,
    )

    s_mt = tf.constant([0, 1], dtype=tf.int32)
    pi_mni = sm.select_pi_by_state(ccp_buy, s_mt)

    assert tuple(pi_mni.shape) == (M, N, I)
    pi_np = pi_mni.numpy()

    np.testing.assert_allclose(pi_np[0, :, :], 0.1)
    np.testing.assert_allclose(pi_np[1, :, :], 0.9)


def test_forward_filter_constant_ccp_half_gives_T_log_half() -> None:
    """With constant CCP=0.5 and all-zero actions, log-likelihood equals T*log(0.5)."""
    tiny_dims, tf_inputs, _, maps = _tiny_env()
    M, N, T, S, i_max = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["T"],
        tiny_dims["S"],
        tiny_dims["I_max"],
    )
    I = i_max + 1

    ccp_buy = tf.fill([M, N, S, I], tf.constant(0.5, tf.float64))
    a_imt = tf.zeros([M, N, T], dtype=tf.int32)

    ll_mn = sm.loglik_hidden_inventory_mn(
        a_imt=a_imt,
        p_state_mt=tf_inputs["p_state_mt"],
        ccp_buy=ccp_buy,
        pi_I0=tf_inputs["pi_I0"],
        lambda_c_mn=tf.fill([M, N], tf.constant(0.3, tf.float64)),
        eps=tf_inputs["eps"],
        maps=maps,
    )

    assert tuple(ll_mn.shape) == (M, N)
    expected = T * np.log(0.5)
    np.testing.assert_allclose(ll_mn.numpy(), expected, rtol=0.0, atol=1e-12)


def test_solve_ccp_buy_shape_range_and_monotone_in_price_when_price_is_absorbing() -> (
    None
):
    """
    With an absorbing price-state transition, a lower current price should weakly
    increase Pr(buy) for every inventory level.
    """
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    tol = tf.constant(1.0e-6, tf.float64)
    max_iter = tf.constant(80, tf.int32)

    S = int(tf_inputs["price_vals"].shape[0])
    p_absorb = tf.eye(S, dtype=tf.float64)

    ccp_buy = sm.solve_ccp_buy(
        u_m=tf_inputs["u_m"],
        price_vals=tf_inputs["price_vals"],
        P_price=p_absorb,
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )

    M, N, I = tiny_dims["M"], tiny_dims["N"], tiny_dims["I_max"] + 1
    assert tuple(ccp_buy.shape) == (M, N, S, I)

    ccp_np = ccp_buy.numpy()
    assert np.isfinite(ccp_np).all()
    assert np.all((ccp_np > 0.0) & (ccp_np < 1.0))

    # With price_vals=[1.0, 0.8], state 1 is cheaper => buy prob should be >= for each I
    assert np.all(ccp_np[:, :, 1, :] >= ccp_np[:, :, 0, :] - 1e-12)


def test_loglik_mn_from_theta_runs_end_to_end_and_is_finite() -> None:
    """`loglik_mn_from_theta` should run end-to-end and return finite (M,N) outputs."""
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    ll_mn = sm.loglik_mn_from_theta(
        theta=theta,
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
