# tests/ching/test_stockpiling_model.py
"""
Unit tests for `ching.stockpiling_model` (Phase-3 core mechanics, multi-product).

These tests validate:
- transformations from unconstrained z-blocks to constrained parameters,
- inventory-map construction,
- flow-utility construction (including boundary penalties),
- price-state slice selection (market×product),
- forward filtering of hidden inventory (log-likelihood),
- DP + CCP construction,
- end-to-end likelihood from constrained theta.

No pytest fixtures are used. Tests build a tiny deterministic environment via
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
    u_mj_np: np.ndarray,
    price_proc: dict[str, np.ndarray],
    pi_i0_np: np.ndarray,
    dp_cfg: dict[str, float | int],
) -> dict[str, tf.Tensor]:
    """
    Build canonical TF inputs for model calls.

    Returns a dict containing:
      a_mnjt, p_state_mjt: int32
      u_mj, price_vals_mj, P_price_mj, pi_I0: float64
      waste_cost, eps, tol: float64
      max_iter: int32
    """
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
      tiny_dims:   dict with M,N,J,T,S,I_max
      tf_inputs:   canonical TF inputs (see `_build_tf_inputs`)
      z_blocks_tf: unconstrained blocks (float64) at the prior mode (all zeros)
      maps:        inventory maps tuple
    """
    tf.random.set_seed(0)

    tiny_dims = cc.tiny_dims()
    dp_cfg = cc.tiny_dp_config()

    panel = cc.panel_np(tiny_dims)
    u_mj = cc.u_mj_np(tiny_dims)
    price_proc = cc.price_process(tiny_dims)
    pi_i0 = cc.pi_I0_uniform(tiny_dims)

    tf_inputs = _build_tf_inputs(panel, u_mj, price_proc, pi_i0, dp_cfg)
    z_blocks_tf = _to_tf_float64_dict(cc.z_blocks_np(tiny_dims))

    maps = sm.build_inventory_maps(
        tf.convert_to_tensor(int(tiny_dims["I_max"]), dtype=tf.int32)
    )
    return tiny_dims, tf_inputs, z_blocks_tf, maps


def test_unconstrained_to_theta_shapes_and_constraints() -> None:
    """z-block transforms should produce correctly-shaped, constrained parameters."""
    tiny_dims, _, z_blocks_tf, _ = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    M, N, J = tiny_dims["M"], tiny_dims["N"], tiny_dims["J"]

    assert set(theta.keys()) == {"beta", "alpha", "v", "fc", "lambda", "u_scale"}

    assert tuple(theta["beta"].shape) == (M, J)
    assert tuple(theta["alpha"].shape) == (M, J)
    assert tuple(theta["v"].shape) == (M, J)
    assert tuple(theta["fc"].shape) == (M, J)
    assert tuple(theta["lambda"].shape) == (M, N)
    assert tuple(theta["u_scale"].shape) == (M,)

    # z=0 -> sigmoid=0.5, exp=1
    np.testing.assert_allclose(theta["beta"].numpy(), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["lambda"].numpy(), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["alpha"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["v"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["fc"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["u_scale"].numpy(), 1.0, rtol=0.0, atol=0.0)

    beta = theta["beta"].numpy()
    lam = theta["lambda"].numpy()
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
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    u0, u1 = sm.make_flow_utilities(
        u_mj=tf_inputs["u_mj"],
        price_vals_mj=tf_inputs["price_vals_mj"],
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        maps=maps,
    )

    M, N, J = tiny_dims["M"], tiny_dims["N"], tiny_dims["J"]
    S = int(tf_inputs["price_vals_mj"].shape[2])
    i_max = int(tiny_dims["I_max"])
    I = i_max + 1

    assert tuple(u0.shape) == (M, N, J, S, I)
    assert tuple(u1.shape) == (M, N, J, S, I)

    u0_np = u0.numpy()
    u1_np = u1.numpy()

    v_mj = theta["v"].numpy()  # (M,J)
    lam_mn = theta["lambda"].numpy()  # (M,N)
    waste = float(tf_inputs["waste_cost"].numpy())

    # u0 = -v_mj * 1{I==0} broadcast over (N,S)
    expected_u0_i0 = np.broadcast_to(-v_mj[:, None, :, None], (M, N, J, S))
    np.testing.assert_allclose(u0_np[:, :, :, :, 0], expected_u0_i0, rtol=0.0, atol=0.0)

    # u0 is 0 for I>0
    np.testing.assert_allclose(u0_np[:, :, :, :, 1:], 0.0, rtol=0.0, atol=0.0)

    # Waste-at-cap penalty applies only at I=I_max and equals -waste*(1-lambda_mn),
    # broadcast over (J,S). Since base_buy is I-invariant, the difference between
    # I=I_max and I=I_max-1 isolates the penalty.
    diff_cap = u1_np[:, :, :, :, i_max] - u1_np[:, :, :, :, i_max - 1]  # (M,N,J,S)
    expected_pen = -waste * (1.0 - lam_mn)[:, :, None, None]  # (M,N,1,1)
    expected_pen = np.broadcast_to(expected_pen, (M, N, J, S))
    np.testing.assert_allclose(diff_cap, expected_pen, rtol=0.0, atol=1e-12)


def test_select_pi_by_state_gathers_correct_state() -> None:
    """`select_pi_by_state` must gather the correct price-state slice per (market,product)."""
    tiny_dims, _, _, _ = _tiny_env()
    M, N, J, S, i_max = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["J"],
        tiny_dims["S"],
        tiny_dims["I_max"],
    )
    I = i_max + 1
    assert S == 2
    assert M == 2 and J == 2  # for this test we rely on tiny_dims() values

    # Build ccp_buy with distinct values by state:
    # state 0 => 0.1, state 1 => 0.9 (constant over i)
    ccp_s0 = tf.fill([M, N, J, 1, I], tf.constant(0.1, tf.float64))
    ccp_s1 = tf.fill([M, N, J, 1, I], tf.constant(0.9, tf.float64))
    ccp_buy = tf.concat([ccp_s0, ccp_s1], axis=3)  # (M,N,J,S,I)

    # Pick different states for each (m,j)
    # m0: [0,1], m1: [1,0]
    s_mj = tf.constant([[0, 1], [1, 0]], dtype=tf.int32)  # (M,J)

    pi_mnjI = sm.select_pi_by_state(ccp_buy, s_mj)
    assert tuple(pi_mnjI.shape) == (M, N, J, I)

    pi_np = pi_mnjI.numpy()
    # (m0,j0)=0.1, (m0,j1)=0.9, (m1,j0)=0.9, (m1,j1)=0.1
    np.testing.assert_allclose(pi_np[0, :, 0, :], 0.1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pi_np[0, :, 1, :], 0.9, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pi_np[1, :, 0, :], 0.9, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pi_np[1, :, 1, :], 0.1, rtol=0.0, atol=0.0)


def test_forward_filter_constant_ccp_half_gives_T_log_half() -> None:
    """With CCP=0.5 and all-zero actions, log-likelihood equals T*log(0.5) per (m,n,j)."""
    tiny_dims, tf_inputs, _, maps = _tiny_env()
    M, N, J, T, S, i_max = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["J"],
        tiny_dims["T"],
        tiny_dims["S"],
        tiny_dims["I_max"],
    )
    I = i_max + 1

    ccp_buy = tf.fill([M, N, J, S, I], tf.constant(0.5, tf.float64))
    a_mnjt = tf.zeros([M, N, J, T], dtype=tf.int32)

    ll_mnj = sm.loglik_hidden_inventory_mnj(
        a_mnjt=a_mnjt,
        p_state_mjt=tf_inputs["p_state_mjt"],
        ccp_buy=ccp_buy,
        pi_I0=tf_inputs["pi_I0"],
        lambda_mn=tf.fill([M, N], tf.constant(0.3, tf.float64)),
        eps=tf_inputs["eps"],
        maps=maps,
    )

    assert tuple(ll_mnj.shape) == (M, N, J)
    expected = T * np.log(0.5)
    np.testing.assert_allclose(ll_mnj.numpy(), expected, rtol=0.0, atol=1e-12)


def test_solve_ccp_buy_shape_range_and_monotone_in_price_when_price_is_absorbing() -> (
    None
):
    """
    With an absorbing price-state transition, a lower current price should weakly
    increase Pr(buy) for every inventory level (for each m,n,j).
    """
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    tol = tf.constant(1.0e-6, tf.float64)
    max_iter = tf.constant(80, tf.int32)

    M, J, S = (
        int(tf_inputs["price_vals_mj"].shape[0]),
        int(tf_inputs["price_vals_mj"].shape[1]),
        int(tf_inputs["price_vals_mj"].shape[2]),
    )

    P_absorb = tf.eye(S, dtype=tf.float64)
    P_absorb_mj = tf.tile(P_absorb[None, None, :, :], [M, J, 1, 1])  # (M,J,S,S)

    ccp_buy = sm.solve_ccp_buy(
        u_mj=tf_inputs["u_mj"],
        price_vals_mj=tf_inputs["price_vals_mj"],
        P_price_mj=P_absorb_mj,
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )

    M0, N0, J0, I0 = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["J"],
        tiny_dims["I_max"] + 1,
    )
    assert tuple(ccp_buy.shape) == (M0, N0, J0, S, I0)

    ccp_np = ccp_buy.numpy()
    assert np.isfinite(ccp_np).all()
    assert np.all((ccp_np > 0.0) & (ccp_np < 1.0))

    # With price_vals=[1.0, 0.8], state 1 is cheaper => buy prob should be >= for each I
    assert np.all(ccp_np[:, :, :, 1, :] >= ccp_np[:, :, :, 0, :] - 1e-12)


def test_loglik_mnj_from_theta_runs_end_to_end_and_is_finite() -> None:
    """`loglik_mnj_from_theta` should run end-to-end and return finite (M,N,J) outputs."""
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    ll_mnj = sm.loglik_mnj_from_theta(
        theta=theta,
        a_mnjt=tf_inputs["a_mnjt"],
        p_state_mjt=tf_inputs["p_state_mjt"],
        u_mj=tf_inputs["u_mj"],
        price_vals_mj=tf_inputs["price_vals_mj"],
        P_price_mj=tf_inputs["P_price_mj"],
        pi_I0=tf_inputs["pi_I0"],
        waste_cost=tf_inputs["waste_cost"],
        eps=tf_inputs["eps"],
        tol=tf.constant(1.0e-6, tf.float64),
        max_iter=tf.constant(60, tf.int32),
        maps=maps,
    )

    M, N, J = tiny_dims["M"], tiny_dims["N"], tiny_dims["J"]
    assert tuple(ll_mnj.shape) == (M, N, J)
    ll_np = ll_mnj.numpy()
    assert np.isfinite(ll_np).all()
