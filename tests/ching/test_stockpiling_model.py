# tests/ching/test_stockpiling_model.py
"""Unit tests for `ching.stockpiling_model` (Phase-3 core mechanics).

These tests validate:
- transformation from unconstrained z-blocks to constrained parameters,
- inventory-map construction,
- flow-utility construction (including boundary penalties),
- DP + CCP construction.

No pytest fixtures are used. Tests build a tiny deterministic environment via
`ching_conftest` helper functions (called directly like normal Python functions).
"""

from __future__ import annotations

import numpy as np

import ching_conftest as cc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

import ching.stockpiling_model as sm


def _to_tf_float64_dict(arrs: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    """Convert a dict of NumPy arrays to float64 TensorFlow tensors."""
    return {k: tf.convert_to_tensor(v, dtype=tf.float64) for k, v in arrs.items()}


def _build_tf_inputs(
    panel_np: dict[str, np.ndarray],
    u_mj_np: np.ndarray,
    price_proc: dict[str, np.ndarray],
    lambda_mn_np: np.ndarray,
    dp_cfg: dict[str, float | int],
) -> dict[str, tf.Tensor]:
    """Build canonical TF inputs for model calls.

    Returns a dict containing:
      a_mnjt, s_mjt: int32
      u_mj, price_vals_mj, P_price_mj, lambda_mn: float64
      waste_cost: float64
    """
    return {
        "a_mnjt": tf.convert_to_tensor(panel_np["a_mnjt"], dtype=tf.int32),
        "s_mjt": tf.convert_to_tensor(panel_np["s_mjt"], dtype=tf.int32),
        "u_mj": tf.convert_to_tensor(u_mj_np, dtype=tf.float64),
        "price_vals_mj": tf.convert_to_tensor(
            price_proc["price_vals_mj"], dtype=tf.float64
        ),
        "P_price_mj": tf.convert_to_tensor(price_proc["P_price_mj"], dtype=tf.float64),
        "lambda_mn": tf.convert_to_tensor(lambda_mn_np, dtype=tf.float64),
        "waste_cost": tf.convert_to_tensor(
            float(dp_cfg["waste_cost"]), dtype=tf.float64
        ),
    }


def _tiny_env() -> tuple[
    dict[str, int],
    dict[str, tf.Tensor],
    dict[str, tf.Tensor],
    sm.InventoryMaps,
]:
    """Build the small deterministic Phase-3 objects used by all tests."""
    tf.random.set_seed(0)

    tiny_dims = cc.tiny_dims()
    dp_cfg = cc.tiny_dp_config()

    panel = cc.panel_np(tiny_dims)
    u_mj = cc.u_mj_np(tiny_dims)
    price_proc = cc.price_process(tiny_dims)
    lambda_mn = cc.lambda_mn_np(tiny_dims)

    tf_inputs = _build_tf_inputs(panel, u_mj, price_proc, lambda_mn, dp_cfg)
    z_blocks_tf = _to_tf_float64_dict(cc.z_blocks_np(tiny_dims))

    maps = cc.inventory_maps_tf(int(tiny_dims["I_max"]))
    return tiny_dims, tf_inputs, z_blocks_tf, maps


def test_unconstrained_to_theta_shapes_and_constraints() -> None:
    """z-block transforms should produce correctly-shaped, constrained parameters."""
    tiny_dims, _, z_blocks_tf, _ = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    M, J = tiny_dims["M"], tiny_dims["J"]

    assert set(theta.keys()) == {"beta", "alpha", "v", "fc", "u_scale"}

    assert tuple(theta["beta"].shape) == ()
    assert tuple(theta["alpha"].shape) == (J,)
    assert tuple(theta["v"].shape) == (J,)
    assert tuple(theta["fc"].shape) == (J,)
    assert tuple(theta["u_scale"].shape) == (M,)

    # z=0 -> sigmoid=0.5, exp=1
    np.testing.assert_allclose(theta["beta"].numpy(), 0.5, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["alpha"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["v"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["fc"].numpy(), 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(theta["u_scale"].numpy(), 1.0, rtol=0.0, atol=0.0)

    beta = float(theta["beta"].numpy())
    assert 0.0 < beta < 1.0
    assert np.all(theta["alpha"].numpy() > 0.0)
    assert np.all(theta["v"].numpy() > 0.0)
    assert np.all(theta["fc"].numpy() > 0.0)
    assert np.all(theta["u_scale"].numpy() > 0.0)


def test_build_inventory_maps_correctness() -> None:
    """Inventory maps should encode boundary-safe up/down moves and masks."""
    tiny_dims, _, _, maps = _tiny_env()
    I_vals, stockout_mask, at_cap_mask, idx_down, idx_up = maps

    i_max = int(tiny_dims["I_max"])
    I = i_max + 1

    assert tuple(I_vals.shape) == (I,)
    assert tuple(idx_down.shape) == (I,)
    assert tuple(idx_up.shape) == (I,)
    assert tuple(stockout_mask.shape) == (I,)
    assert tuple(at_cap_mask.shape) == (I,)

    # For i_max=2: I=[0,1,2], down=[0,0,1], up=[1,2,2]
    np.testing.assert_array_equal(I_vals.numpy(), np.asarray([0, 1, 2], dtype=np.int32))
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
        lambda_mn=tf_inputs["lambda_mn"],
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

    v_j = theta["v"].numpy()  # (J,)
    lam_mn = tf_inputs["lambda_mn"].numpy()  # (M,N)
    waste = float(tf_inputs["waste_cost"].numpy())

    # u0 = -v_j * 1{I==0} broadcast over (M,N,S)
    expected_u0_i0 = -v_j[None, None, :, None]  # (1,1,J,1)
    expected_u0_i0 = np.broadcast_to(expected_u0_i0, (M, N, J, S))
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


def test_solve_ccp_buy_shape_range_and_monotone_in_price_when_price_is_absorbing() -> (
    None
):
    """With an absorbing price-state transition, a lower current price should
    weakly increase Pr(buy) for every inventory level (for each m,n,j)."""
    tiny_dims, tf_inputs, z_blocks_tf, maps = _tiny_env()
    theta = sm.unconstrained_to_theta(z_blocks_tf)

    M, J, S = (
        int(tf_inputs["price_vals_mj"].shape[0]),
        int(tf_inputs["price_vals_mj"].shape[1]),
        int(tf_inputs["price_vals_mj"].shape[2]),
    )

    P_absorb = tf.eye(S, dtype=tf.float64)
    P_absorb_mj = tf.tile(P_absorb[None, None, :, :], [M, J, 1, 1])  # (M,J,S,S)

    ccp_buy, q0, q1 = sm.solve_ccp_buy(
        u_mj=tf_inputs["u_mj"],
        price_vals_mj=tf_inputs["price_vals_mj"],
        P_price_mj=P_absorb_mj,
        theta=theta,
        lambda_mn=tf_inputs["lambda_mn"],
        waste_cost=tf_inputs["waste_cost"],
        maps=maps,
        tol=1.0e-6,
        max_iter=80,
    )

    M0, N0, J0, I0 = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["J"],
        tiny_dims["I_max"] + 1,
    )
    assert tuple(ccp_buy.shape) == (M0, N0, J0, S, I0)
    assert tuple(q0.shape) == (M0, N0, J0, S, I0)
    assert tuple(q1.shape) == (M0, N0, J0, S, I0)

    ccp_np = ccp_buy.numpy()
    assert np.isfinite(ccp_np).all()
    assert np.all((ccp_np > 0.0) & (ccp_np < 1.0))

    # State 1 is cheaper than state 0 under cc.price_process(), so buy prob should be >=
    assert np.all(ccp_np[:, :, :, 1, :] >= ccp_np[:, :, :, 0, :] - 1e-12)
