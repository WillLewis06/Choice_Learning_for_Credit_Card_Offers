# tests/ching/test_stockpiling_posterior.py
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import ching.stockpiling_posterior as sp


@pytest.fixture
def inventory_maps_tf(tiny_dims: dict[str, int]):
    """Inventory maps as (idx_down, idx_up, stockout_mask, at_cap_mask)."""
    I_max = tf.constant(int(tiny_dims["I_max"]), dtype=tf.int32)
    return sp.build_inventory_maps(I_max)


def _sigmas_tf(sigmas: dict[str, float]) -> dict[str, tf.Tensor]:
    return {
        k: tf.convert_to_tensor(float(v), dtype=tf.float64) for k, v in sigmas.items()
    }


def test_unconstrained_to_theta_shapes_and_constraints(
    tiny_dims: dict[str, int],
    z_blocks_tf: dict[str, tf.Tensor],
) -> None:
    theta = sp.unconstrained_to_theta(z_blocks_tf)

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

    # Range checks
    beta = theta["beta"].numpy()
    lam = theta["lambda_c"].numpy()
    assert np.all((beta > 0.0) & (beta < 1.0))
    assert np.all((lam > 0.0) & (lam < 1.0))
    assert np.all(theta["alpha"].numpy() > 0.0)
    assert np.all(theta["v"].numpy() > 0.0)
    assert np.all(theta["fc"].numpy() > 0.0)
    assert np.all(theta["u_scale"].numpy() > 0.0)


def test_build_inventory_maps_correctness(
    tiny_dims: dict[str, int],
    inventory_maps_tf,
) -> None:
    idx_down, idx_up, stockout_mask, at_cap_mask = inventory_maps_tf
    I_max = tiny_dims["I_max"]
    I = I_max + 1

    assert tuple(idx_down.shape) == (I,)
    assert tuple(idx_up.shape) == (I,)
    assert tuple(stockout_mask.shape) == (I,)
    assert tuple(at_cap_mask.shape) == (I,)

    # For I_max=2: down = [0,0,1], up = [1,2,2]
    np.testing.assert_array_equal(
        idx_down.numpy(), np.asarray([0, 0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(idx_up.numpy(), np.asarray([1, 2, 2], dtype=np.int32))

    np.testing.assert_array_equal(stockout_mask.numpy(), np.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(at_cap_mask.numpy(), np.asarray([0.0, 0.0, 1.0]))


def test_make_flow_utilities_penalties(
    tf_inputs: dict[str, tf.Tensor],
    z_blocks_tf: dict[str, tf.Tensor],
    inventory_maps_tf,
    tiny_dims: dict[str, int],
) -> None:
    _, _, stockout_mask, at_cap_mask = inventory_maps_tf
    theta = sp.unconstrained_to_theta(z_blocks_tf)

    u0, u1 = sp.make_flow_utilities(
        u_m=tf_inputs["u_m"],
        price_vals=tf_inputs["price_vals"],
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        stockout_mask=stockout_mask,
        at_cap_mask=at_cap_mask,
    )

    M, N = tiny_dims["M"], tiny_dims["N"]
    S = int(tf_inputs["price_vals"].shape[0])
    I_max = tiny_dims["I_max"]
    I = I_max + 1

    assert tuple(u0.shape) == (M, N, S, I)
    assert tuple(u1.shape) == (M, N, S, I)

    u0_np = u0.numpy()
    u1_np = u1.numpy()
    v_np = theta["v"].numpy()
    lam_np = theta["lambda_c"].numpy()
    waste = float(tf_inputs["waste_cost"].numpy())

    # u0 = -v * 1{I==0}
    expected_u0_I0 = np.broadcast_to(-v_np[:, :, None], (M, N, S))
    np.testing.assert_allclose(u0_np[:, :, :, 0], expected_u0_I0)

    # u0 = 0 for I>0
    np.testing.assert_allclose(u0_np[:, :, :, 1:], 0.0)

    # u1 penalty at cap: -waste_cost*(1-lambda_c)*1{I==I_max}
    diff_cap = u1_np[:, :, :, I_max] - u1_np[:, :, :, I_max - 1]
    expected_pen = np.broadcast_to(-waste * (1.0 - lam_np)[:, :, None], (M, N, S))
    np.testing.assert_allclose(diff_cap, expected_pen)


def test_select_pi_by_state_gathers_correct_state(
    tiny_dims: dict[str, int],
) -> None:
    M, N, S, I_max = tiny_dims["M"], tiny_dims["N"], tiny_dims["S"], tiny_dims["I_max"]
    I = I_max + 1
    assert S == 2

    # ccp_buy[m,n,0,i]=0.1, ccp_buy[m,n,1,i]=0.9
    ccp_buy = tf.concat(
        [
            tf.fill([M, N, 1, I], tf.constant(0.1, tf.float64)),
            tf.fill([M, N, 1, I], tf.constant(0.9, tf.float64)),
        ],
        axis=2,
    )

    s_mt = tf.constant([0, 1], dtype=tf.int32)  # per-market
    pi_mnI = sp.select_pi_by_state(ccp_buy, s_mt)

    assert tuple(pi_mnI.shape) == (M, N, I)
    pi_np = pi_mnI.numpy()

    np.testing.assert_allclose(pi_np[0, :, :], 0.1)
    np.testing.assert_allclose(pi_np[1, :, :], 0.9)


def test_forward_filter_constant_ccp_half_gives_T_log_half(
    tiny_dims: dict[str, int],
    tf_inputs: dict[str, tf.Tensor],
    inventory_maps_tf,
) -> None:
    M, N, T, S, I_max = (
        tiny_dims["M"],
        tiny_dims["N"],
        tiny_dims["T"],
        tiny_dims["S"],
        tiny_dims["I_max"],
    )
    I = I_max + 1

    # Constant CCP = 0.5 for all (m,n,s,I)
    ccp_buy = tf.fill([M, N, S, I], tf.constant(0.5, tf.float64))

    # All-zero actions: likelihood each period is 0.5 regardless of belief/state.
    a_imt = tf.zeros([M, N, T], dtype=tf.int32)

    ll_mn = sp.loglik_hidden_inventory_mn(
        a_imt=a_imt,
        p_state_mt=tf_inputs["p_state_mt"],
        ccp_buy=ccp_buy,
        pi_I0=tf_inputs["pi_I0"],
        lambda_c_mn=tf.fill([M, N], tf.constant(0.3, tf.float64)),
        eps=tf_inputs["eps"],
        maps=inventory_maps_tf,
    )

    assert tuple(ll_mn.shape) == (M, N)
    expected = T * np.log(0.5)
    np.testing.assert_allclose(ll_mn.numpy(), expected, rtol=0.0, atol=1e-12)


def test_solve_ccp_buy_shape_range_and_monotone_in_price_when_price_is_absorbing(
    tiny_dims: dict[str, int],
    tf_inputs: dict[str, tf.Tensor],
    z_blocks_tf: dict[str, tf.Tensor],
    inventory_maps_tf,
) -> None:
    """
    Use an absorbing price-state transition to remove "current state predicts future states"
    effects. Then lower current price should weakly increase Pr(buy) for every inventory.
    """
    theta = sp.unconstrained_to_theta(z_blocks_tf)

    tol = tf.constant(1.0e-6, tf.float64)
    max_iter = tf.constant(80, tf.int32)

    S = int(tf_inputs["price_vals"].shape[0])
    P_absorb = tf.eye(S, dtype=tf.float64)

    ccp_buy = sp.solve_ccp_buy(
        u_m=tf_inputs["u_m"],
        price_vals=tf_inputs["price_vals"],
        P_price=P_absorb,
        theta=theta,
        waste_cost=tf_inputs["waste_cost"],
        tol=tol,
        max_iter=max_iter,
        maps=inventory_maps_tf,
    )

    M, N, I = tiny_dims["M"], tiny_dims["N"], tiny_dims["I_max"] + 1
    assert tuple(ccp_buy.shape) == (M, N, S, I)

    ccp_np = ccp_buy.numpy()
    assert np.isfinite(ccp_np).all()
    assert np.all((ccp_np > 0.0) & (ccp_np < 1.0))

    # With price_vals=[1.0, 0.8], state 1 is cheaper => buy prob should be >= for each I
    assert np.all(ccp_np[:, :, 1, :] >= ccp_np[:, :, 0, :] - 1e-12)


def test_loglik_mn_runs_end_to_end_and_is_finite(
    tiny_dims: dict[str, int],
    tf_inputs: dict[str, tf.Tensor],
    z_blocks_tf: dict[str, tf.Tensor],
    inventory_maps_tf,
) -> None:
    tol = tf.constant(1.0e-6, tf.float64)
    max_iter = tf.constant(60, tf.int32)

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
        tol=tol,
        max_iter=max_iter,
        maps=inventory_maps_tf,
    )

    M, N = tiny_dims["M"], tiny_dims["N"]
    assert tuple(ll_mn.shape) == (M, N)
    ll_np = ll_mn.numpy()
    assert np.isfinite(ll_np).all()


def test_logpost_views_combine_ll_and_prior_correctly_with_patch(
    tiny_dims: dict[str, int],
    tf_inputs: dict[str, tf.Tensor],
    z_blocks_tf: dict[str, tf.Tensor],
    sigmas: dict[str, float],
    inventory_maps_tf,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    M, N = tiny_dims["M"], tiny_dims["N"]
    sigmas_tf = _sigmas_tf(sigmas)

    ll_fake = tf.reshape(
        tf.linspace(tf.constant(-0.2, tf.float64), tf.constant(0.3, tf.float64), M * N),
        [M, N],
    )

    def _fake_loglik_mn(*args, **kwargs):
        return ll_fake

    monkeypatch.setattr(sp, "loglik_mn", _fake_loglik_mn, raising=True)

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
        sigmas=sigmas_tf,
        maps=inventory_maps_tf,
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
        sigmas=sigmas_tf,
        maps=inventory_maps_tf,
    )

    prior_us = sp.logprior_normal_m(z_blocks_tf["z_u_scale"], sigmas_tf["z_u_scale"])
    np.testing.assert_allclose(
        out_us.numpy(), (tf.reduce_sum(ll_fake, axis=1) + prior_us).numpy()
    )
