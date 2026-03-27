"""Structural model utilities for the Ching-style stockpiling sampler.

This module provides the model-side building blocks used by the posterior and
sampler layers:
- unconstrained-to-constrained parameter transforms,
- inventory-grid precomputations,
- flow-utility construction,
- Bellman fixed-point solution,
- forward simulation from solved CCPs.
"""

from __future__ import annotations

import tensorflow as tf

__all__ = [
    "InventoryMaps",
    "build_inventory_maps",
    "ccp_from_q",
    "logsumexp_q",
    "make_flow_utilities",
    "simulate_purchases_given_ccp",
    "solve_ccp_buy",
    "solve_value_function",
    "unconstrained_to_theta",
]

# (I_vals, stockout_mask, at_cap_mask, idx_down, idx_up)
InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


@tf.function(jit_compile=True)
def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Map unconstrained sampler state to constrained structural parameters.

    Any leading sample dimensions are preserved, so this transform can be used
    on a single chain state or on a retained stack of draws.
    """
    return {
        "beta": tf.sigmoid(z["z_beta"]),
        "alpha": tf.exp(z["z_alpha"]),
        "v": tf.exp(z["z_v"]),
        "fc": tf.exp(z["z_fc"]),
        "u_scale": tf.exp(z["z_u_scale"]),
    }


def build_inventory_maps(I_max: int) -> InventoryMaps:
    """Build inventory-grid masks and clipped transition indices."""
    i_vals = tf.range(I_max + 1, dtype=tf.int32)
    stockout_mask = tf.cast(tf.equal(i_vals, 0), tf.float64)
    at_cap_mask = tf.cast(tf.equal(i_vals, I_max), tf.float64)
    idx_down = tf.maximum(i_vals - 1, 0)
    idx_up = tf.minimum(i_vals + 1, I_max)
    return i_vals, stockout_mask, at_cap_mask, idx_down, idx_up


@tf.function(jit_compile=True)
def make_flow_utilities(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Construct no-buy and buy flow utilities on the full state grid.

    Returns:
      u0: (M, N, J, S, I) no-buy flow utility.
      u1: (M, N, J, S, I) buy flow utility.
    """
    _, stockout_mask, at_cap_mask, _, _ = maps

    alpha_j = theta["alpha"]
    v_j = theta["v"]
    fc_j = theta["fc"]
    u_scale_m = theta["u_scale"]

    stockout_1111i = stockout_mask[None, None, None, None, :]
    at_cap_1111i = at_cap_mask[None, None, None, None, :]

    u_eff_m1j11 = (u_scale_m[:, None] * u_mj)[:, None, :, None, None]
    alpha_11j11 = alpha_j[None, None, :, None, None]
    fc_11j11 = fc_j[None, None, :, None, None]
    price_m1js1 = price_vals_mj[:, None, :, :, None]

    base_buy_m1js1 = u_eff_m1j11 - alpha_11j11 * price_m1js1 - fc_11j11
    waste_term_mn111i = (
        waste_cost * (1.0 - lambda_mn)[:, :, None, None, None] * at_cap_1111i
    )
    u1 = base_buy_m1js1 - waste_term_mn111i

    u0_11j1i = -v_j[None, None, :, None, None] * stockout_1111i
    u0 = tf.broadcast_to(u0_11j1i, tf.shape(u1))
    return u0, u1


@tf.function(jit_compile=True)
def expected_over_next_price(V: tf.Tensor, P_price_mj: tf.Tensor) -> tf.Tensor:
    """Average continuation values over next-period price states."""
    return tf.einsum("mjsr,mnjri->mnjsi", P_price_mj, V)


@tf.function(jit_compile=True)
def expected_over_next_inv_no_buy(
    cont_s: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
) -> tf.Tensor:
    """Average continuation values over next inventory conditional on no-buy."""
    cont_c0 = cont_s
    cont_c1 = tf.gather(cont_s, idx_down, axis=4)
    return p_c0_mn111 * cont_c0 + p_c1_mn111 * cont_c1


@tf.function(jit_compile=True)
def expected_over_next_inv_buy(
    cont_s: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_up: tf.Tensor,
) -> tf.Tensor:
    """Average continuation values over next inventory conditional on buy."""
    cont_c0 = tf.gather(cont_s, idx_up, axis=4)
    cont_c1 = cont_s
    return p_c0_mn111 * cont_c0 + p_c1_mn111 * cont_c1


@tf.function(jit_compile=True)
def logsumexp_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """Compute log(exp(q0) + exp(q1)) elementwise."""
    return q0 + tf.nn.softplus(q1 - q0)


@tf.function(jit_compile=True)
def ccp_from_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """Return the buy probability under the binary logit decision rule."""
    return tf.sigmoid(q1 - q0)


@tf.function(jit_compile=True)
def bellman_update(
    V: tf.Tensor,
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta: tf.Tensor,
    P_price_mj: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Apply one Bellman operator step and return V_new, q0, and q1."""
    cont_s = expected_over_next_price(V, P_price_mj)
    cont0 = expected_over_next_inv_no_buy(cont_s, p_c0_mn111, p_c1_mn111, idx_down)
    cont1 = expected_over_next_inv_buy(cont_s, p_c0_mn111, p_c1_mn111, idx_up)

    q0 = u0 + tf.reshape(beta, (1, 1, 1, 1, 1)) * cont0
    q1 = u1 + tf.reshape(beta, (1, 1, 1, 1, 1)) * cont1
    V_new = logsumexp_q(q0, q1)
    return V_new, q0, q1


@tf.function(jit_compile=True)
def solve_value_function(
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta: tf.Tensor,
    P_price_mj: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    tol: float,
    max_iter: int,
) -> tf.Tensor:
    """Solve the Bellman fixed point by value-function iteration."""
    V0 = tf.zeros_like(u0)
    max_diff0 = tf.constant(float("inf"), dtype=tf.float64)

    def cond_fn(_: tf.Tensor, max_diff: tf.Tensor) -> tf.Tensor:
        return max_diff > tol

    def body_fn(V_prev: tf.Tensor, _: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        V_new, _, _ = bellman_update(
            V=V_prev,
            u0=u0,
            u1=u1,
            beta=beta,
            P_price_mj=P_price_mj,
            p_c0_mn111=p_c0_mn111,
            p_c1_mn111=p_c1_mn111,
            idx_down=idx_down,
            idx_up=idx_up,
        )
        max_diff = tf.reduce_max(tf.abs(V_new - V_prev))
        return V_new, max_diff

    V_final, _ = tf.while_loop(
        cond=cond_fn,
        body=body_fn,
        loop_vars=(V0, max_diff0),
        maximum_iterations=max_iter,
    )
    return V_final


@tf.function(jit_compile=True)
def solve_ccp_buy(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
    tol: float,
    max_iter: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Solve the buy CCPs and associated choice-specific values."""
    _, _, _, idx_down, idx_up = maps

    p_c1_mn111 = lambda_mn[:, :, None, None, None]
    p_c0_mn111 = (1.0 - lambda_mn)[:, :, None, None, None]
    beta = tf.reshape(theta["beta"], ())

    u0, u1 = make_flow_utilities(
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        theta=theta,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        maps=maps,
    )

    V_final = solve_value_function(
        u0=u0,
        u1=u1,
        beta=beta,
        P_price_mj=P_price_mj,
        p_c0_mn111=p_c0_mn111,
        p_c1_mn111=p_c1_mn111,
        idx_down=idx_down,
        idx_up=idx_up,
        tol=tol,
        max_iter=max_iter,
    )

    _, q0, q1 = bellman_update(
        V=V_final,
        u0=u0,
        u1=u1,
        beta=beta,
        P_price_mj=P_price_mj,
        p_c0_mn111=p_c0_mn111,
        p_c1_mn111=p_c1_mn111,
        idx_down=idx_down,
        idx_up=idx_up,
    )
    ccp_buy = ccp_from_q(q0, q1)
    return ccp_buy, q0, q1


@tf.function(jit_compile=True)
def simulate_purchases_given_ccp(
    ccp_buy: tf.Tensor,
    s_mjt: tf.Tensor,
    lambda_mn: tf.Tensor,
    pi_I0: tf.Tensor,
    I_max: int,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Simulate purchases, consumption, and inventories using stateless RNG.

    Args:
      ccp_buy: (M, N, J, S, I) buy probabilities.
      s_mjt: (M, J, T) observed price-state sequence.
      lambda_mn: (M, N) consumption probabilities.
      pi_I0: (I,) initial inventory distribution.
      I_max: Maximum inventory level.
      seed: (2,) stateless RNG seed.

    Returns:
      a_mnjt: (M, N, J, T) purchases.
      c_mnjt: (M, N, J, T) consumption.
      I_mnjt: (M, N, J, T + 1) inventory path.
    """
    M = tf.shape(ccp_buy)[0]
    N = tf.shape(ccp_buy)[1]
    J = tf.shape(ccp_buy)[2]
    T = tf.shape(s_mjt)[2]

    init_seed, loop_seed = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=2),
        axis=0,
    )

    cdf = tf.cumsum(pi_I0)
    u_init = tf.random.stateless_uniform((M, N, J), seed=init_seed, dtype=tf.float64)
    I_init = tf.reduce_sum(
        tf.cast(u_init[..., None] > cdf[None, None, None, :], tf.int32),
        axis=-1,
    )

    I_path = tf.TensorArray(dtype=tf.int32, size=T + 1, clear_after_read=False)
    a_path = tf.TensorArray(dtype=tf.int32, size=T, clear_after_read=False)
    c_path = tf.TensorArray(dtype=tf.int32, size=T, clear_after_read=False)
    I_path = I_path.write(0, I_init)

    def cond(
        t: tf.Tensor,
        _: tf.Tensor,
        __: tf.TensorArray,
        ___: tf.TensorArray,
        ____: tf.TensorArray,
        _____: tf.Tensor,
    ) -> tf.Tensor:
        return t < T

    def body(
        t: tf.Tensor,
        I_curr: tf.Tensor,
        I_acc: tf.TensorArray,
        a_acc: tf.TensorArray,
        c_acc: tf.TensorArray,
        seed_curr: tf.Tensor,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.TensorArray,
        tf.TensorArray,
        tf.TensorArray,
        tf.Tensor,
    ]:
        seeds = tf.random.experimental.stateless_split(seed_curr, num=3)
        next_seed = seeds[0]
        buy_seed = seeds[1]
        cons_seed = seeds[2]

        s_mj = s_mjt[:, :, t]
        s_grid = tf.broadcast_to(s_mj[:, None, :], (M, N, J))

        p_by_s = tf.gather(ccp_buy, s_grid, axis=3, batch_dims=3)
        p_buy = tf.gather(p_by_s, I_curr, axis=3, batch_dims=3)

        u_buy = tf.random.stateless_uniform((M, N, J), seed=buy_seed, dtype=tf.float64)
        a_t = tf.cast(u_buy < p_buy, tf.int32)

        u_cons = tf.random.stateless_uniform(
            (M, N, J),
            seed=cons_seed,
            dtype=tf.float64,
        )
        c_t = tf.cast(u_cons < lambda_mn[:, :, None], tf.int32)

        I_next = tf.clip_by_value(I_curr + a_t - c_t, 0, I_max)

        a_acc = a_acc.write(t, a_t)
        c_acc = c_acc.write(t, c_t)
        I_acc = I_acc.write(t + 1, I_next)
        return t + 1, I_next, I_acc, a_acc, c_acc, next_seed

    _, _, I_path, a_path, c_path, _ = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            I_init,
            I_path,
            a_path,
            c_path,
            loop_seed,
        ),
    )

    a_mnjt = tf.transpose(a_path.stack(), perm=[1, 2, 3, 0])
    c_mnjt = tf.transpose(c_path.stack(), perm=[1, 2, 3, 0])
    I_mnjt = tf.transpose(I_path.stack(), perm=[1, 2, 3, 0])
    return a_mnjt, c_mnjt, I_mnjt
