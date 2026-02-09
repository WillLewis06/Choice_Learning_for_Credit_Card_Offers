# ching/stockpiling_posterior.py
#
# Pure TensorFlow posterior components for a minimal Ching-style stockpiling model
# with latent inventory and seller-observed (purchases, prices) only.
#
# Updated design for batched RW-MH (rw_mh_step):
# - Consumers are distinct within each market: parameters are consumer-specific:
#     beta, alpha, v, fc, lambda_c  have shape (M, N)
# - u_scale is market-specific (shared across consumers within market):
#     u_scale has shape (M,)
#
# To support elementwise acceptance in rw_mh_step:
# - Likelihood is exposed as per-consumer contributions loglik_mn with shape (M, N).
# - Priors are exposed per-block with shapes matching the updated parameter block:
#     consumer-block priors: (M, N)
#     u_scale prior:         (M,)
# - Separate log-density "views" are provided (no Python branching):
#     logpost_z_beta_mn(...):   (M, N)
#     logpost_z_alpha_mn(...):  (M, N)
#     logpost_z_v_mn(...):      (M, N)
#     logpost_z_fc_mn(...):     (M, N)
#     logpost_z_lambda_mn(...): (M, N)
#     logpost_u_scale_m(...):   (M,)
#
# Inventory maps are precomputed outside posterior evaluation and passed in:
#   maps = (D_down, D_up, stockout_mask, at_cap_mask)

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

InventoryMaps = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Map unconstrained parameters z[*] to constrained theta[*].

    Required z keys and shapes:
      z_beta    (M,N) -> beta      in (0,1)
      z_alpha   (M,N) -> alpha     > 0
      z_v       (M,N) -> v         > 0
      z_fc      (M,N) -> fc        > 0
      z_lambda  (M,N) -> lambda_c  in (0,1)
      z_u_scale (M,)  -> u_scale   > 0 (shared across consumers within each market)

    Returned theta keys:
      beta, alpha, v, fc, lambda_c, u_scale
    """
    return {
        "beta": tf.math.sigmoid(z["z_beta"]),
        "alpha": tf.exp(z["z_alpha"]),
        "v": tf.exp(z["z_v"]),
        "fc": tf.exp(z["z_fc"]),
        "lambda_c": tf.math.sigmoid(z["z_lambda"]),
        "u_scale": tf.exp(z["z_u_scale"]),
    }


# =============================================================================
# Inventory maps (precompute once per run)
# =============================================================================


def build_inventory_maps(I_max: tf.Tensor) -> InventoryMaps:
    """
    Build deterministic inventory mapping matrices and masks.

    Inventory states: I in {0,...,I_max}, size I = I_max+1.

    Deterministic mappings:
      down(i) = max(i-1, 0)
      up(i)   = min(i+1, I_max)

    Returns:
      D_down: (I, I) float64 with D_down[i, down(i)] = 1
      D_up:   (I, I) float64 with D_up[i,   up(i)]  = 1
      stockout_mask: (I,) float64 indicator 1{I==0}
      at_cap_mask:   (I,) float64 indicator 1{I==I_max}
    """
    I_max_i = tf.cast(I_max, tf.int32)
    I = I_max_i + 1

    i = tf.range(I, dtype=tf.int32)  # (I,)
    col_down = tf.maximum(i - 1, 0)
    col_up = tf.minimum(i + 1, I_max_i)

    idx_down = tf.stack([i, col_down], axis=1)  # (I,2)
    idx_up = tf.stack([i, col_up], axis=1)  # (I,2)

    updates = tf.ones((I,), dtype=tf.float64)
    shape_II = tf.stack([I, I])

    D_down = tf.scatter_nd(idx_down, updates, shape_II)
    D_up = tf.scatter_nd(idx_up, updates, shape_II)

    stockout_mask = tf.cast(tf.equal(i, 0), tf.float64)  # (I,)
    at_cap_mask = tf.cast(tf.equal(i, I_max_i), tf.float64)  # (I,)

    return D_down, D_up, stockout_mask, at_cap_mask


def _unpack_maps(
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    D_down, D_up, stockout_mask, at_cap_mask = maps
    return D_down, D_up, stockout_mask, at_cap_mask


# =============================================================================
# DP / CCP computation (batched over markets and consumers)
# =============================================================================


def _logsumexp2(x0: tf.Tensor, x1: tf.Tensor) -> tf.Tensor:
    m = tf.maximum(x0, x1)
    return m + tf.math.log(tf.exp(x0 - m) + tf.exp(x1 - m))


def make_flow_utilities(
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    theta: dict[str, tf.Tensor],
    waste_cost: tf.Tensor,
    stockout_mask: tf.Tensor,
    at_cap_mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Flow utilities on (m,n,s,I):

      u1(m,n,s,I) = u_scale[m] * u_m[m] - alpha(m,n) * price_vals[s] - fc(m,n)
                   - waste_cost * (1 - lambda_c(m,n)) * 1{I==I_max}

      u0(m,n,s,I) = -v(m,n) * 1{I==0}

    Outputs:
      u0, u1: (M,N,S,I)
    """
    I = tf.shape(stockout_mask)[0]  # = I_max + 1
    M = tf.shape(u_m)[0]
    N = tf.shape(theta["beta"])[1]
    S = tf.shape(price_vals)[0]
    shape_mnsi = tf.stack([M, N, S, I])

    u_m_scaled_mn11 = (theta["u_scale"] * u_m)[:, None, None, None]  # (M,1,1,1)
    alpha_mn11 = theta["alpha"][:, :, None, None]  # (M,N,1,1)
    fc_mn11 = theta["fc"][:, :, None, None]  # (M,N,1,1)
    price_11s1 = price_vals[None, None, :, None]  # (1,1,S,1)

    u1_base = u_m_scaled_mn11 - alpha_mn11 * price_11s1 - fc_mn11  # (M,N,S,1)
    u1 = tf.broadcast_to(u1_base, shape_mnsi)  # (M,N,S,I)

    one = tf.constant(1.0, dtype=tf.float64)
    p_c0_mn11 = (one - theta["lambda_c"])[:, :, None, None]  # (M,N,1,1)
    u1 = u1 - waste_cost * p_c0_mn11 * at_cap_mask[None, None, None, :]  # (M,N,S,I)

    u0 = tf.broadcast_to(
        -theta["v"][:, :, None, None] * stockout_mask[None, None, None, :],
        shape_mnsi,
    )
    return u0, u1


def expected_over_next_price(V: tf.Tensor, P_price: tf.Tensor) -> tf.Tensor:
    """
    EV_next[m,n,s,I] = sum_{s'} P_price[s,s'] * V[m,n,s',I]
    """
    return tf.einsum("ab,mnbi->mnai", P_price, V)


def bellman_update(
    V: tf.Tensor,
    u0: tf.Tensor,
    u1: tf.Tensor,
    theta: dict[str, tf.Tensor],
    P_price: tf.Tensor,
    D_down: tf.Tensor,
    D_up: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One Bellman update.

    Returns:
      V_new, Q0, Q1: all (M,N,S,I)
    """
    EV_next = expected_over_next_price(V, P_price)  # (M,N,S,I)

    beta_mn11 = theta["beta"][:, :, None, None]
    one = tf.constant(1.0, dtype=tf.float64)
    p_c0_mn11 = (one - theta["lambda_c"])[:, :, None, None]
    p_c1_mn11 = theta["lambda_c"][:, :, None, None]

    EV_down = tf.einsum("mnsj,ij->mnsi", EV_next, D_down)
    EV_up = tf.einsum("mnsj,ij->mnsi", EV_next, D_up)

    cont0 = p_c0_mn11 * EV_next + p_c1_mn11 * EV_down
    cont1 = p_c0_mn11 * EV_up + p_c1_mn11 * EV_next

    Q0 = u0 + beta_mn11 * cont0
    Q1 = u1 + beta_mn11 * cont1
    V_new = _logsumexp2(Q0, Q1)
    return V_new, Q0, Q1


def solve_value_function(
    u0: tf.Tensor,
    u1: tf.Tensor,
    theta: dict[str, tf.Tensor],
    P_price: tf.Tensor,
    D_down: tf.Tensor,
    D_up: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
) -> tf.Tensor:
    """
    Value iteration.

    Returns:
      V_final: (M,N,S,I)
    """
    max_iter_i = tf.cast(max_iter, tf.int32)

    V0 = tf.zeros_like(u0)
    diff0 = tf.constant(1.0e30, dtype=tf.float64)

    def cond(it: tf.Tensor, V_curr: tf.Tensor, diff_curr: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(it < max_iter_i, diff_curr > tol)

    def body(it: tf.Tensor, V_curr: tf.Tensor, diff_curr: tf.Tensor):
        V_new, _, _ = bellman_update(V_curr, u0, u1, theta, P_price, D_down, D_up)
        diff_new = tf.reduce_max(tf.abs(V_new - V_curr))
        return it + 1, V_new, diff_new

    it0 = tf.constant(0, tf.int32)
    _, V_final, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[it0, V0, diff0],
        shape_invariants=[
            it0.get_shape(),
            tf.TensorShape([None, None, None, None]),
            diff0.get_shape(),
        ],
    )
    return V_final


def solve_ccp_buy(
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    theta: dict[str, tf.Tensor],
    waste_cost: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Solve DP and return buy CCP table.

    Returns:
      ccp_buy: (M,N,S,I) with ccp_buy[m,n,s,i] = P(a=1 | m,n,s,i)
    """
    D_down, D_up, stockout_mask, at_cap_mask = _unpack_maps(maps)

    u0, u1 = make_flow_utilities(
        u_m=u_m,
        price_vals=price_vals,
        theta=theta,
        waste_cost=waste_cost,
        stockout_mask=stockout_mask,
        at_cap_mask=at_cap_mask,
    )
    V_final = solve_value_function(u0, u1, theta, P_price, D_down, D_up, tol, max_iter)
    _, Q0, Q1 = bellman_update(V_final, u0, u1, theta, P_price, D_down, D_up)
    denom = _logsumexp2(Q0, Q1)
    return tf.exp(Q1 - denom)


# =============================================================================
# Forward filtering likelihood (per-consumer contributions)
# =============================================================================


def initial_inventory_belief(
    pi_I0: tf.Tensor, M: tf.Tensor, N: tf.Tensor, eps: tf.Tensor
) -> tf.Tensor:
    """
    Initial belief b0 over inventory for each (m,n).

    Returns:
      b0: (M,N,I) normalized.
    """
    I = tf.shape(pi_I0)[0]
    shape_mni = tf.stack(
        [tf.cast(M, tf.int32), tf.cast(N, tf.int32), tf.cast(I, tf.int32)]
    )

    b0 = tf.broadcast_to(pi_I0[None, None, :], shape_mni)
    denom = tf.reduce_sum(b0, axis=2, keepdims=True)
    return b0 / tf.maximum(denom, eps)


def select_pi_by_state(ccp_buy: tf.Tensor, s_mt: tf.Tensor) -> tf.Tensor:
    """
    Select CCPs at the observed price state.

    Inputs:
      ccp_buy: (M,N,S,I)
      s_mt:    (M,) int state per market at time t

    Returns:
      pi_mnI: (M,N,I)
    """
    M = tf.shape(ccp_buy)[0]
    N = tf.shape(ccp_buy)[1]
    s_mn = tf.broadcast_to(tf.cast(s_mt, tf.int32)[:, None], tf.stack([M, N]))
    return tf.gather(ccp_buy, s_mn, axis=2, batch_dims=2)


def action_likelihood_by_inventory(pi_mnI: tf.Tensor, a_mn: tf.Tensor) -> tf.Tensor:
    """
    Returns:
      lik_I: (M,N,I) = P(a_mn | I)
    """
    a = tf.cast(a_mn, tf.float64)
    one = tf.constant(1.0, dtype=tf.float64)
    return a[:, :, None] * pi_mnI + (one - a)[:, :, None] * (one - pi_mnI)


def bayes_update_belief_mn(
    b: tf.Tensor, lik_I: tf.Tensor, eps: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Returns:
      w_post:     (M,N,I) unnormalized weights proportional to posterior over I_t
      ll_step_mn: (M,N)   log predictive probability for each (m,n)
    """
    w_post = b * lik_I  # (M,N,I)
    pred = tf.reduce_sum(w_post, axis=2)  # (M,N)
    pred_clip = tf.maximum(pred, eps)
    ll_step_mn = tf.math.log(pred_clip)  # (M,N)
    return w_post, ll_step_mn


def transition_belief(
    w_post: tf.Tensor,
    a_mn: tf.Tensor,
    lambda_c_mn: tf.Tensor,
    D_down: tf.Tensor,
    D_up: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Propagate belief one step forward:
      I' = clip(I + a - c, 0, I_max),  c ~ Bernoulli(lambda_c_mn)

    Returns:
      b_next: (M,N,I) normalized.
    """
    a = tf.cast(a_mn, tf.float64)
    one = tf.constant(1.0, dtype=tf.float64)

    p_c0_mn1 = (one - lambda_c_mn)[:, :, None]
    p_c1_mn1 = lambda_c_mn[:, :, None]

    w_down = tf.einsum("mni,ij->mnj", w_post, D_down)
    w_up = tf.einsum("mni,ij->mnj", w_post, D_up)

    w_next0 = p_c0_mn1 * w_post + p_c1_mn1 * w_down
    w_next1 = p_c0_mn1 * w_up + p_c1_mn1 * w_post
    w_next = (one - a)[:, :, None] * w_next0 + a[:, :, None] * w_next1

    denom = tf.reduce_sum(w_next, axis=2, keepdims=True)
    return w_next / tf.maximum(denom, eps)


def loglik_hidden_inventory_mn(
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    ccp_buy: tf.Tensor,
    pi_I0: tf.Tensor,
    lambda_c_mn: tf.Tensor,
    eps: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Forward filter integrating out latent inventory.

    Returns:
      loglik_mn: (M,N) where each entry is sum_t log P(a_{m,n,t} | history)
    """
    D_down, D_up, _, _ = _unpack_maps(maps)

    M = tf.shape(a_imt)[0]
    N = tf.shape(a_imt)[1]
    T = tf.shape(a_imt)[2]

    b0 = initial_inventory_belief(pi_I0, M, N, eps)
    ll0 = tf.zeros(tf.stack([M, N]), dtype=tf.float64)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor):
        s_mt = p_state_mt[:, t]  # (M,)
        a_mn = a_imt[:, :, t]  # (M,N)

        pi_mnI = select_pi_by_state(ccp_buy, s_mt)  # (M,N,I)
        lik_I = action_likelihood_by_inventory(pi_mnI, a_mn)

        w_post, ll_step_mn = bayes_update_belief_mn(b_curr, lik_I, eps)
        b_next = transition_belief(w_post, a_mn, lambda_c_mn, D_down, D_up, eps)

        return t + 1, b_next, ll_curr + ll_step_mn

    t0 = tf.constant(0, tf.int32)
    _, _, ll_final = tf.while_loop(
        cond,
        body,
        loop_vars=[t0, b0, ll0],
        shape_invariants=[
            t0.get_shape(),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None]),
        ],
    )
    return ll_final


# =============================================================================
# Priors (shapes match the updated MH blocks)
# =============================================================================


def logprior_normal_mn(z_block_mn: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Elementwise Normal(0, sigma^2) log prior (up to constants), shape (M,N)."""
    return -0.5 * tf.square(z_block_mn / sigma)


def logprior_normal_m(z_block_m: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Elementwise Normal(0, sigma^2) log prior (up to constants), shape (M,)."""
    return -0.5 * tf.square(z_block_m / sigma)


def logprior_consumer_mn(
    z: dict[str, tf.Tensor], sigmas: dict[str, tf.Tensor]
) -> tf.Tensor:
    """Sum of consumer-block priors, returned per (m,n), shape (M,N)."""
    return (
        logprior_normal_mn(z["z_beta"], sigmas["z_beta"])
        + logprior_normal_mn(z["z_alpha"], sigmas["z_alpha"])
        + logprior_normal_mn(z["z_v"], sigmas["z_v"])
        + logprior_normal_mn(z["z_fc"], sigmas["z_fc"])
        + logprior_normal_mn(z["z_lambda"], sigmas["z_lambda"])
    )


def logprior_u_scale_m(
    z: dict[str, tf.Tensor], sigmas: dict[str, tf.Tensor]
) -> tf.Tensor:
    """Prior for u_scale block, shape (M,)."""
    return logprior_normal_m(z["z_u_scale"], sigmas["z_u_scale"])


# =============================================================================
# Likelihood and log-density views for rw_mh_step
# =============================================================================


def loglik_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Per-consumer log-likelihood contributions, shape (M,N).

    Computes CCPs via DP under theta(z), then runs a forward filter per consumer.
    """
    theta = unconstrained_to_theta(z)
    ccp_buy = solve_ccp_buy(
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        theta=theta,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    return loglik_hidden_inventory_mn(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        ccp_buy=ccp_buy,
        pi_I0=pi_I0,
        lambda_c_mn=theta["lambda_c"],
        eps=eps,
        maps=maps,
    )


def logpost_z_beta_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_beta."""
    ll = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp = logprior_normal_mn(z["z_beta"], sigmas["z_beta"])
    return ll + lp


def logpost_z_alpha_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_alpha."""
    ll = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp = logprior_normal_mn(z["z_alpha"], sigmas["z_alpha"])
    return ll + lp


def logpost_z_v_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_v."""
    ll = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp = logprior_normal_mn(z["z_v"], sigmas["z_v"])
    return ll + lp


def logpost_z_fc_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_fc."""
    ll = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp = logprior_normal_mn(z["z_fc"], sigmas["z_fc"])
    return ll + lp


def logpost_z_lambda_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_lambda."""
    ll = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp = logprior_normal_mn(z["z_lambda"], sigmas["z_lambda"])
    return ll + lp


def logpost_u_scale_m(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    (M,) log posterior view for updating z_u_scale.

    Returns:
      sum_n loglik_mn(z)[m,n] + logprior_u_scale_m(z)[m]
    """
    ll_mn = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    ll_m = tf.reduce_sum(ll_mn, axis=1)  # (M,)
    lp_m = logprior_u_scale_m(z, sigmas)  # (M,)
    return ll_m + lp_m


def logpost(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigmas: dict[str, tf.Tensor],
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Scalar joint log posterior for debugging/validation:

      sum_{m,n} loglik_mn(z)[m,n]
      + sum_{m,n} logprior_consumer_mn(z)[m,n]
      + sum_m     logprior_u_scale_m(z)[m]
    """
    ll_mn = loglik_mn(
        z,
        a_imt,
        p_state_mt,
        u_m,
        price_vals,
        P_price,
        pi_I0,
        waste_cost,
        eps,
        tol,
        max_iter,
        maps,
    )
    lp_consumer_mn = logprior_consumer_mn(z, sigmas)
    lp_u_m = logprior_u_scale_m(z, sigmas)
    return tf.reduce_sum(ll_mn) + tf.reduce_sum(lp_consumer_mn) + tf.reduce_sum(lp_u_m)
