"""
Pure TensorFlow core mechanics for a minimal Ching-style stockpiling model
with latent inventory and seller-observed (purchases, prices) only.

This module contains:
- Parameter transforms (z -> theta) and consumption mixture weights
- Inventory state maps and deterministic pushforward operators
- Dynamic program (value iteration) and buy CCP table construction
- Forward filter integrating out latent inventory to compute log-likelihood contributions

It intentionally excludes:
- Priors
- Log-posterior "views" for RW-MH block updates
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

InventoryMaps = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

one_f64 = tf.constant(1.0, dtype=tf.float64)


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
      z_u_scale (M,)  -> u_scale   > 0 (shared within each market)

    Returns:
      dict[str, tf.Tensor]: constrained parameters with the same shapes.
    """
    return {
        "beta": tf.math.sigmoid(z["z_beta"]),
        "alpha": tf.exp(z["z_alpha"]),
        "v": tf.exp(z["z_v"]),
        "fc": tf.exp(z["z_fc"]),
        "lambda_c": tf.math.sigmoid(z["z_lambda"]),
        "u_scale": tf.exp(z["z_u_scale"]),
    }


def _consumption_probs(lambda_c_mn: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Consumption mixture weights.

    Args:
      lambda_c_mn: (M,N) probability of consumption in a period.

    Returns:
      (p_c0, p_c1) with shapes (M,N), where p_c0 = 1-lambda_c, p_c1 = lambda_c.
    """
    p_c1 = lambda_c_mn
    p_c0 = one_f64 - lambda_c_mn
    return p_c0, p_c1


# =============================================================================
# Inventory maps (precompute once per run)
# =============================================================================


def build_inventory_maps(I_max: tf.Tensor) -> InventoryMaps:
    """
    Build deterministic inventory mapping index vectors and masks.

    Inventory states: I in {0,...,I_max}, size I = I_max+1.

    Deterministic mappings:
      down(i) = max(i-1, 0)
      up(i)   = min(i+1, I_max)

    Returns:
      idx_down: (I,) int32 with idx_down[i] = down(i)
      idx_up:   (I,) int32 with idx_up[i]   = up(i)
      stockout_mask: (I,) float64 indicator 1{I==0}
      at_cap_mask:   (I,) float64 indicator 1{I==I_max}
    """
    I_max_i = tf.cast(I_max, tf.int32)
    I = I_max_i + 1

    i = tf.range(I, dtype=tf.int32)
    idx_down = tf.maximum(i - 1, 0)
    idx_up = tf.minimum(i + 1, I_max_i)

    stockout_mask = tf.cast(tf.equal(i, 0), tf.float64)
    at_cap_mask = tf.cast(tf.equal(i, I_max_i), tf.float64)

    return idx_down, idx_up, stockout_mask, at_cap_mask


def _unpack_maps(
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Unpack inventory maps tuple.

    Returns:
      (idx_down, idx_up, stockout_mask, at_cap_mask)
    """
    idx_down, idx_up, stockout_mask, at_cap_mask = maps
    return idx_down, idx_up, stockout_mask, at_cap_mask


def _pushforward_inventory_mass(w_mni: tf.Tensor, idx_next_i: tf.Tensor) -> tf.Tensor:
    """
    Push mass forward under a deterministic mapping i -> idx_next_i[i].

    This implements:
      w_next[..., j] = sum_{i : idx_next_i[i] = j} w[..., i]
    using unsorted_segment_sum.

    Inputs:
      w_mni:      (M,N,I)
      idx_next_i: (I,) int32 mapping each current inventory i to next index

    Returns:
      w_next_mni: (M,N,I)
    """
    I = tf.shape(w_mni)[2]

    w_imn = tf.transpose(w_mni, perm=[2, 0, 1])  # (I,M,N)
    w_iK = tf.reshape(w_imn, tf.stack([I, -1]))  # (I, M*N)

    w_next_iK = tf.math.unsorted_segment_sum(
        w_iK, idx_next_i, num_segments=I
    )  # (I, M*N)

    w_next_imn = tf.reshape(
        w_next_iK, tf.stack([I, tf.shape(w_mni)[0], tf.shape(w_mni)[1]])
    )  # (I,M,N)
    return tf.transpose(w_next_imn, perm=[1, 2, 0])  # (M,N,I)


# =============================================================================
# DP / CCP computation (batched over markets and consumers)
# =============================================================================


def _logsumexp2(x0: tf.Tensor, x1: tf.Tensor) -> tf.Tensor:
    """
    Stable elementwise log(exp(x0) + exp(x1)).

    Args:
      x0, x1: broadcastable tensors

    Returns:
      tensor: elementwise log-sum-exp.
    """
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
    I = tf.shape(stockout_mask)[0]
    M = tf.shape(u_m)[0]
    N = tf.shape(theta["beta"])[1]
    S = tf.shape(price_vals)[0]
    shape_mnsi = tf.stack([M, N, S, I])

    u_m_scaled_mn11 = (theta["u_scale"] * u_m)[:, None, None, None]  # (M,1,1,1)
    alpha_mn11 = theta["alpha"][:, :, None, None]  # (M,N,1,1)
    fc_mn11 = theta["fc"][:, :, None, None]  # (M,N,1,1)
    price_11s1 = price_vals[None, None, :, None]  # (1,1,S,1)

    u1_base_mns1 = u_m_scaled_mn11 - alpha_mn11 * price_11s1 - fc_mn11  # (M,N,S,1)
    u1 = tf.broadcast_to(u1_base_mns1, shape_mnsi)  # (M,N,S,I)

    p_c0_mn11, _ = _consumption_probs(theta["lambda_c"])
    p_c0_mn11 = p_c0_mn11[:, :, None, None]  # (M,N,1,1)
    u1 = u1 - waste_cost * p_c0_mn11 * at_cap_mask[None, None, None, :]

    u0_mn1i = -theta["v"][:, :, None, None] * stockout_mask[None, None, None, :]
    u0 = tf.broadcast_to(u0_mn1i, shape_mnsi)
    return u0, u1


def expected_over_next_price(V: tf.Tensor, P_price: tf.Tensor) -> tf.Tensor:
    """
    EV_next[m,n,s,I] = sum_{s'} P_price[s,s'] * V[m,n,s',I].

    Inputs:
      V:       (M,N,S,I)
      P_price: (S,S)

    Returns:
      (M,N,S,I)
    """
    return tf.einsum("ab,mnbi->mnai", P_price, V)


def bellman_update(
    V: tf.Tensor,
    u0: tf.Tensor,
    u1: tf.Tensor,
    theta: dict[str, tf.Tensor],
    P_price: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One Bellman update for logit shocks.

    Consumption mixture:
      - if a=0: next inventory is i (c=0) or down(i) (c=1)
      - if a=1: next inventory is up(i) (c=0) or i (c=1)

    Returns:
      (v_new, q0, q1): all (M,N,S,I)
    """
    ev_next = expected_over_next_price(V, P_price)  # (M,N,S,I)

    beta_mn11 = theta["beta"][:, :, None, None]
    p_c0_mn11, p_c1_mn11 = _consumption_probs(theta["lambda_c"])
    p_c0_mn11 = p_c0_mn11[:, :, None, None]
    p_c1_mn11 = p_c1_mn11[:, :, None, None]

    ev_down = tf.gather(ev_next, idx_down, axis=3)  # (M,N,S,I)
    ev_up = tf.gather(ev_next, idx_up, axis=3)  # (M,N,S,I)

    cont0 = p_c0_mn11 * ev_next + p_c1_mn11 * ev_down
    cont1 = p_c0_mn11 * ev_up + p_c1_mn11 * ev_next

    q0 = u0 + beta_mn11 * cont0
    q1 = u1 + beta_mn11 * cont1
    v_new = _logsumexp2(q0, q1)
    return v_new, q0, q1


def solve_value_function(
    u0: tf.Tensor,
    u1: tf.Tensor,
    theta: dict[str, tf.Tensor],
    P_price: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
) -> tf.Tensor:
    """
    Value iteration until max-norm change <= tol or max_iter reached.

    Inputs:
      u0,u1: (M,N,S,I)

    Returns:
      v_final: (M,N,S,I)
    """
    v0 = tf.zeros_like(u0)
    diff0 = tf.constant(1.0e30, dtype=tf.float64)

    def cond(it: tf.Tensor, v_curr: tf.Tensor, diff_curr: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(it < max_iter, diff_curr > tol)

    def body(it: tf.Tensor, v_curr: tf.Tensor, diff_curr: tf.Tensor):
        v_new, _, _ = bellman_update(v_curr, u0, u1, theta, P_price, idx_down, idx_up)
        diff_new = tf.reduce_max(tf.abs(v_new - v_curr))
        return it + 1, v_new, diff_new

    it0 = tf.constant(0, tf.int32)
    _, v_final, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[it0, v0, diff0],
        shape_invariants=[
            it0.get_shape(),
            tf.TensorShape([None, None, None, None]),
            diff0.get_shape(),
        ],
    )
    return v_final


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
    idx_down, idx_up, stockout_mask, at_cap_mask = _unpack_maps(maps)

    u0, u1 = make_flow_utilities(
        u_m=u_m,
        price_vals=price_vals,
        theta=theta,
        waste_cost=waste_cost,
        stockout_mask=stockout_mask,
        at_cap_mask=at_cap_mask,
    )

    v_final = solve_value_function(
        u0, u1, theta, P_price, idx_down, idx_up, tol, max_iter
    )
    _, q0, q1 = bellman_update(v_final, u0, u1, theta, P_price, idx_down, idx_up)
    denom = _logsumexp2(q0, q1)
    return tf.exp(q1 - denom)


# =============================================================================
# Forward filtering likelihood (per-consumer contributions)
# =============================================================================


def initial_inventory_belief(
    pi_I0: tf.Tensor, M: tf.Tensor, N: tf.Tensor, eps: tf.Tensor
) -> tf.Tensor:
    """
    Initial inventory belief b0 for each (m,n).

    Returns:
      b0: (M,N,I) normalized, with I = len(pi_I0).
    """
    I = tf.shape(pi_I0)[0]
    b0 = tf.broadcast_to(pi_I0[None, None, :], tf.stack([M, N, I]))
    denom = tf.reduce_sum(b0, axis=2, keepdims=True)
    return b0 / tf.maximum(denom, eps)


def select_pi_by_state(ccp_buy: tf.Tensor, s_mt: tf.Tensor) -> tf.Tensor:
    """
    Select CCPs at the observed price state.

    Inputs:
      ccp_buy: (M,N,S,I)
      s_mt:    (M,) int32 state per market at time t

    Returns:
      pi_mnI: (M,N,I)
    """
    M = tf.shape(ccp_buy)[0]
    N = tf.shape(ccp_buy)[1]
    S = tf.shape(ccp_buy)[2]
    I = tf.shape(ccp_buy)[3]

    ccp_flat = tf.reshape(ccp_buy, tf.stack([M * N, S, I]))  # (M*N,S,I)

    s_mn = tf.tile(s_mt[:, None], tf.stack([1, N]))  # (M,N)
    s_flat = tf.reshape(s_mn, tf.stack([M * N]))  # (M*N,)

    b = tf.range(M * N, dtype=tf.int32)
    idx = tf.stack([b, s_flat], axis=1)  # (M*N,2)
    pi_flat = tf.gather_nd(ccp_flat, idx)  # (M*N,I)
    return tf.reshape(pi_flat, tf.stack([M, N, I]))


def action_likelihood_by_inventory(pi_mnI: tf.Tensor, a_mn: tf.Tensor) -> tf.Tensor:
    """
    Action likelihood conditional on inventory.

    Returns:
      lik_I: (M,N,I) where lik_I[m,n,i] = P(a_mn[m,n] | I=i).
    """
    a = tf.cast(a_mn, tf.float64)
    return a[:, :, None] * pi_mnI + (one_f64 - a)[:, :, None] * (one_f64 - pi_mnI)


def bayes_update_belief_mn(
    b: tf.Tensor, lik_I: tf.Tensor, eps: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Bayesian update step b(I) -> posterior weights proportional to b(I)*lik(I).

    Returns:
      w_post:     (M,N,I) unnormalized weights
      ll_step_mn: (M,N)   log predictive probability for each (m,n)
    """
    w_post = b * lik_I
    pred = tf.reduce_sum(w_post, axis=2)
    pred_clip = tf.maximum(pred, eps)
    ll_step_mn = tf.math.log(pred_clip)
    return w_post, ll_step_mn


def transition_belief(
    w_post: tf.Tensor,
    a_mn: tf.Tensor,
    lambda_c_mn: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Propagate belief one step forward:
      I' = clip(I + a - c, 0, I_max),  c ~ Bernoulli(lambda_c_mn)

    Returns:
      b_next: (M,N,I) normalized.
    """
    a = tf.cast(a_mn, tf.float64)
    p_c0_mn1, p_c1_mn1 = _consumption_probs(lambda_c_mn)
    p_c0_mn1 = p_c0_mn1[:, :, None]
    p_c1_mn1 = p_c1_mn1[:, :, None]

    w_down = _pushforward_inventory_mass(w_post, idx_down)
    w_up = _pushforward_inventory_mass(w_post, idx_up)

    w_next0 = p_c0_mn1 * w_post + p_c1_mn1 * w_down
    w_next1 = p_c0_mn1 * w_up + p_c1_mn1 * w_post
    w_next = (one_f64 - a)[:, :, None] * w_next0 + a[:, :, None] * w_next1

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
    Forward filter integrating out latent inventory per (m,n).

    Inputs:
      a_imt:      (M,N,T)
      p_state_mt: (M,T)
      ccp_buy:    (M,N,S,I)
      pi_I0:      (I,)
      lambda_c_mn:(M,N)

    Returns:
      loglik_mn: (M,N) where each entry is sum_t log P(a_{m,n,t} | history)
    """
    idx_down, idx_up, _, _ = _unpack_maps(maps)

    M = tf.shape(a_imt)[0]
    N = tf.shape(a_imt)[1]
    T = tf.shape(a_imt)[2]

    b0 = initial_inventory_belief(pi_I0, M, N, eps)
    ll0 = tf.zeros(tf.stack([M, N]), dtype=tf.float64)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor):
        s_mt = p_state_mt[:, t]
        a_mn = a_imt[:, :, t]

        pi_mnI = select_pi_by_state(ccp_buy, s_mt)
        lik_I = action_likelihood_by_inventory(pi_mnI, a_mn)

        w_post, ll_step_mn = bayes_update_belief_mn(b_curr, lik_I, eps)
        b_next = transition_belief(w_post, a_mn, lambda_c_mn, idx_down, idx_up, eps)
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
# Convenience: likelihood from theta
# =============================================================================


def loglik_mn_from_theta(
    theta: dict[str, tf.Tensor],
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
    Per-consumer log-likelihood contributions from constrained parameters theta.

    Returns:
      (M,N)
    """
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


# =============================================================================
# evaluation functions
# =============================================================================


def forward_filter_predict_p_buy_imt(
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    ccp_buy: tf.Tensor,
    pi_I0: tf.Tensor,
    lambda_c_mn: tf.Tensor,
    eps: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Forward filter integrating out latent inventory, returning model-consistent
    predicted buy probabilities p_buy_imt.

    At each time t, this returns the one-step-ahead predictive probability:
      p_buy[m,n,t] = sum_I b_t[m,n,I] * P(a=1 | s_mt, I)

    where b_t is the belief over inventory BEFORE observing a_{t}, and it is
    updated using the same Bayes + transition steps as loglik_hidden_inventory_mn.

    Inputs:
      a_imt:      (M,N,T) observed actions
      p_state_mt: (M,T) observed price state indices
      ccp_buy:    (M,N,S,I) CCP table from solve_ccp_buy
      pi_I0:      (I,) initial inventory distribution
      lambda_c_mn:(M,N) consumption probability
      eps:        scalar float64 for clipping/normalization
      maps:       inventory maps (idx_down, idx_up, masks)

    Returns:
      p_buy_imt: (M,N,T) float64 in (0,1)
    """
    idx_down, idx_up, _, _ = _unpack_maps(maps)

    M = tf.shape(a_imt)[0]
    N = tf.shape(a_imt)[1]
    T = tf.shape(a_imt)[2]

    b0 = initial_inventory_belief(pi_I0, M, N, eps)

    p_ta = tf.TensorArray(dtype=tf.float64, size=T, clear_after_read=False)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, p_ta_curr: tf.TensorArray) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, p_ta_curr: tf.TensorArray):
        s_mt = tf.cast(p_state_mt[:, t], tf.int32)  # (M,)
        a_mn = a_imt[:, :, t]  # (M,N)

        pi_mnI = select_pi_by_state(ccp_buy, s_mt)  # (M,N,I)

        # One-step-ahead predictive buy probability (before seeing a_t).
        p_buy_mn = tf.reduce_sum(b_curr * pi_mnI, axis=2)  # (M,N)
        p_ta_curr = p_ta_curr.write(t, p_buy_mn)

        # Standard filter update using observed a_t (same as likelihood).
        lik_I = action_likelihood_by_inventory(pi_mnI, a_mn)
        w_post, _ = bayes_update_belief_mn(b_curr, lik_I, eps)
        b_next = transition_belief(w_post, a_mn, lambda_c_mn, idx_down, idx_up, eps)

        return t + 1, b_next, p_ta_curr

    t0 = tf.constant(0, tf.int32)
    _, _, p_ta_final = tf.while_loop(cond, body, loop_vars=[t0, b0, p_ta])

    p_tmn = p_ta_final.stack()  # (T,M,N)
    return tf.transpose(p_tmn, [1, 2, 0])  # (M,N,T)


def predict_p_buy_imt_from_theta(
    theta: dict[str, tf.Tensor],
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
    Convenience wrapper: solve DP to get ccp_buy, then run the forward filter
    to return model-consistent predictive probabilities p_buy_imt.

    If theta omits "u_scale", it defaults to ones(M).
    """
    theta_use = dict(theta)
    if "u_scale" not in theta_use:
        M = tf.shape(u_m)[0]
        theta_use["u_scale"] = tf.ones(tf.stack([M]), dtype=tf.float64)

    ccp_buy = solve_ccp_buy(
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        theta=theta_use,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )

    return forward_filter_predict_p_buy_imt(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        ccp_buy=ccp_buy,
        pi_I0=pi_I0,
        lambda_c_mn=theta_use["lambda_c"],
        eps=eps,
        maps=maps,
    )
