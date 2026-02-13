"""
Pure TensorFlow core mechanics for a Ching-style stockpiling model with latent inventory.

This module is the Phase-3 "core model" layer. It assumes:
  - ALL products are modeled jointly (product axis J is explicit)
  - Price processes are exogenous and observed, and are market×product specific:
      P_price_mj    (M,J,S,S)
      price_vals_mj (M,J,S)
  - Seller observes (a_mnjt, s_mjt) only; inventory and consumption are latent.

Key tensors:
  - a_mnjt:      (M,N,J,T) {0,1} purchases
  - p_state_mjt: (M,J,T)   price state indices in {0,...,S-1}
  - u_mj:        (M,J)     fixed intercept from Phase 1–2: delta + E_bar + n

Parameters (theta):
  - per market-product (M,J): beta, alpha, v, fc
  - per market (M,):          u_scale
  - per market-consumer (M,N):lambda

This file contains:
  - Parameter transforms (z -> theta)
  - Inventory state maps
  - DP / CCP computation (batched over M,N,J)
  - Forward filter integrating out latent inventory for likelihood and predictions
"""

from __future__ import annotations

import tensorflow as tf

# InventoryMaps is intentionally just a tuple of tensors (no NamedTuple/dataclass).
# (idx_down, idx_up, stockout_mask, at_cap_mask)
InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

ONE_F64 = tf.constant(1.0, dtype=tf.float64)


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Map unconstrained parameters z[*] to constrained theta[*].

    Required z keys and shapes:
      z_beta    (M,J) -> beta     in (0,1)
      z_alpha   (M,J) -> alpha    > 0
      z_v       (M,J) -> v        > 0
      z_fc      (M,J) -> fc       > 0
      z_lambda  (M,N) -> lambda   in (0,1)
      z_u_scale (M,)  -> u_scale  > 0 (shared within each market)

    Returns:
      dict[str, tf.Tensor] with keys:
        beta (M,J), alpha (M,J), v (M,J), fc (M,J), lambda (M,N), u_scale (M,)
    """
    return {
        "beta": tf.math.sigmoid(z["z_beta"]),
        "alpha": tf.exp(z["z_alpha"]),
        "v": tf.exp(z["z_v"]),
        "fc": tf.exp(z["z_fc"]),
        "lambda": tf.math.sigmoid(z["z_lambda"]),
        "u_scale": tf.exp(z["z_u_scale"]),
    }


def _consumption_probs(lambda_mn: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Consumption mixture weights.

    Args:
      lambda_mn: (M,N) float64 in (0,1)

    Returns:
      (p_c0, p_c1): both (M,N) float64 where
        p_c1 = P(c=1) = lambda
        p_c0 = P(c=0) = 1 - lambda
    """
    lambda_mn = tf.cast(lambda_mn, tf.float64)
    p_c1 = tf.clip_by_value(lambda_mn, 0.0, 1.0)
    p_c0 = ONE_F64 - p_c1
    return p_c0, p_c1


def _ensure_theta(theta: dict[str, tf.Tensor], u_mj: tf.Tensor) -> dict[str, tf.Tensor]:
    """Ensure required theta keys exist and enforce float64."""
    theta_use = dict(theta)

    if "u_scale" not in theta_use:
        M = tf.shape(u_mj)[0]
        theta_use["u_scale"] = tf.ones((M,), dtype=tf.float64)

    for k in ("beta", "alpha", "v", "fc", "lambda", "u_scale"):
        theta_use[k] = tf.cast(theta_use[k], tf.float64)

    return theta_use


# =============================================================================
# Inventory maps
# =============================================================================


def build_inventory_maps(I_max: tf.Tensor) -> InventoryMaps:
    """
    Build deterministic inventory mapping index vectors and masks.

    Inventory states: I in {0,...,I_max}, size I = I_max+1.

    Deterministic mappings:
      down(i) = max(i-1, 0)
      up(i)   = min(i+1, I_max)

    Returns:
      idx_down:      (I,) int32 with idx_down[i] = down(i)
      idx_up:        (I,) int32 with idx_up[i]   = up(i)
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
    idx_down, idx_up, stockout_mask, at_cap_mask = maps
    return idx_down, idx_up, stockout_mask, at_cap_mask


def _pushforward_inventory_mass(w_mnji: tf.Tensor, idx_next_i: tf.Tensor) -> tf.Tensor:
    """
    Push mass forward under a deterministic mapping i -> idx_next_i[i].

    Implements:
      w_next[..., j] = sum_{i : idx_next_i[i] = j} w[..., i]

    Inputs:
      w_mnji:     (M,N,J,I)
      idx_next_i: (I,) int32 mapping each current inventory i to next index

    Returns:
      w_next_mnji: (M,N,J,I)
    """
    I = tf.shape(w_mnji)[3]

    w_imnj = tf.transpose(w_mnji, perm=[3, 0, 1, 2])  # (I,M,N,J)
    w_iK = tf.reshape(w_imnj, tf.stack([I, -1]))  # (I, M*N*J)

    w_next_iK = tf.math.unsorted_segment_sum(w_iK, idx_next_i, num_segments=I)

    w_next_imnj = tf.reshape(
        w_next_iK,
        tf.stack([I, tf.shape(w_mnji)[0], tf.shape(w_mnji)[1], tf.shape(w_mnji)[2]]),
    )  # (I,M,N,J)
    return tf.transpose(w_next_imnj, perm=[1, 2, 3, 0])  # (M,N,J,I)


# =============================================================================
# DP / CCP computation (batched over markets, consumers, products)
# =============================================================================


def make_flow_utilities(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Construct flow utilities u0,u1 for no-buy and buy actions.

    Inputs:
      u_mj:          (M,J) float64
      price_vals_mj: (M,J,S) float64
      theta:
        alpha (M,J), v (M,J), fc (M,J), lambda (M,N), u_scale (M,)
      waste_cost: scalar float64
      maps: inventory maps (contains stockout and at-cap masks)

    Returns:
      u0,u1: both (M,N,J,S,I) float64
    """
    _, _, stockout_mask, at_cap_mask = _unpack_maps(maps)

    u_mj = tf.cast(u_mj, tf.float64)
    price_vals_mj = tf.cast(price_vals_mj, tf.float64)
    waste_cost = tf.cast(waste_cost, tf.float64)

    alpha_mj = tf.cast(theta["alpha"], tf.float64)  # (M,J)
    v_mj = tf.cast(theta["v"], tf.float64)  # (M,J)
    fc_mj = tf.cast(theta["fc"], tf.float64)  # (M,J)
    lambda_mn = tf.cast(theta["lambda"], tf.float64)  # (M,N)
    u_scale_m = tf.cast(theta["u_scale"], tf.float64)  # (M,)

    # Dimensions
    M = tf.shape(lambda_mn)[0]
    N = tf.shape(lambda_mn)[1]
    J = tf.shape(u_mj)[1]
    S = tf.shape(price_vals_mj)[2]
    I = tf.shape(stockout_mask)[0]

    # Masks (broadcast-ready)
    stockout_1111I = stockout_mask[None, None, None, None, :]  # (1,1,1,1,I)
    at_cap_1111I = at_cap_mask[None, None, None, None, :]  # (1,1,1,1,I)

    # --- Buy utility u1 ---
    # u_eff(m,j) = u_scale(m) * u_mj(m,j)
    u_eff_m1j11 = (u_scale_m[:, None] * u_mj)[:, None, :, None, None]  # (M,1,J,1,1)

    # base_buy = u_eff - alpha * price(s) - fc
    alpha_m1j11 = alpha_mj[:, None, :, None, None]  # (M,1,J,1,1)
    fc_m1j11 = fc_mj[:, None, :, None, None]  # (M,1,J,1,1)
    price_m1js1 = price_vals_mj[:, None, :, :, None]  # (M,1,J,S,1)
    base_buy_m1js1 = u_eff_m1j11 - alpha_m1j11 * price_m1js1 - fc_m1j11  # (M,1,J,S,1)

    # Waste-at-cap penalty: waste_cost*(1-lambda_mn)*1{I=I_max}
    one_minus_lambda_mn = ONE_F64 - tf.clip_by_value(lambda_mn, 0.0, 1.0)  # (M,N)
    waste_term_mn111I = (
        waste_cost * one_minus_lambda_mn[:, :, None, None, None] * at_cap_1111I
    )  # (M,N,1,1,I)

    # Broadcasting produces (M,N,J,S,I)
    u1 = base_buy_m1js1 - waste_term_mn111I

    # --- No-buy utility u0 ---
    # u0 = -v_mj * 1{I=0}
    u0_m1j1I = -v_mj[:, None, :, None, None] * stockout_1111I  # (M,1,J,1,I)
    u0 = tf.broadcast_to(u0_m1j1I, tf.stack([M, N, J, S, I]))

    return u0, u1


def expected_over_next_price(V: tf.Tensor, P_price_mj: tf.Tensor) -> tf.Tensor:
    """
    Compute E[V(s', I) | current s] under market×product-specific price transitions.

    Inputs:
      V:          (M,N,J,S,I)
      P_price_mj: (M,J,S,S) with P[s,s'] row-stochastic

    Returns:
      EV: (M,N,J,S,I) where EV[..., s, i] = sum_{s'} P[s,s'] * V[..., s', i]
    """
    V = tf.cast(V, tf.float64)
    P = tf.cast(P_price_mj, tf.float64)
    return tf.einsum("mjab,mnjbi->mnjai", P, V)


def bellman_update(
    V: tf.Tensor,
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One Bellman update for logit shocks.

    Returns:
      (V_new, q0, q1): all (M,N,J,S,I)
    """
    ev_next = expected_over_next_price(V, P_price_mj)  # (M,N,J,S,I)

    beta_m1j11 = tf.cast(beta_mj, tf.float64)[:, None, :, None, None]  # (M,1,J,1,1)

    ev_down = tf.gather(ev_next, idx_down, axis=4)  # (M,N,J,S,I)
    ev_up = tf.gather(ev_next, idx_up, axis=4)  # (M,N,J,S,I)

    cont0 = p_c0_mn111 * ev_next + p_c1_mn111 * ev_down
    cont1 = p_c0_mn111 * ev_up + p_c1_mn111 * ev_next

    q0 = u0 + beta_m1j11 * cont0
    q1 = u1 + beta_m1j11 * cont1
    V_new = tf.reduce_logsumexp(tf.stack([q0, q1], axis=0), axis=0)
    return V_new, q0, q1


def solve_value_function(
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
) -> tf.Tensor:
    """
    Value iteration until max-norm change <= tol or max_iter reached.

    Returns:
      V_final: (M,N,J,S,I)
    """
    V0 = tf.zeros_like(u0)
    diff0 = tf.constant(1.0e30, dtype=tf.float64)

    tol = tf.cast(tol, tf.float64)
    max_iter = tf.cast(max_iter, tf.int32)

    def cond(it: tf.Tensor, V: tf.Tensor, diff: tf.Tensor):
        return tf.logical_and(it < max_iter, diff > tol)

    def body(it: tf.Tensor, V: tf.Tensor, diff: tf.Tensor):
        V_new, _, _ = bellman_update(
            V,
            u0,
            u1,
            beta_mj,
            P_price_mj,
            p_c0_mn111,
            p_c1_mn111,
            idx_down,
            idx_up,
        )
        diff_new = tf.reduce_max(tf.abs(V_new - V))
        return it + 1, V_new, diff_new

    it0 = tf.constant(0, tf.int32)
    _, V_final, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[it0, V0, diff0],
        shape_invariants=[
            it0.get_shape(),
            tf.TensorShape([None, None, None, None, None]),
            diff0.get_shape(),
        ],
    )
    return V_final


def solve_ccp_buy(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    waste_cost: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Solve the DP and return CCP table for buying.

    Inputs:
      u_mj:          (M,J)
      price_vals_mj: (M,J,S)
      P_price_mj:    (M,J,S,S)
      theta: beta(M,J), alpha(M,J), v(M,J), fc(M,J), lambda(M,N), u_scale(M,)
      maps: inventory maps

    Returns:
      ccp_buy: (M,N,J,S,I) with P(a=1 | s, I)
    """
    theta = _ensure_theta(theta, u_mj)
    u_mj = tf.cast(u_mj, tf.float64)
    price_vals_mj = tf.cast(price_vals_mj, tf.float64)
    P_price_mj = tf.cast(P_price_mj, tf.float64)
    waste_cost = tf.cast(waste_cost, tf.float64)

    idx_down, idx_up, _, _ = _unpack_maps(maps)

    u0, u1 = make_flow_utilities(
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        theta=theta,
        waste_cost=waste_cost,
        maps=maps,
    )

    p_c0_mn, p_c1_mn = _consumption_probs(theta["lambda"])  # (M,N)
    p_c0_mn111 = p_c0_mn[:, :, None, None, None]  # (M,N,1,1,1)
    p_c1_mn111 = p_c1_mn[:, :, None, None, None]

    V_final = solve_value_function(
        u0=u0,
        u1=u1,
        beta_mj=theta["beta"],
        P_price_mj=P_price_mj,
        p_c0_mn111=p_c0_mn111,
        p_c1_mn111=p_c1_mn111,
        idx_down=idx_down,
        idx_up=idx_up,
        tol=tol,
        max_iter=max_iter,
    )

    _, q0, q1 = bellman_update(
        V_final,
        u0,
        u1,
        theta["beta"],
        P_price_mj,
        p_c0_mn111,
        p_c1_mn111,
        idx_down,
        idx_up,
    )

    return tf.math.sigmoid(q1 - q0)


# =============================================================================
# Forward filter (latent inventory integration)
# =============================================================================


def initial_inventory_belief(
    pi_I0: tf.Tensor,
    M: tf.Tensor,
    N: tf.Tensor,
    J: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Broadcast initial inventory belief across (m,n,j).

    Args:
      pi_I0: (I,) float64, should sum to 1
      M,N,J: scalar int tensors
      eps:   scalar float64

    Returns:
      b0: (M,N,J,I)
    """
    pi = tf.cast(pi_I0, tf.float64)
    eps = tf.cast(eps, tf.float64)
    pi = tf.maximum(pi, eps)
    pi = pi / tf.reduce_sum(pi)

    I = tf.shape(pi)[0]
    return tf.broadcast_to(pi[None, None, None, :], tf.stack([M, N, J, I]))


def select_pi_by_state(ccp_buy: tf.Tensor, s_mj: tf.Tensor) -> tf.Tensor:
    """
    Select CCPs at the observed price state.

    Inputs:
      ccp_buy: (M,N,J,S,I)
      s_mj:    (M,J) int32 state per market-product at time t

    Returns:
      pi_mnjI: (M,N,J,I)
    """
    s_mj = tf.cast(s_mj, tf.int32)

    # Move batch dims (M,J) to the front and gather along S with batch_dims=2.
    ccp_mjnsI = tf.transpose(ccp_buy, perm=[0, 2, 1, 3, 4])  # (M,J,N,S,I)
    pi_mjnI = tf.gather(ccp_mjnsI, s_mj, axis=3, batch_dims=2)  # (M,J,N,I)
    return tf.transpose(pi_mjnI, perm=[0, 2, 1, 3])  # (M,N,J,I)


def action_likelihood_by_inventory(pi_mnjI: tf.Tensor, a_mnj: tf.Tensor) -> tf.Tensor:
    """
    Likelihood of observed action given inventory state.

    Inputs:
      pi_mnjI: (M,N,J,I) probability of buy if in inventory I at observed price state
      a_mnj:   (M,N,J) observed actions {0,1}

    Returns:
      lik_I: (M,N,J,I) where lik_I = pi if a=1 else (1-pi)
    """
    pi = tf.cast(pi_mnjI, tf.float64)
    a = tf.cast(a_mnj, tf.float64)[:, :, :, None]
    return a * pi + (ONE_F64 - a) * (ONE_F64 - pi)


def bayes_update_belief_mnj(
    b_curr: tf.Tensor,
    lik_I: tf.Tensor,
    eps: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Bayes update of inventory belief.

    Returns:
      b_post:  (M,N,J,I)
      ll_step: (M,N,J) log predictive probability at this step
    """
    eps = tf.cast(eps, tf.float64)
    w = tf.cast(b_curr, tf.float64) * tf.cast(lik_I, tf.float64)
    Z = tf.reduce_sum(w, axis=3, keepdims=True)  # (M,N,J,1)
    Zc = tf.maximum(Z, eps)
    b_post = w / Zc
    ll_step = tf.math.log(Zc)[:, :, :, 0]  # (M,N,J)
    return b_post, ll_step


def transition_belief(
    b_post: tf.Tensor,
    a_mnj: tf.Tensor,
    lambda_mn: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Predictive belief transition b_{t+1} from posterior b_t after observing a_t.

    Returns:
      b_next: (M,N,J,I)
    """
    eps = tf.cast(eps, tf.float64)

    p_c0_mn, p_c1_mn = _consumption_probs(lambda_mn)  # (M,N)
    p_c0 = p_c0_mn[:, :, None, None]  # (M,N,1,1)
    p_c1 = p_c1_mn[:, :, None, None]

    b = tf.cast(b_post, tf.float64)

    b_down = _pushforward_inventory_mass(b, idx_down)  # (M,N,J,I)
    b_up = _pushforward_inventory_mass(b, idx_up)  # (M,N,J,I)

    b_a0 = p_c0 * b + p_c1 * b_down
    b_a1 = p_c0 * b_up + p_c1 * b

    a_bool = tf.not_equal(a_mnj, 0)[:, :, :, None]  # (M,N,J,1)
    b_next = tf.where(a_bool, b_a1, b_a0)

    Z = tf.reduce_sum(b_next, axis=3, keepdims=True)
    Zc = tf.maximum(Z, eps)
    return b_next / Zc


def _filter_step(
    b_curr: tf.Tensor,
    a_mnj: tf.Tensor,
    s_mj: tf.Tensor,
    ccp_buy: tf.Tensor,
    lambda_mn: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    eps: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """One forward-filter step shared by loglik and predictive p_buy."""
    pi_mnjI = select_pi_by_state(ccp_buy, s_mj)  # (M,N,J,I)
    p_buy_mnj = tf.reduce_sum(b_curr * pi_mnjI, axis=3)  # (M,N,J)

    lik_I = action_likelihood_by_inventory(pi_mnjI, a_mnj)  # (M,N,J,I)
    b_post, ll_step = bayes_update_belief_mnj(b_curr, lik_I, eps)

    b_next = transition_belief(b_post, a_mnj, lambda_mn, idx_down, idx_up, eps)
    return b_next, ll_step, p_buy_mnj


def loglik_hidden_inventory_mnj(
    a_mnjt: tf.Tensor,
    p_state_mjt: tf.Tensor,
    ccp_buy: tf.Tensor,
    pi_I0: tf.Tensor,
    lambda_mn: tf.Tensor,
    eps: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Forward filter integrating out latent inventory per (m,n,j).

    Returns:
      loglik_mnj: (M,N,J)
    """
    idx_down, idx_up, _, _ = _unpack_maps(maps)

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    b0 = initial_inventory_belief(pi_I0, M, N, J, eps)
    ll0 = tf.zeros((M, N, J), dtype=tf.float64)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor):
        s_mj = tf.cast(p_state_mjt[:, :, t], tf.int32)  # (M,J)
        a_mnj = a_mnjt[:, :, :, t]  # (M,N,J)

        b_next, ll_step, _ = _filter_step(
            b_curr=b_curr,
            a_mnj=a_mnj,
            s_mj=s_mj,
            ccp_buy=ccp_buy,
            lambda_mn=lambda_mn,
            idx_down=idx_down,
            idx_up=idx_up,
            eps=eps,
        )
        return t + 1, b_next, ll_curr + ll_step

    t0 = tf.constant(0, tf.int32)
    _, _, ll_final = tf.while_loop(
        cond,
        body,
        loop_vars=[t0, b0, ll0],
        shape_invariants=[
            t0.get_shape(),
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None, None, None]),
        ],
    )
    return ll_final


def _prepare_theta_and_ccp(
    theta: dict[str, tf.Tensor],
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    waste_cost: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    """Shared prep: ensure theta, then solve CCPs."""
    theta_use = _ensure_theta(theta, u_mj)
    ccp_buy = solve_ccp_buy(
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        theta=theta_use,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    return theta_use, ccp_buy


def loglik_mnj_from_theta(
    theta: dict[str, tf.Tensor],
    a_mnjt: tf.Tensor,
    p_state_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Convenience wrapper: solve DP to get ccp_buy, then run the forward filter.

    If theta omits "u_scale", it defaults to ones(M).

    Returns:
      loglik_mnj: (M,N,J)
    """
    theta_use, ccp_buy = _prepare_theta_and_ccp(
        theta=theta,
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    return loglik_hidden_inventory_mnj(
        a_mnjt=a_mnjt,
        p_state_mjt=p_state_mjt,
        ccp_buy=ccp_buy,
        pi_I0=pi_I0,
        lambda_mn=theta_use["lambda"],
        eps=eps,
        maps=maps,
    )


# =============================================================================
# Predictive probabilities for evaluation
# =============================================================================


def forward_filter_predict_p_buy_mnjt(
    a_mnjt: tf.Tensor,
    p_state_mjt: tf.Tensor,
    ccp_buy: tf.Tensor,
    pi_I0: tf.Tensor,
    lambda_mn: tf.Tensor,
    eps: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Forward filter returning one-step-ahead predictive buy probabilities.

    At each time t:
      p_buy[m,n,j,t] = sum_I b_t[m,n,j,I] * P(a=1 | s_mjt, I)
    """
    idx_down, idx_up, _, _ = _unpack_maps(maps)

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    b0 = initial_inventory_belief(pi_I0, M, N, J, eps)
    p_ta = tf.TensorArray(dtype=tf.float64, size=T, clear_after_read=False)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, p_ta_curr: tf.TensorArray) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, p_ta_curr: tf.TensorArray):
        s_mj = tf.cast(p_state_mjt[:, :, t], tf.int32)  # (M,J)
        a_mnj = a_mnjt[:, :, :, t]  # (M,N,J)

        b_next, _, p_buy_mnj = _filter_step(
            b_curr=b_curr,
            a_mnj=a_mnj,
            s_mj=s_mj,
            ccp_buy=ccp_buy,
            lambda_mn=lambda_mn,
            idx_down=idx_down,
            idx_up=idx_up,
            eps=eps,
        )

        p_ta_curr = p_ta_curr.write(t, p_buy_mnj)
        return t + 1, b_next, p_ta_curr

    t0 = tf.constant(0, tf.int32)
    _, _, p_ta_final = tf.while_loop(cond, body, loop_vars=[t0, b0, p_ta])

    p_tmnJ = p_ta_final.stack()  # (T,M,N,J)
    return tf.transpose(p_tmnJ, [1, 2, 3, 0])  # (M,N,J,T)


def predict_p_buy_mnjt_from_theta(
    theta: dict[str, tf.Tensor],
    a_mnjt: tf.Tensor,
    p_state_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Convenience wrapper: solve DP to get ccp_buy, then run the forward filter.

    If theta omits "u_scale", it defaults to ones(M).

    Returns:
      p_buy_mnjt: (M,N,J,T)
    """
    theta_use, ccp_buy = _prepare_theta_and_ccp(
        theta=theta,
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    return forward_filter_predict_p_buy_mnjt(
        a_mnjt=a_mnjt,
        p_state_mjt=p_state_mjt,
        ccp_buy=ccp_buy,
        pi_I0=pi_I0,
        lambda_mn=theta_use["lambda"],
        eps=eps,
        maps=maps,
    )
