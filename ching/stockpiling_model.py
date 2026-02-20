"""Stockpiling dynamic discrete-choice model (Ching-style).

This module implements the structural DP/CCP solver for a binary buy decision with
inventory dynamics and an exogenous Markov price process.

Indices:
  m = 0..M-1   market
  n = 0..N-1   consumer (within market)
  j = 0..J-1   product
  s = 0..S-1   exogenous price state
  I = 0..I_max inventory units (per product)

Inputs (float64 unless noted):
  u_mj:          (M, J)        fixed intercept utility from Phase 1–2
  P_price_mj:    (M, J, S, S)  price-state Markov transitions (row-stochastic)
  price_vals_mj: (M, J, S)     price levels for each price state
  lambda_mn:     (M, N)        consumption probability in (0, 1) (passed, not estimated)
  waste_cost:    scalar        waste penalty weight

Parameters (theta, float64):
  beta:    scalar in (0, 1)    discount factor (shared across markets/products)
  alpha:   (J,)  > 0           price sensitivity
  v:       (J,)  > 0           stockout penalty
  fc:      (J,)  > 0           fixed purchase cost
  u_scale: (M,)  > 0           nuisance scale on u_mj used in estimation

Role of u_scale:
  - DGP: u_scale_m is fixed to 1 for all m.
  - Estimation: u_scale_m may be included as a nuisance parameter; in this project
    it can be frozen (e.g., u_scale_m ≡ 1) for controlled tests.

Flow utilities (action a ∈ {0, 1}, Type-I EV shocks for logit CCPs):
  No buy (a=0):  U0 = -v_j * 1{I=0}
  Buy (a=1):     U1 = u_scale_m * u_mj - alpha_j * price(s) - fc_j
                       - waste_cost * (1 - lambda_mn) * 1{I=I_max}
"""

from __future__ import annotations

import tensorflow as tf


# =============================================================================
# Constants
# =============================================================================

ONE_F64 = tf.constant(1.0, dtype=tf.float64)


# =============================================================================
# Parameter transforms (estimation utilities)
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Map unconstrained parameters to constrained theta.

    This helper is typically used by estimators. The structural solver uses theta
    directly.

    Required keys (float64):
      z_beta, z_alpha, z_v, z_fc, z_u_scale

    Returns (float64):
      beta (scalar), alpha (J,), v (J,), fc (J,), u_scale (M,)
    """
    beta = tf.sigmoid(z["z_beta"])  # (0,1)
    alpha = tf.exp(z["z_alpha"])  # >0
    v = tf.exp(z["z_v"])  # >0
    fc = tf.exp(z["z_fc"])  # >0
    u_scale = tf.exp(z["z_u_scale"])  # >0

    return {
        "beta": tf.reshape(beta, ()),
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "u_scale": u_scale,
    }


def _ensure_theta(theta: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Ensure required theta keys exist and normalize beta to a scalar."""
    required = ("beta", "alpha", "v", "fc", "u_scale")
    missing = [k for k in required if k not in theta]
    if missing:
        raise KeyError(f"theta is missing required keys: {missing}")

    theta_use = dict(theta)
    theta_use["beta"] = tf.reshape(theta_use["beta"], ())
    return theta_use


# =============================================================================
# Inventory grid maps
# =============================================================================

# (I_vals, stockout_mask, at_cap_mask, idx_down, idx_up)
InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


def build_inventory_maps(I_max: int) -> InventoryMaps:
    """Precompute inventory-grid masks and transition indices for I ∈ {0..I_max}."""
    I_vals = tf.range(I_max + 1, dtype=tf.int32)  # (I,)

    stockout_mask = tf.cast(tf.equal(I_vals, 0), tf.float64)  # (I,)
    at_cap_mask = tf.cast(tf.equal(I_vals, I_max), tf.float64)  # (I,)

    # Clipped inventory transitions for +/- 1 inventory changes.
    idx_down = tf.maximum(I_vals - 1, 0)  # (I,)
    idx_up = tf.minimum(I_vals + 1, I_max)  # (I,)

    return I_vals, stockout_mask, at_cap_mask, idx_down, idx_up


def _unpack_maps(
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    I_vals, stockout_mask, at_cap_mask, idx_down, idx_up = maps
    return I_vals, stockout_mask, at_cap_mask, idx_down, idx_up


# =============================================================================
# Flow utilities
# =============================================================================


def make_flow_utilities(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Construct flow utilities for no-buy (u0) and buy (u1).

    Returns:
      u0, u1: (M, N, J, S, I)
    """
    _, stockout_mask, at_cap_mask, _, _ = _unpack_maps(maps)

    # Parameters
    alpha_j = theta["alpha"]  # (J,)
    v_j = theta["v"]  # (J,)
    fc_j = theta["fc"]  # (J,)
    u_scale_m = theta["u_scale"]  # (M,)

    # Dimensions
    M = tf.shape(u_mj)[0]
    J = tf.shape(u_mj)[1]
    N = tf.shape(lambda_mn)[1]
    S = tf.shape(price_vals_mj)[2]
    I = tf.shape(stockout_mask)[0]

    # Inventory masks (broadcast-ready)
    stockout_1111I = stockout_mask[None, None, None, None, :]  # (1,1,1,1,I)
    at_cap_1111I = at_cap_mask[None, None, None, None, :]  # (1,1,1,1,I)

    # Buy utility
    # u_eff(m,j) = u_scale(m) * u_mj(m,j)
    u_eff_m1j11 = (u_scale_m[:, None] * u_mj)[:, None, :, None, None]  # (M,1,J,1,1)

    alpha_11j11 = alpha_j[None, None, :, None, None]  # (1,1,J,1,1)
    fc_11j11 = fc_j[None, None, :, None, None]  # (1,1,J,1,1)
    price_m1js1 = price_vals_mj[:, None, :, :, None]  # (M,1,J,S,1)

    base_buy_m1js1 = u_eff_m1j11 - alpha_11j11 * price_m1js1 - fc_11j11  # (M,1,J,S,1)

    # Waste-at-cap penalty: waste_cost*(1-lambda_mn)*1{I=I_max}
    one_minus_lambda_mn = ONE_F64 - lambda_mn  # (M,N)
    waste_term_mn111I = (
        waste_cost * one_minus_lambda_mn[:, :, None, None, None] * at_cap_1111I
    )  # (M,N,1,1,I)

    # Broadcast to (M,N,J,S,I)
    u1 = base_buy_m1js1 - waste_term_mn111I

    # No-buy utility: -v(j)*1{I=0}
    u0_11j1I = -v_j[None, None, :, None, None] * stockout_1111I  # (1,1,J,1,I)
    u0 = tf.broadcast_to(u0_11j1I, tf.stack([M, N, J, S, I]))

    return u0, u1


# =============================================================================
# Expectations over next states
# =============================================================================


def expected_over_next_price(V: tf.Tensor, P_price_mj: tf.Tensor) -> tf.Tensor:
    """E[V(s', I) | current s] under the Markov price transition."""
    # cont[m,n,j,s,i] = sum_{s'} P[m,j,s,s'] * V[m,n,j,s',i]
    return tf.einsum("mjsr,mnjri->mnjsi", P_price_mj, V)


def expected_over_next_inv(
    cont_s: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    action: int,
) -> tf.Tensor:
    """E[cont_s(s', I') | current I, action] integrating over c ∈ {0,1}."""
    # Gather cont at I' = clip(I + a - c, 0, I_max).
    if action == 0:
        # a=0: if c=0 -> I' = I ; if c=1 -> I' = I-1
        cont_c0 = cont_s
        cont_c1 = tf.gather(cont_s, idx_down, axis=4)
    else:
        # a=1: if c=0 -> I' = I+1 ; if c=1 -> I' = I
        cont_c0 = tf.gather(cont_s, idx_up, axis=4)
        cont_c1 = cont_s

    return p_c0_mn111 * cont_c0 + p_c1_mn111 * cont_c1


# =============================================================================
# Logit choice-specific values and CCPs
# =============================================================================


def logsumexp_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """Compute log(exp(q0)+exp(q1)) elementwise."""
    return q0 + tf.nn.softplus(q1 - q0)


def ccp_from_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """Logit CCP for buy: sigmoid(q1-q0)."""
    return tf.sigmoid(q1 - q0)


# =============================================================================
# Value function solver
# =============================================================================


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
    """One Bellman update for the logit DDC."""
    cont_s = expected_over_next_price(V, P_price_mj)

    cont0 = expected_over_next_inv(
        cont_s, p_c0_mn111, p_c1_mn111, idx_down, idx_up, action=0
    )
    cont1 = expected_over_next_inv(
        cont_s, p_c0_mn111, p_c1_mn111, idx_down, idx_up, action=1
    )

    beta_11111 = tf.reshape(beta, (1, 1, 1, 1, 1))
    q0 = u0 + beta_11111 * cont0
    q1 = u1 + beta_11111 * cont1

    V_new = logsumexp_q(q0, q1)
    return V_new, q0, q1


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
    """Solve V = T(V) via fixed-point iteration."""
    V0 = tf.zeros_like(u0)
    max_diff0 = tf.constant(float("inf"), dtype=tf.float64)
    tol_t = tf.constant(tol, dtype=tf.float64)

    def cond_fn(_: tf.Tensor, max_diff: tf.Tensor) -> tf.Tensor:
        return max_diff > tol_t

    def body_fn(V_prev: tf.Tensor, _: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        V_new, _, _ = bellman_update(
            V_prev, u0, u1, beta, P_price_mj, p_c0_mn111, p_c1_mn111, idx_down, idx_up
        )
        diff = tf.reduce_max(tf.abs(V_new - V_prev))
        return V_new, diff

    V_final, _ = tf.while_loop(
        cond=cond_fn,
        body=body_fn,
        loop_vars=(V0, max_diff0),
        maximum_iterations=max_iter,
    )
    return V_final


# =============================================================================
# CCP solver
# =============================================================================


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
    """Solve buy CCPs on the (s, I) grid for each (m, n, j).

    Returns:
      ccp_buy, q0, q1: each (M, N, J, S, I)
    """
    theta_use = _ensure_theta(theta)

    _, _, _, idx_down, idx_up = _unpack_maps(maps)

    # Consumption probabilities
    p_c1_mn111 = lambda_mn[:, :, None, None, None]  # (M,N,1,1,1)
    p_c0_mn111 = (ONE_F64 - lambda_mn)[:, :, None, None, None]  # (M,N,1,1,1)

    # Flow utilities
    u0, u1 = make_flow_utilities(
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        theta=theta_use,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        maps=maps,
    )

    # Solve V
    V_final = solve_value_function(
        u0=u0,
        u1=u1,
        beta=theta_use["beta"],
        P_price_mj=P_price_mj,
        p_c0_mn111=p_c0_mn111,
        p_c1_mn111=p_c1_mn111,
        idx_down=idx_down,
        idx_up=idx_up,
        tol=tol,
        max_iter=max_iter,
    )

    # Recover q0, q1 and CCPs at the fixed point
    _, q0, q1 = bellman_update(
        V_final,
        u0,
        u1,
        theta_use["beta"],
        P_price_mj,
        p_c0_mn111,
        p_c1_mn111,
        idx_down,
        idx_up,
    )
    ccp_buy = ccp_from_q(q0, q1)

    return ccp_buy, q0, q1


# =============================================================================
# Helper: forward simulation given CCPs
# =============================================================================


def simulate_purchases_given_ccp(
    ccp_buy: tf.Tensor,
    s_mjt: tf.Tensor,
    lambda_mn: tf.Tensor,
    pi_I0: tf.Tensor,
    I_max: int,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Simulate purchases, consumption, and inventories given CCPs and price states.

    Inputs:
      ccp_buy: (M, N, J, S, I) buy probabilities
      s_mjt:   (M, J, T)       observed price states (int32)
      lambda_mn: (M, N)        consumption probabilities
      pi_I0:   (I,)            initial inventory distribution over {0..I_max}
      I_max:   int
      rng:     tf.random.Generator

    Returns:
      a_mnjt: (M, N, J, T)   purchases in {0,1}
      c_mnjt: (M, N, J, T)   consumption in {0,1}
      I_mnjt: (M, N, J, T+1) inventory path
    """
    M = tf.shape(ccp_buy)[0]
    N = tf.shape(ccp_buy)[1]
    J = tf.shape(ccp_buy)[2]
    T = tf.shape(s_mjt)[2]

    # Sample initial inventory I_0 ~ pi_I0 independently across (m,n,j).
    cdf = tf.cumsum(pi_I0)  # (I,)
    u0 = rng.uniform((M, N, J), dtype=tf.float64)
    I_mnj = tf.reduce_sum(
        tf.cast(u0[..., None] > cdf[None, None, None, :], tf.int32), axis=-1
    )

    I_path = tf.TensorArray(tf.int32, size=T + 1, clear_after_read=False)
    a_path = tf.TensorArray(tf.int32, size=T, clear_after_read=False)
    c_path = tf.TensorArray(tf.int32, size=T, clear_after_read=False)

    I_path = I_path.write(0, I_mnj)

    for t in tf.range(T):
        s_mj = s_mjt[:, :, t]  # (M,J)
        s_grid = tf.broadcast_to(s_mj[:, None, :], (M, N, J))  # (M,N,J)

        # p_buy(m,n,j) = ccp_buy[m,n,j, s_mjt[m,j,t], I_mnj[m,n,j]]
        p_by_s = tf.gather(ccp_buy, s_grid, axis=3, batch_dims=3)  # (M,N,J,I)
        p_buy = tf.gather(p_by_s, I_mnj, axis=3, batch_dims=3)  # (M,N,J)

        # Purchases
        u = rng.uniform((M, N, J), dtype=tf.float64)
        a = tf.cast(u < p_buy, tf.int32)

        # Consumption: independent across products j, with common lambda per (m,n)
        u_c = rng.uniform((M, N, J), dtype=tf.float64)
        c = tf.cast(u_c < lambda_mn[:, :, None], tf.int32)

        # Inventory transition: clip(I + a - c, 0, I_max)
        I_mnj = tf.clip_by_value(I_mnj + a - c, 0, I_max)

        a_path = a_path.write(t, a)
        c_path = c_path.write(t, c)
        I_path = I_path.write(t + 1, I_mnj)

    a_mnjt = tf.transpose(a_path.stack(), perm=[1, 2, 3, 0])
    c_mnjt = tf.transpose(c_path.stack(), perm=[1, 2, 3, 0])
    I_mnjt = tf.transpose(I_path.stack(), perm=[1, 2, 3, 0])

    return a_mnjt, c_mnjt, I_mnjt
