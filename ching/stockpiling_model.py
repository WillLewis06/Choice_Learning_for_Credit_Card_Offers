"""Stockpiling dynamic discrete-choice model (Ching-style).

Indices:
  m = 0..M-1 market
  n = 0..N-1 consumer (within market)
  j = 0..J-1 product
  s = 0..S-1 exogenous price state
  I = 0..I_max inventory units (per product)

Known inputs:
  u_mj: (M, J) fixed intercept utility from Phase 1–2
  P_price_mj: (M, J, S, S) price-state Markov transitions (row-stochastic)
  price_vals_mj: (M, J, S) price levels for each price state

Model parameters (theta):
  beta:    (1,)   discount factor in (0, 1), shared across markets/products
  alpha:   (J,)   price sensitivity > 0, product-specific
  v:       (J,)   stockout penalty > 0, product-specific
  fc:      (J,)   fixed purchase cost > 0, product-specific
  u_scale: (M,)   positive nuisance scale on u_mj during estimation

Known (passed as inputs, not estimated):
  lambda_mn: (M, N) consumption probability in (0, 1), market-consumer specific

Flow utilities (action a ∈ {0, 1}, Type-I EV shocks for logit CCPs):
  No buy (a=0):  U0 = - v_j * 1{I=0}
  Buy (a=1):     U1 = u_scale_m * u_mj - alpha_j * price - fc_j
                       - waste_cost * (1 - lambda_mn) * 1{I=I_max}
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


# =============================================================================
# Constants
# =============================================================================

ZERO_F64 = tf.constant(0.0, dtype=tf.float64)
ONE_F64 = tf.constant(1.0, dtype=tf.float64)


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Map unconstrained parameters z[*] to constrained theta[*].

    Required z keys and shapes:
      z_beta    (1,)   -> beta     in (0, 1)
      z_alpha   (J,)   -> alpha    > 0
      z_v       (J,)   -> v        > 0
      z_fc      (J,)   -> fc       > 0
      z_u_scale (M,)   -> u_scale  > 0 (shared within each market)

    Returns:
      dict[str, tf.Tensor] with keys:
        beta (1,), alpha (J,), v (J,), fc (J,), u_scale (M,)
    """
    z_beta = tf.cast(z["z_beta"], tf.float64)
    z_alpha = tf.cast(z["z_alpha"], tf.float64)
    z_v = tf.cast(z["z_v"], tf.float64)
    z_fc = tf.cast(z["z_fc"], tf.float64)
    z_u_scale = tf.cast(z["z_u_scale"], tf.float64)

    # Constrained supports
    beta = tf.sigmoid(z_beta)  # (0,1)
    alpha = tf.exp(z_alpha)  # >0
    v = tf.exp(z_v)  # >0
    fc = tf.exp(z_fc)  # >0
    u_scale = tf.exp(z_u_scale)  # >0

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "u_scale": u_scale,
    }


def _ensure_theta(theta: dict[str, tf.Tensor], u_mj: tf.Tensor) -> dict[str, tf.Tensor]:
    """Return a float64 theta dict and enforce required keys.

    Required keys and shapes:
      beta (1,), alpha (J,), v (J,), fc (J,), u_scale (M,)
    """
    required = ("beta", "alpha", "v", "fc", "u_scale")
    missing = [k for k in required if k not in theta]
    if missing:
        raise KeyError(f"theta is missing required keys: {missing}")

    theta_use = dict(theta)
    for k in required:
        theta_use[k] = tf.cast(theta_use[k], tf.float64)

    # Normalize beta to shape (1,) to match the parameterization.
    theta_use["beta"] = tf.reshape(theta_use["beta"], (1,))

    return theta_use


# =============================================================================
# Inventory grid maps
# =============================================================================

InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


def build_inventory_maps(I_max: int) -> InventoryMaps:
    """
    Precompute simple inventory-grid maps used in utility/transition calculations.

    Returns:
      I_vals:        (I,)  int32 values 0..I_max
      is_zero_mask:  (I,)  float64 mask for I==0
      stockout_mask: (I,)  float64 mask for I==0 (alias)
      at_cap_mask:   (I,)  float64 mask for I==I_max
    """
    I_vals = tf.range(I_max + 1, dtype=tf.int32)
    is_zero_mask = tf.cast(tf.equal(I_vals, 0), tf.float64)
    stockout_mask = is_zero_mask
    at_cap_mask = tf.cast(tf.equal(I_vals, I_max), tf.float64)
    return I_vals, is_zero_mask, stockout_mask, at_cap_mask


def _unpack_maps(
    maps: InventoryMaps,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    I_vals, is_zero_mask, stockout_mask, at_cap_mask = maps
    return I_vals, is_zero_mask, stockout_mask, at_cap_mask


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
    """
    Construct flow utilities u0,u1 for no-buy and buy actions.

    Inputs:
      u_mj:          (M,J) float64
      price_vals_mj: (M,J,S) float64
      theta:
        beta (1,), alpha (J,), v (J,), fc (J,), u_scale (M,)
      lambda_mn: (M,N) float64 consumption probabilities (passed, not estimated)
      waste_cost: scalar float64
      maps: inventory maps (contains stockout and at-cap masks)

    Returns:
      u0,u1: both (M,N,J,S,I) float64
    """
    _, _, stockout_mask, at_cap_mask = _unpack_maps(maps)

    u_mj = tf.cast(u_mj, tf.float64)
    price_vals_mj = tf.cast(price_vals_mj, tf.float64)
    waste_cost = tf.cast(waste_cost, tf.float64)

    # Parameters
    alpha_j = tf.cast(theta["alpha"], tf.float64)  # (J,)
    v_j = tf.cast(theta["v"], tf.float64)  # (J,)
    fc_j = tf.cast(theta["fc"], tf.float64)  # (J,)
    u_scale_m = tf.cast(theta["u_scale"], tf.float64)  # (M,)
    lambda_mn = tf.clip_by_value(tf.cast(lambda_mn, tf.float64), 0.0, 1.0)  # (M,N)

    # Dimensions
    M = tf.shape(u_mj)[0]
    J = tf.shape(u_mj)[1]
    N = tf.shape(lambda_mn)[1]
    S = tf.shape(price_vals_mj)[2]
    I = tf.shape(stockout_mask)[0]

    # Masks (broadcast-ready)
    stockout_1111I = stockout_mask[None, None, None, None, :]  # (1,1,1,1,I)
    at_cap_1111I = at_cap_mask[None, None, None, None, :]  # (1,1,1,1,I)

    # --- Buy utility u1 ---
    # u_eff(m,j) = u_scale(m) * u_mj(m,j)
    u_eff_m1j11 = (u_scale_m[:, None] * u_mj)[:, None, :, None, None]  # (M,1,J,1,1)

    # base_buy = u_eff - alpha(j) * price(s) - fc(j)
    alpha_11j11 = alpha_j[None, None, :, None, None]  # (1,1,J,1,1)
    fc_11j11 = fc_j[None, None, :, None, None]  # (1,1,J,1,1)
    price_m1js1 = price_vals_mj[:, None, :, :, None]  # (M,1,J,S,1)
    base_buy_m1js1 = u_eff_m1j11 - alpha_11j11 * price_m1js1 - fc_11j11  # (M,1,J,S,1)

    # Waste-at-cap penalty: waste_cost*(1-lambda_mn)*1{I=I_max}
    one_minus_lambda_mn = ONE_F64 - tf.clip_by_value(lambda_mn, 0.0, 1.0)  # (M,N)
    waste_term_mn111I = (
        waste_cost * one_minus_lambda_mn[:, :, None, None, None] * at_cap_1111I
    )  # (M,N,1,1,I)

    # Broadcasting produces (M,N,J,S,I)
    u1 = base_buy_m1js1 - waste_term_mn111I

    # --- No-buy utility u0 ---
    # u0 = -v(j) * 1{I=0}
    u0_11j1I = -v_j[None, None, :, None, None] * stockout_1111I  # (1,1,J,1,I)
    u0 = tf.broadcast_to(u0_11j1I, tf.stack([M, N, J, S, I]))

    return u0, u1


# =============================================================================
# Expectations over next states
# =============================================================================


def expected_over_next_price(
    V: tf.Tensor, P_price_mj: tf.Tensor, action: int
) -> tf.Tensor:
    """
    E[V(s', I) | current s] marginalizing over next price state s' using P_price_mj.

    Inputs:
      V:          (M,N,J,S,I)
      P_price_mj: (M,J,S,S)

    Returns:
      cont: (M,N,J,S,I) where cont[m,n,j,s,i] = sum_{s'} P[m,j,s,s'] * V[m,n,j,s',i]
    """
    P = tf.cast(P_price_mj, tf.float64)  # (M,J,S,S')
    V = tf.cast(V, tf.float64)  # (M,N,J,S',I)
    return tf.einsum("mjsr,mnjri->mnjsi", P, V)


def expected_over_next_inv(
    cont_s: tf.Tensor,
    p_c0_mn111: tf.Tensor,
    p_c1_mn111: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
    action: int,
) -> tf.Tensor:
    """
    E[cont_s(s', I') | current I, action] marginalizing over consumption shock c∈{0,1}.

    Inputs:
      cont_s:     (M,N,J,S,I) continuation values after price transition
      p_c0_mn111: (M,N,1,1,1)
      p_c1_mn111: (M,N,1,1,1)
      idx_down:   (I,) indices for I' when (a - c) = -1
      idx_up:     (I,) indices for I' when (a - c) = +1

    Returns:
      cont: (M,N,J,S,I)
    """
    cont_s = tf.cast(cont_s, tf.float64)

    # Gather cont at I' = I + a - c (clipped)
    if action == 0:
        # a=0: if c=0 -> I' = I ; if c=1 -> I' = I-1 (clipped)
        cont_c0 = cont_s
        cont_c1 = tf.gather(cont_s, idx_down, axis=4)
    else:
        # a=1: if c=0 -> I' = I+1 (clipped); if c=1 -> I' = I
        cont_c0 = tf.gather(cont_s, idx_up, axis=4)
        cont_c1 = cont_s

    cont = (
        tf.cast(p_c0_mn111, tf.float64) * cont_c0
        + tf.cast(p_c1_mn111, tf.float64) * cont_c1
    )
    return cont


# =============================================================================
# Logit choice-specific values and CCPs
# =============================================================================


def logsumexp_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """
    Log-sum-exp aggregator for two alternatives.

    Inputs:
      q0,q1: (M,N,J,S,I)

    Returns:
      V: (M,N,J,S,I)
    """
    m = tf.maximum(q0, q1)
    return m + tf.math.log(tf.exp(q0 - m) + tf.exp(q1 - m))


def ccp_from_q(q0: tf.Tensor, q1: tf.Tensor) -> tf.Tensor:
    """
    Logit CCP for buy action.

    Inputs:
      q0,q1: (M,N,J,S,I)

    Returns:
      ccp_buy: (M,N,J,S,I) in (0,1)
    """
    # P(a=1) = exp(q1) / (exp(q0)+exp(q1)) = sigmoid(q1-q0)
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
    """
    One Bellman update for the logit DDC.

    Shapes:
      V:           (M,N,J,S,I)
      u0,u1:       (M,N,J,S,I)
      beta:        (1,) (broadcast as a scalar)
      P_price_mj:  (M,J,S,S)
      p_c0_mn111:  (M,N,1,1,1)  P(c=0)
      p_c1_mn111:  (M,N,1,1,1)  P(c=1)
      idx_down:    (I,) integer indices for I' when (a - c) = -1
      idx_up:      (I,) integer indices for I' when (a - c) = +1

    Returns:
      V_new,q0,q1 each (M,N,J,S,I)
    """
    cont0 = expected_over_next_price(V, P_price_mj, action=0)
    cont1 = expected_over_next_price(V, P_price_mj, action=1)

    cont0 = expected_over_next_inv(
        cont0, p_c0_mn111, p_c1_mn111, idx_down, idx_up, action=0
    )
    cont1 = expected_over_next_inv(
        cont1, p_c0_mn111, p_c1_mn111, idx_down, idx_up, action=1
    )

    beta_11111 = tf.reshape(tf.cast(beta, tf.float64), (1, 1, 1, 1, 1))
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
    """
    Solve the Bellman fixed point V = T(V) via iteration.

    Shapes:
      u0,u1:       (M,N,J,S,I)
      beta:        (1,)
      P_price_mj:  (M,J,S,S)
      p_c0_mn111:  (M,N,1,1,1)
      p_c1_mn111:  (M,N,1,1,1)
      idx_down:    (I,)
      idx_up:      (I,)

    Returns:
      V: (M,N,J,S,I)
    """
    V = tf.zeros_like(u0)

    def cond_fn(i: tf.Tensor, V_prev: tf.Tensor, max_diff: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(i < max_iter, max_diff > tol)

    def body_fn(
        i: tf.Tensor, V_prev: tf.Tensor, max_diff: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        V_new, _, _ = bellman_update(
            V_prev, u0, u1, beta, P_price_mj, p_c0_mn111, p_c1_mn111, idx_down, idx_up
        )
        diff = tf.reduce_max(tf.abs(V_new - V_prev))
        return i + 1, V_new, diff

    i0 = tf.constant(0, dtype=tf.int32)
    max_diff0 = tf.constant(float("inf"), dtype=tf.float64)

    _, V_final, _ = tf.while_loop(
        cond=cond_fn,
        body=body_fn,
        loop_vars=(i0, V, max_diff0),
        maximum_iterations=max_iter,
    )
    return V_final


# =============================================================================
# Top-level CCP solver
# =============================================================================


def solve_ccp_buy(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Solve for buy CCPs on the (s,I) grid for each (m,n,j).

    Inputs:
      u_mj: (M,J)
      price_vals_mj: (M,J,S)
      P_price_mj: (M,J,S,S)
      theta: beta(1,), alpha(J,), v(J,), fc(J,), u_scale(M,)
      lambda_mn: (M,N) consumption probabilities (passed, not estimated)
      waste_cost: scalar
      maps: inventory maps

    Returns:
      ccp_buy: (M,N,J,S,I)
      q0,q1:   (M,N,J,S,I)
    """
    theta = _ensure_theta(theta, u_mj)

    I_vals, _, _, _ = _unpack_maps(maps)
    I = tf.shape(I_vals)[0]

    # Consumption probabilities
    lam_mn = tf.clip_by_value(tf.cast(lambda_mn, tf.float64), 0.0, 1.0)
    p_c1_mn111 = lam_mn[:, :, None, None, None]  # (M,N,1,1,1)
    p_c0_mn111 = (ONE_F64 - lam_mn)[:, :, None, None, None]  # (M,N,1,1,1)

    # Inventory transitions for +/-1 and 0 moves (clipped)
    idx = tf.range(I, dtype=tf.int32)
    idx_down = tf.maximum(idx - 1, 0)
    idx_up = tf.minimum(idx + 1, I - 1)

    # Flow utilities
    u0, u1 = make_flow_utilities(u_mj, price_vals_mj, theta, lam_mn, waste_cost, maps)

    # Solve V
    V_final = solve_value_function(
        u0=u0,
        u1=u1,
        beta=theta["beta"],
        P_price_mj=P_price_mj,
        p_c0_mn111=p_c0_mn111,
        p_c1_mn111=p_c1_mn111,
        idx_down=idx_down,
        idx_up=idx_up,
        tol=tol,
        max_iter=max_iter,
    )

    # Recover q0,q1 and CCPs at the fixed point
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
    ccp_buy = ccp_from_q(q0, q1)
    return ccp_buy, q0, q1


# =============================================================================
# Optional caching wrapper
# =============================================================================


@dataclass(frozen=True)
class CcpCacheKey:
    """
    Cache key for CCP computations.

    This is a best-effort key; it assumes identical tensors imply identical CCPs.
    """

    u_mj_id: int
    price_vals_mj_id: int
    P_price_mj_id: int
    theta_id: int
    lambda_mn_id: int
    waste_cost_id: int
    I_max: int


_ccp_cache: dict[CcpCacheKey, tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = {}


def solve_ccp_buy_cached(
    u_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    theta: dict[str, tf.Tensor],
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    maps: InventoryMaps,
    tol: float = 1e-6,
    max_iter: int = 2000,
    use_cache: bool = True,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Cached wrapper for solve_ccp_buy.

    Note: CCPs depend on lambda_mn; the cache key includes its object identity.

    Note: caching is only safe in pure-eager usage where object identity is meaningful.
    """
    I_vals, _, _, _ = _unpack_maps(maps)
    I_max = (
        int(I_vals.shape[0]) - 1
        if I_vals.shape.rank == 1 and I_vals.shape[0] is not None
        else -1
    )

    key = CcpCacheKey(
        u_mj_id=id(u_mj),
        price_vals_mj_id=id(price_vals_mj),
        P_price_mj_id=id(P_price_mj),
        theta_id=id(theta),
        lambda_mn_id=id(lambda_mn),
        waste_cost_id=id(waste_cost),
        I_max=I_max,
    )
    if use_cache and key in _ccp_cache:
        return _ccp_cache[key]

    out = solve_ccp_buy(
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        theta=theta,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        maps=maps,
        tol=tol,
        max_iter=max_iter,
    )
    if use_cache:
        _ccp_cache[key] = out
    return out


# =============================================================================
# Helper: forward simulation given CCPs (optional)
# =============================================================================


def simulate_purchases_given_ccp(
    ccp_buy: tf.Tensor,
    s_mjt: tf.Tensor,
    lambda_mn: tf.Tensor,
    I_max: int,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Simulate purchases and inventory paths given CCPs and observed price states.

    Inputs:
      ccp_buy: (M,N,J,S,I) buy probabilities
      s_mjt:   (M,J,T) observed price states
      lambda_mn: (M,N) consumption probabilities
      I_max: scalar int
      rng: tf.random.Generator

    Returns:
      a_mnjt: (M,N,J,T) purchases in {0,1}
      c_mnjt: (M,N,J,T) consumption in {0,1}
      I_mnjt: (M,N,J,T+1) inventory path
    """
    ccp_buy = tf.cast(ccp_buy, tf.float64)
    s_mjt = tf.cast(s_mjt, tf.int32)
    lambda_mn = tf.clip_by_value(tf.cast(lambda_mn, tf.float64), 0.0, 1.0)

    M = tf.shape(ccp_buy)[0]
    N = tf.shape(ccp_buy)[1]
    J = tf.shape(ccp_buy)[2]
    T = tf.shape(s_mjt)[2]
    I = I_max + 1

    # Initialize
    I_mnj = tf.zeros((M, N, J), dtype=tf.int32)
    I_path = tf.TensorArray(tf.int32, size=T + 1, clear_after_read=False)
    a_path = tf.TensorArray(tf.int32, size=T, clear_after_read=False)
    c_path = tf.TensorArray(tf.int32, size=T, clear_after_read=False)

    I_path = I_path.write(0, I_mnj)

    for t in tf.range(T):
        s_mj = s_mjt[:, :, t]  # (M,J)

        # Gather buy probabilities at current (s, I)
        # Need per (m,n,j) gather over s and I
        # Build indices for gather_nd on ccp_buy: [m,n,j,s,I]
        m_idx = tf.range(M)[:, None, None]
        n_idx = tf.range(N)[None, :, None]
        j_idx = tf.range(J)[None, None, :]

        m_grid = tf.broadcast_to(m_idx, (M, N, J))
        n_grid = tf.broadcast_to(n_idx, (M, N, J))
        j_grid = tf.broadcast_to(j_idx, (M, N, J))
        s_grid = tf.broadcast_to(s_mj[:, None, :], (M, N, J))
        I_grid = I_mnj

        gather_idx = tf.stack([m_grid, n_grid, j_grid, s_grid, I_grid], axis=-1)
        p_buy = tf.gather_nd(ccp_buy, gather_idx)  # (M,N,J)

        # Sample purchases
        u = rng.uniform((M, N, J), dtype=tf.float64)
        a = tf.cast(u < p_buy, tf.int32)

        # Sample consumption: independent across products j, with common lambda per (m,n)
        u_c = rng.uniform((M, N, J), dtype=tf.float64)
        c = tf.cast(u_c < lambda_mn[:, :, None], tf.int32)  # (M,N,J)

        # Update inventory: clip(I + a - c, 0, I_max)
        I_mnj = tf.clip_by_value(I_mnj + a - c, 0, I_max)

        a_path = a_path.write(t, a)
        c_path = c_path.write(t, c)
        I_path = I_path.write(t + 1, I_mnj)

    a_mnjt = tf.transpose(a_path.stack(), perm=[1, 2, 3, 0])
    c_mnjt = tf.transpose(c_path.stack(), perm=[1, 2, 3, 0])
    I_mnjt = tf.transpose(I_path.stack(), perm=[1, 2, 3, 0])
    return a_mnjt, c_mnjt, I_mnjt
