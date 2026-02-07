# ching/stockpiling_posterior.py
#
# Pure TensorFlow posterior components for the minimal Ching-style stockpiling model
# with latent inventory and observed (purchases, prices) only.
#
# No class state. No NumPy. Designed to be callable from a tf.function MCMC step.
#
# Assumption: floating-point inputs (u_m, price_vals, P_price, pi_I0, eps, tol, z_*)
# are tf.float64.

from __future__ import annotations

import tensorflow as tf


def unconstrained_to_theta(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    beta = tf.math.sigmoid(z_beta)
    alpha = tf.exp(z_alpha)
    v = tf.exp(z_v)
    fc = tf.exp(z_fc)
    lambda_c = tf.math.sigmoid(z_lambda)
    u_scale = tf.exp(z_u_scale)
    return beta, alpha, v, fc, lambda_c, u_scale


def build_transition_matrices(
    I_max: tf.Tensor,
    lambda_c: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Inventory states I in {0,...,I_max}, size I = I_max+1.

    Transition:
      I' = clip(I + a - c, 0, I_max), where c ~ Bernoulli(lambda_c).

    Returns:
      T0: (I,I) for action a=0
      T1: (I,I) for action a=1
    """
    I_max_i = tf.cast(I_max, tf.int32)
    I = I_max_i + 1

    i = tf.range(I, dtype=tf.int32)  # (I,)

    one = tf.constant(1.0, dtype=tf.float64)
    p_c1 = lambda_c
    p_c0 = one - p_c1

    shape_vec_I = tf.stack([I])  # (1,)
    shape_mat_II = tf.stack([I, I])  # (2,)

    # Action 0:
    #   c=0 -> I' = I
    #   c=1 -> I' = max(I-1, 0)
    col_stay = i
    col_down = tf.maximum(i - 1, 0)

    idx_stay = tf.stack([i, col_stay], axis=1)  # (I,2)
    idx_down = tf.stack([i, col_down], axis=1)  # (I,2)

    upd_stay0 = tf.fill(shape_vec_I, p_c0)
    upd_down0 = tf.fill(shape_vec_I, p_c1)

    T0 = tf.scatter_nd(idx_stay, upd_stay0, shape_mat_II) + tf.scatter_nd(
        idx_down, upd_down0, shape_mat_II
    )

    # Action 1:
    #   c=0 -> I' = min(I+1, I_max)
    #   c=1 -> I' = I
    col_up = tf.minimum(i + 1, I_max_i)
    idx_diag1 = tf.stack([i, i], axis=1)
    idx_up = tf.stack([i, col_up], axis=1)

    upd_diag1 = tf.fill(shape_vec_I, p_c1)
    upd_up1 = tf.fill(shape_vec_I, p_c0)

    T1 = tf.scatter_nd(idx_diag1, upd_diag1, shape_mat_II) + tf.scatter_nd(
        idx_up, upd_up1, shape_mat_II
    )

    return T0, T1


def make_flow_utilities(
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    I_max: tf.Tensor,
    alpha: tf.Tensor,
    v: tf.Tensor,
    fc: tf.Tensor,
    lambda_c: tf.Tensor,
    u_scale: tf.Tensor,
    waste_cost: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    u1(m,s,I) = u_scale * u_m[m] - alpha * price_vals[s] - fc
               - waste_cost * P(c=0) * 1{I==I_max}   (waste at capacity)
    u0(m,s,I) = -v * 1{I==0}

    Under the model timing (decision then consumption) with consumption integrated out,
    a buy at I==I_max is wasteful only when c=0, which occurs with probability
    P(c=0) = 1 - lambda_c.

    Outputs:
      u0, u1: (M,S,I)
    """
    I_max_i = tf.cast(I_max, tf.int32)
    I = I_max_i + 1

    M = tf.shape(u_m)[0]
    S = tf.shape(price_vals)[0]
    shape_msi = tf.stack([M, S, I])  # (3,)

    u_m_ms1 = tf.reshape(
        u_m, tf.stack([M, tf.constant(1, tf.int32), tf.constant(1, tf.int32)])
    )
    p_1s1 = tf.reshape(
        price_vals, tf.stack([tf.constant(1, tf.int32), S, tf.constant(1, tf.int32)])
    )

    # Buy utility (base, inventory-invariant).
    u1_ms1 = u_scale * u_m_ms1 - alpha * p_1s1 - fc
    u1 = tf.broadcast_to(u1_ms1, shape_msi)

    # Waste penalty when buying at I==I_max and c=0 (integrated out).
    one = tf.constant(1.0, dtype=tf.float64)
    p_c0 = one - lambda_c

    I_grid = tf.range(I, dtype=tf.int32)
    at_cap = tf.cast(tf.equal(I_grid, I_max_i), tf.float64)  # (I,)
    at_cap_1_1_I = tf.reshape(
        at_cap,
        tf.stack([tf.constant(1, tf.int32), tf.constant(1, tf.int32), I]),
    )
    waste_pen_1_1_I = waste_cost * p_c0 * at_cap_1_1_I
    u1 = u1 - waste_pen_1_1_I  # broadcast over (M,S)

    # No-buy utility: stockout penalty.
    stockout = tf.cast(tf.equal(I_grid, 0), tf.float64)  # (I,)
    stockout_1_1_I = tf.reshape(
        stockout,
        tf.stack([tf.constant(1, tf.int32), tf.constant(1, tf.int32), I]),
    )
    u0 = tf.broadcast_to(-v * stockout_1_1_I, shape_msi)

    return u0, u1


def expected_over_next_price(V: tf.Tensor, P_price: tf.Tensor) -> tf.Tensor:
    return tf.einsum("ab,mbi->mai", P_price, V)


def _logsumexp2(x0: tf.Tensor, x1: tf.Tensor) -> tf.Tensor:
    m = tf.maximum(x0, x1)
    return m + tf.math.log(tf.exp(x0 - m) + tf.exp(x1 - m))


def bellman_update(
    V: tf.Tensor,
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta: tf.Tensor,
    T0: tf.Tensor,
    T1: tf.Tensor,
    P_price: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    One Bellman update.

    Returns:
      V_new, Q0, Q1  all (M,S,I)
    """
    EV_next = expected_over_next_price(V, P_price)  # (M,S,I)

    cont0 = tf.einsum("msj,ij->msi", EV_next, T0)
    cont1 = tf.einsum("msj,ij->msi", EV_next, T1)

    Q0 = u0 + beta * cont0
    Q1 = u1 + beta * cont1

    V_new = _logsumexp2(Q0, Q1)
    return V_new, Q0, Q1


def solve_value_function(
    u0: tf.Tensor,
    u1: tf.Tensor,
    beta: tf.Tensor,
    T0: tf.Tensor,
    T1: tf.Tensor,
    P_price: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
) -> tf.Tensor:
    """
    Value iteration.

    Returns:
      V_final: (M,S,I)
    """
    max_iter_i = tf.cast(max_iter, tf.int32)

    V0 = tf.zeros_like(u0)
    diff0 = tf.constant(1.0e30, dtype=tf.float64)

    def cond(it: tf.Tensor, V_curr: tf.Tensor, diff_curr: tf.Tensor) -> tf.Tensor:
        return tf.logical_and(it < max_iter_i, diff_curr > tol)

    def body(it: tf.Tensor, V_curr: tf.Tensor, diff_curr: tf.Tensor):
        V_new, _, _ = bellman_update(V_curr, u0, u1, beta, T0, T1, P_price)
        diff_new = tf.reduce_max(tf.abs(V_new - V_curr))
        return it + 1, V_new, diff_new

    it0 = tf.constant(0, tf.int32)
    _, V_final, _ = tf.while_loop(
        cond,
        body,
        loop_vars=[it0, V0, diff0],
        shape_invariants=[
            it0.get_shape(),
            tf.TensorShape([None, None, None]),
            diff0.get_shape(),
        ],
    )
    return V_final


def solve_ccp_buy(
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    I_max: tf.Tensor,
    beta: tf.Tensor,
    alpha: tf.Tensor,
    v: tf.Tensor,
    fc: tf.Tensor,
    lambda_c: tf.Tensor,
    u_scale: tf.Tensor,
    waste_cost: tf.Tensor,
    T0: tf.Tensor,
    T1: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
) -> tf.Tensor:
    """
    Solve DP and return buy CCP table.

    Returns:
      ccp_buy: (M,S,I)
    """
    u0, u1 = make_flow_utilities(
        u_m,
        price_vals,
        I_max,
        alpha,
        v,
        fc,
        lambda_c,
        u_scale,
        waste_cost,
    )
    V_final = solve_value_function(u0, u1, beta, T0, T1, P_price, tol, max_iter)

    # Compute Q0,Q1 at the converged V (avoids the "one-iteration-lag" issue).
    _, Q0, Q1 = bellman_update(V_final, u0, u1, beta, T0, T1, P_price)
    denom = _logsumexp2(Q0, Q1)
    return tf.exp(Q1 - denom)


def initial_inventory_belief(
    pi_I0: tf.Tensor,
    M: tf.Tensor,
    N: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Returns:
      b0: (M,N,I)
    """
    I = tf.shape(pi_I0)[0]
    shape_mni = tf.stack(
        [tf.cast(M, tf.int32), tf.cast(N, tf.int32), tf.cast(I, tf.int32)]
    )

    pi_1_1_I = tf.reshape(
        pi_I0,
        tf.stack(
            [tf.constant(1, tf.int32), tf.constant(1, tf.int32), tf.cast(I, tf.int32)]
        ),
    )
    b0 = tf.broadcast_to(pi_1_1_I, shape_mni)

    denom = tf.reduce_sum(b0, axis=2, keepdims=True)
    return b0 / tf.maximum(denom, eps)


def select_pi_by_state(ccp_buy: tf.Tensor, s_mt: tf.Tensor) -> tf.Tensor:
    """
    Returns:
      pi_mI: (M,I) with pi_mI[m,:] = ccp_buy[m, s_mt[m], :]
    """
    return tf.gather(ccp_buy, tf.cast(s_mt, tf.int32), axis=1, batch_dims=1)


def action_likelihood_by_inventory(pi_mI: tf.Tensor, a_mn: tf.Tensor) -> tf.Tensor:
    """
    Returns:
      lik_I: (M,N,I) = P(a_mn | I, s_t)
    """
    a = tf.cast(a_mn, tf.float64)
    pi = tf.expand_dims(pi_mI, axis=1)  # (M,1,I)
    one = tf.constant(1.0, dtype=tf.float64)
    return a[:, :, None] * pi + (one - a[:, :, None]) * (one - pi)


def bayes_update_belief(
    b: tf.Tensor,
    lik_I: tf.Tensor,
    eps: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Returns:
      b_post: (M,N,I)
      ll_step: scalar
    """
    unnorm = b * lik_I
    pred = tf.reduce_sum(unnorm, axis=2)  # (M,N)
    pred_safe = pred + eps
    b_post = unnorm / tf.expand_dims(pred_safe, axis=2)
    ll_step = tf.reduce_sum(tf.math.log(pred_safe))
    return b_post, ll_step


def transition_belief(
    b_post: tf.Tensor,
    a_mn: tf.Tensor,
    T0: tf.Tensor,
    T1: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Returns:
      b_next: (M,N,I)
    """
    a = tf.cast(a_mn, tf.float64)
    one = tf.constant(1.0, dtype=tf.float64)

    b0 = tf.einsum("mni,ij->mnj", b_post, T0)
    b1 = tf.einsum("mni,ij->mnj", b_post, T1)

    b_next = (one - a)[:, :, None] * b0 + a[:, :, None] * b1
    denom = tf.reduce_sum(b_next, axis=2, keepdims=True)
    return b_next / tf.maximum(denom, eps)


def loglik_hidden_inventory(
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    ccp_buy: tf.Tensor,
    pi_I0: tf.Tensor,
    T0: tf.Tensor,
    T1: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """
    Forward filter integrating out latent inventory.

    Returns:
      loglik: scalar
    """
    M = tf.shape(a_imt)[0]
    N = tf.shape(a_imt)[1]
    T = tf.shape(a_imt)[2]

    b0 = initial_inventory_belief(pi_I0, M, N, eps)
    ll0 = tf.constant(0.0, dtype=tf.float64)

    def cond(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor) -> tf.Tensor:
        return t < T

    def body(t: tf.Tensor, b_curr: tf.Tensor, ll_curr: tf.Tensor):
        s_mt = p_state_mt[:, t]  # (M,)
        a_mn = a_imt[:, :, t]  # (M,N)

        pi_mI = select_pi_by_state(ccp_buy, s_mt)  # (M,I)
        lik_I = action_likelihood_by_inventory(pi_mI, a_mn)  # (M,N,I)

        b_post, ll_step = bayes_update_belief(b_curr, lik_I, eps)
        b_next = transition_belief(b_post, a_mn, T0, T1, eps)
        return t + 1, b_next, ll_curr + ll_step

    t0 = tf.constant(0, tf.int32)
    _, _, ll_final = tf.while_loop(
        cond,
        body,
        loop_vars=[t0, b0, ll0],
        shape_invariants=[
            t0.get_shape(),
            tf.TensorShape([None, None, None]),
            ll0.get_shape(),
        ],
    )
    return ll_final


def logprior_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
    sigma_beta: tf.Tensor,
    sigma_alpha: tf.Tensor,
    sigma_v: tf.Tensor,
    sigma_fc: tf.Tensor,
    sigma_lambda: tf.Tensor,
    sigma_u_scale: tf.Tensor,
) -> tf.Tensor:
    """
    Independent Normal(0, sigma^2) priors on unconstrained parameters z_*,
    up to an additive constant.
    """

    def quad(z: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.square(z / s)

    return (
        quad(z_beta, sigma_beta)
        + quad(z_alpha, sigma_alpha)
        + quad(z_v, sigma_v)
        + quad(z_fc, sigma_fc)
        + quad(z_lambda, sigma_lambda)
        + quad(z_u_scale, sigma_u_scale)
    )


def logpost(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    I_max: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    sigma_beta: tf.Tensor,
    sigma_alpha: tf.Tensor,
    sigma_v: tf.Tensor,
    sigma_fc: tf.Tensor,
    sigma_lambda: tf.Tensor,
    sigma_u_scale: tf.Tensor,
) -> tf.Tensor:
    """
    logpost = loglik(a | p, theta, u_m) + logprior(z)

    Notes:
      - EV1 scale is normalized to 1 in the DP/CCP mapping.
      - u_scale rescales the fixed u_m passed in from Phase 2 into the Phase-3 EV1 scale.
      - waste_cost penalizes buying at I==I_max when c=0 (integrated out via lambda_c).
    """
    beta, alpha, v, fc, lambda_c, u_scale = unconstrained_to_theta(
        z_beta, z_alpha, z_v, z_fc, z_lambda, z_u_scale
    )

    # Build transitions once and reuse for DP + filtering.
    T0, T1 = build_transition_matrices(I_max, lambda_c)

    ccp_buy = solve_ccp_buy(
        u_m,
        price_vals,
        P_price,
        I_max,
        beta,
        alpha,
        v,
        fc,
        lambda_c,
        u_scale,
        waste_cost,
        T0,
        T1,
        tol,
        max_iter,
    )

    ll = loglik_hidden_inventory(
        a_imt,
        p_state_mt,
        ccp_buy,
        pi_I0,
        T0,
        T1,
        eps,
    )

    lp = logprior_z(
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_lambda,
        z_u_scale,
        sigma_beta,
        sigma_alpha,
        sigma_v,
        sigma_fc,
        sigma_lambda,
        sigma_u_scale,
    )

    return ll + lp
