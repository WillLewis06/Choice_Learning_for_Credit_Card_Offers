"""
Ching-style stockpiling DGP (single-product, seller-observed).

Upstream truth inputs:
  - delta_true (J,)
  - E_bar_true (M,)
  - njt_true   (M,J)
  - product_index (int)

Market intercept for the chosen product:
  u_m_true[m] = delta_true[product_index] + E_bar_true[m] + njt_true[m, product_index]

Seller-observed outputs:
  - a_imt      (M, N, T)  purchases (0/1, stored as int64)
  - p_state_mt (M, T)     price states (int64)
  - u_m_true   (M,)       fixed market intercepts for the chosen product

Model conventions:
  - latent inventory carried across markets (same consumers across market blocks)
  - price chain starts from a random initial state, then continues across markets
  - waste-at-cap penalty in the DP: waste_cost * (1 - lambda_c) * 1{I == I_max}
  - EV1 shocks scale normalized to 1 -> log-sum-exp value and softmax CCPs

Validation is assumed to happen elsewhere.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Helpers
# =============================================================================


def logsumexp2(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Elementwise log(exp(x0) + exp(x1)) with basic stabilization."""
    m = np.maximum(x0, x1)
    return m + np.log(np.exp(x0 - m) + np.exp(x1 - m))


def compute_u_m(
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
    product_index: int,
) -> np.ndarray:
    """u_m_true[m] = delta_true[j*] + E_bar_true[m] + njt_true[m, j*]."""
    return delta_true[product_index] + E_bar_true + njt_true[:, product_index]


def simulate_price_states(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
    start_state: int,
) -> np.ndarray:
    """
    Simulate a price-state Markov chain of length T.

    Returns:
      p_state_t: (T,) int in {0,...,S-1}
    """
    cdf = np.cumsum(P_price, axis=1)
    p_state_t = np.empty((T,), dtype=np.int64)
    p_state_t[0] = start_state

    u = rng.random(T - 1)
    for t in range(1, T):
        prev = p_state_t[t - 1]
        p_state_t[t] = np.searchsorted(cdf[prev], u[t - 1], side="right")

    return p_state_t


def simulate_consumption(
    rng: np.random.Generator,
    N: int,
    T: int,
    lambda_c: float,
) -> np.ndarray:
    """
    c_it ~ Bernoulli(lambda_c), shape (N,T), boolean array.
    """
    return rng.random((N, T)) < lambda_c


def next_inventory(
    I: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    I_max: int,
) -> np.ndarray:
    """I_next = clip(I + a - c, 0, I_max)."""
    I_next = I + a - c
    return np.clip(I_next, 0, I_max)


# =============================================================================
# DP solver (single market): returns CCP table
# =============================================================================


def solve_ccp_buy(
    u_m: float,
    beta: float,
    alpha: float,
    v: float,
    fc: float,
    lambda_c: float,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Solve the DP for one market and return buy CCPs:

      ccp_buy[s, I] = P(a=1 | s, I)

    Shapes:
      - P_price: (S,S)
      - price_vals: (S,)
      - ccp_buy: (S, I_max+1)
    """
    S = P_price.shape[0]
    I_size = I_max + 1

    I_grid = np.arange(I_size, dtype=np.int64)

    # Inventory index transitions for c in {0,1}
    I_a0_c0 = I_grid
    I_a0_c1 = np.maximum(I_grid - 1, 0)
    I_a1_c0 = np.minimum(I_grid + 1, I_max)
    I_a1_c1 = I_grid

    # Flow utilities (S, I)
    u0 = (-v * (I_grid == 0))[None, :]  # (1, I) -> broadcast to (S, I)

    base_buy_s = u_m - alpha * price_vals - fc  # (S,)
    u1 = base_buy_s[:, None] + np.zeros((S, I_size), dtype=np.float64)

    # Waste-at-cap penalty: waste_cost * P(c=0) * 1{I==I_max}
    u1 = u1 - (waste_cost * (1.0 - lambda_c)) * (I_grid == I_max)[None, :]

    V = np.zeros((S, I_size), dtype=np.float64)

    for _ in range(max_iter):
        EV_next = P_price @ V  # (S, I)

        cont0 = (1.0 - lambda_c) * EV_next[:, I_a0_c0] + lambda_c * EV_next[:, I_a0_c1]
        cont1 = (1.0 - lambda_c) * EV_next[:, I_a1_c0] + lambda_c * EV_next[:, I_a1_c1]

        Q0 = u0 + beta * cont0
        Q1 = u1 + beta * cont1

        V_new = logsumexp2(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    # CCPs evaluated at converged V (no one-iteration lag)
    EV_next = P_price @ V
    cont0 = (1.0 - lambda_c) * EV_next[:, I_a0_c0] + lambda_c * EV_next[:, I_a0_c1]
    cont1 = (1.0 - lambda_c) * EV_next[:, I_a1_c0] + lambda_c * EV_next[:, I_a1_c1]
    Q0 = u0 + beta * cont0
    Q1 = u1 + beta * cont1
    denom = logsumexp2(Q0, Q1)

    return np.exp(Q1 - denom)


# =============================================================================
# Generator
# =============================================================================


def generate_dgp(
    seed: int,
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
    product_index: int,
    N: int,
    T: int,
    theta_true: dict[str, float],
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Generate seller-observed data for the stockpiling model.

    Returns:
      a_imt      (M, N, T)
      p_state_mt (M, T)
      u_m_true   (M,)
      theta_true (dict)
    """
    rng = np.random.default_rng(seed)

    M = E_bar_true.shape[0]
    S = P_price.shape[0]

    u_m_true = compute_u_m(delta_true, E_bar_true, njt_true, product_index)

    beta = theta_true["beta"]
    alpha = theta_true["alpha"]
    v = theta_true["v"]
    fc = theta_true["fc"]
    lambda_c = theta_true["lambda_c"]

    p_state_mt = np.zeros((M, T), dtype=np.int64)
    a_imt = np.zeros((M, N, T), dtype=np.int64)

    # Consumers persist across markets: initialize once, carry inventory through all markets.
    I_curr = rng.integers(0, I_max + 1, size=N, dtype=np.int64)

    # Price chain starts random, then continues across markets.
    s0 = rng.integers(0, S)

    for m in range(M):
        s_t = simulate_price_states(rng, P_price, T, s0)
        p_state_mt[m] = s_t
        s0 = s_t[-1]

        ccp_buy = solve_ccp_buy(
            u_m=u_m_true[m],
            beta=beta,
            alpha=alpha,
            v=v,
            fc=fc,
            lambda_c=lambda_c,
            I_max=I_max,
            P_price=P_price,
            price_vals=price_vals,
            waste_cost=waste_cost,
            tol=tol,
            max_iter=max_iter,
        )

        c_block = simulate_consumption(rng, N, T, lambda_c)

        for t in range(T):
            s = s_t[t]
            prob_buy = ccp_buy[s, I_curr]
            a_t = rng.random(N) < prob_buy  # boolean

            a_imt[m, :, t] = a_t
            I_curr = next_inventory(I_curr, a_t, c_block[:, t], I_max)

    return a_imt, p_state_mt, u_m_true, theta_true
