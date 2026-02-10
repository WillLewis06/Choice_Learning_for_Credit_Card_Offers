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
  - markets are independent
    - consumers (latent inventories) are generated independently per market (inventory resets per market)
    - price chain is simulated independently per market (fresh random start state per market)
  - Ching parameters are consumer-specific in this DGP:
    - one draw per (market, consumer) for each parameter
    - theta_true is a dict of arrays, each shape (M, N): beta, alpha, v, fc, lambda_c
  - waste-at-cap penalty in the DP: waste_cost * (1 - lambda_c) * 1{I == I_max}
  - EV1 shocks scale normalized to 1 -> log-sum-exp value and softmax CCPs
  - u_scale is estimator-only (utility rescaling during estimation) and is not used in the DGP

Validation is assumed to happen elsewhere.
"""

from __future__ import annotations

import numpy as np

from ching.stockpiling_input_validation import validate_stockpiling_dgp_inputs

# =============================================================================
# DGP hyperparameters (z-scale Normal) for per-(market, consumer) theta sampling
# =============================================================================


def _logit(p: float) -> float:
    """logit(p) = log(p / (1-p)) for scalar p in (0,1)."""
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    # Clip only to prevent overflow in exp for extreme inputs.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# One draw per (market, consumer) on z-scale; then transform:
#   beta, lambda_c: sigmoid(z)
#   alpha, v, fc:   exp(z)
#
# Means are centered using Ching/Osborne empirical magnitudes where appropriate.
# lambda_c is a design choice in this simplified Bernoulli-consumption DGP.
THETA_Z_HYPERS: dict[str, dict[str, float]] = {
    "beta": {"mu": _logit(0.71), "sd": 0.35},
    "alpha": {"mu": float(np.log(0.27)), "sd": 0.50},
    "v": {"mu": float(np.log(0.48)), "sd": 0.50},
    "fc": {"mu": float(np.log(1.83)), "sd": 0.50},
    "lambda_c": {"mu": _logit(0.30), "sd": 0.35},
}


def sample_theta_true_consumer_independent(
    rng: np.random.Generator,
    M: int,
    N: int,
) -> dict[str, np.ndarray]:
    """
    Sample one parameter draw per (market, consumer) on the unconstrained (z) scale,
    then transform to constrained space.

    Returns:
      theta_true dict with keys: beta, alpha, v, fc, lambda_c
      Each value is float64 array of shape (M, N).
    """
    # Sample z arrays (M, N)
    z_beta = rng.normal(
        THETA_Z_HYPERS["beta"]["mu"], THETA_Z_HYPERS["beta"]["sd"], size=(M, N)
    )
    z_alpha = rng.normal(
        THETA_Z_HYPERS["alpha"]["mu"], THETA_Z_HYPERS["alpha"]["sd"], size=(M, N)
    )
    z_v = rng.normal(THETA_Z_HYPERS["v"]["mu"], THETA_Z_HYPERS["v"]["sd"], size=(M, N))
    z_fc = rng.normal(
        THETA_Z_HYPERS["fc"]["mu"], THETA_Z_HYPERS["fc"]["sd"], size=(M, N)
    )
    z_lambda = rng.normal(
        THETA_Z_HYPERS["lambda_c"]["mu"],
        THETA_Z_HYPERS["lambda_c"]["sd"],
        size=(M, N),
    )

    # Transform to constrained space (M, N)
    beta = _sigmoid(z_beta).astype(np.float64, copy=False)
    lambda_c = _sigmoid(z_lambda).astype(np.float64, copy=False)

    alpha = np.exp(z_alpha).astype(np.float64, copy=False)
    v = np.exp(z_v).astype(np.float64, copy=False)
    fc = np.exp(z_fc).astype(np.float64, copy=False)

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "lambda_c": lambda_c,
    }


def sample_theta_true_market_broadcast(
    rng: np.random.Generator,
    M: int,
    N: int,
) -> dict[str, np.ndarray]:
    """
    Backwards-compatible name.

    Previously: sampled one draw per market and broadcast across N consumers.
    Now: consumer-independent sampling (one draw per (m,n)).
    """
    return sample_theta_true_consumer_independent(rng=rng, M=M, N=N)


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
    lambda_c_n: np.ndarray,
) -> np.ndarray:
    """
    c_int ~ Bernoulli(lambda_c_n[i]) independently over i,t.

    Inputs:
      lambda_c_n: (N,) per-consumer consumption probability in (0,1)

    Returns:
      c_block: (N,T) boolean array
    """
    u = rng.random((N, T))
    return u < lambda_c_n[:, None]


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
# DP solver (single consumer in a single market): returns CCP table
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
    Solve the DP for one consumer in one market and return buy CCPs:

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
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate seller-observed data for the stockpiling model.

    This DGP samples consumer-specific parameters internally:
      - one draw per (market, consumer) for each parameter (beta, alpha, v, fc, lambda_c)

    Returns:
      a_imt      (M, N, T)
      p_state_mt (M, T)
      u_m_true   (M,)
      theta_true (dict of (M,N) arrays)
    """
    rng = np.random.default_rng(seed)

    M = int(E_bar_true.shape[0])
    S = int(P_price.shape[0])

    # Sample per-(market, consumer) theta (shape (M,N)).
    theta_true = sample_theta_true_consumer_independent(rng=rng, M=M, N=N)

    validate_stockpiling_dgp_inputs(
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        product_index=product_index,
        N=N,
        T=T,
        theta_true=theta_true,
        I_max=I_max,
        P_price=P_price,
        price_vals=price_vals,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
    )

    u_m_true = compute_u_m(delta_true, E_bar_true, njt_true, product_index)

    beta_all = theta_true["beta"]  # (M,N)
    alpha_all = theta_true["alpha"]  # (M,N)
    v_all = theta_true["v"]  # (M,N)
    fc_all = theta_true["fc"]  # (M,N)
    lambda_all = theta_true["lambda_c"]  # (M,N)

    p_state_mt = np.zeros((M, T), dtype=np.int64)
    a_imt = np.zeros((M, N, T), dtype=np.int64)

    for m in range(M):
        # Market-specific price chain (independent across markets).
        start_state = int(rng.integers(0, S))
        s_t = simulate_price_states(rng, P_price, T, start_state)
        p_state_mt[m] = s_t

        # Independent consumers per market: initialize inventory fresh per market.
        I_curr = rng.integers(0, I_max + 1, size=N, dtype=np.int64)

        # Consumer-specific parameters for this market (shape (N,)).
        beta_mn = beta_all[m]
        alpha_mn = alpha_all[m]
        v_mn = v_all[m]
        fc_mn = fc_all[m]
        lambda_c_mn = lambda_all[m]

        # Solve DP/CCPs per consumer (simple and legible; not optimized).
        ccp_buy_nsi = np.zeros((N, S, I_max + 1), dtype=np.float64)
        for n in range(N):
            ccp_buy_nsi[n] = solve_ccp_buy(
                u_m=float(u_m_true[m]),
                beta=float(beta_mn[n]),
                alpha=float(alpha_mn[n]),
                v=float(v_mn[n]),
                fc=float(fc_mn[n]),
                lambda_c=float(lambda_c_mn[n]),
                I_max=I_max,
                P_price=P_price,
                price_vals=price_vals,
                waste_cost=waste_cost,
                tol=tol,
                max_iter=max_iter,
            )

        # Consumer-specific consumption shocks (independent across i,t).
        c_block = simulate_consumption(rng, N, T, lambda_c_mn.astype(np.float64))

        for t in range(T):
            s = s_t[t]

            # prob_buy[n] = ccp_buy_nsi[n, s, I_curr[n]]
            pi_nI = ccp_buy_nsi[:, s, :]  # (N, I)
            prob_buy = pi_nI[np.arange(N), I_curr]  # (N,)

            a_t = rng.random(N) < prob_buy  # boolean
            a_imt[m, :, t] = a_t

            I_curr = next_inventory(I_curr, a_t, c_block[:, t], I_max)

    return a_imt, p_state_mt, u_m_true, theta_true
