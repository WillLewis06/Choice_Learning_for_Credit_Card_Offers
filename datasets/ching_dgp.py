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
  - a_imt      (M, N, T) purchases (0/1, stored as int64)
  - p_state_mt (M, T)    price states (int64)
  - u_m_true   (M,)      fixed market intercepts for the chosen product

Model conventions:
  - markets are independent
  - consumers are independent within each market
  - consumer parameters are sampled independently per (market, consumer)
  - inventory and consumption are latent (unobserved by seller)
"""

from __future__ import annotations

import numpy as np

from ching.stockpiling_input_validation import validate_stockpiling_dgp_inputs


def _logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# One draw per (market, consumer) on z-scale; then transform:
#   beta, lambda_c: sigmoid(z)
#   alpha, v, fc:   exp(z)
theta_z_hypers: dict[str, dict[str, float]] = {
    "beta": {"mu": _logit(0.71), "sd": 0.35},
    "alpha": {"mu": float(np.log(0.27)), "sd": 0.50},
    "v": {"mu": float(np.log(0.48)), "sd": 0.50},
    "fc": {"mu": float(np.log(1.83)), "sd": 0.50},
    "lambda_c": {"mu": _logit(0.30), "sd": 0.35},
}


def sample_theta_true(
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
    keys = ("beta", "alpha", "v", "fc", "lambda_c")
    z = {
        k: rng.normal(theta_z_hypers[k]["mu"], theta_z_hypers[k]["sd"], size=(M, N))
        for k in keys
    }

    out: dict[str, np.ndarray] = {}

    # Sigmoid transforms
    for k in ("beta", "lambda_c"):
        out[k] = _sigmoid(z[k]).astype(np.float64, copy=False)

    # Exp transforms
    for k in ("alpha", "v", "fc"):
        out[k] = np.exp(z[k]).astype(np.float64, copy=False)

    return out


def logsumexp2(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    m = np.maximum(x0, x1)
    return m + np.log(np.exp(x0 - m) + np.exp(x1 - m))


def compute_u_m(
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
    product_index: int,
) -> np.ndarray:
    return delta_true[product_index] + E_bar_true + njt_true[:, product_index]


def simulate_price_states(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
    start_state: int,
) -> np.ndarray:
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
    u = rng.random((N, T))
    return u < lambda_c_n[:, None]


def next_inventory(
    I: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    I_max: int,
) -> np.ndarray:
    I_next = I + a - c
    return np.clip(I_next, 0, I_max)


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

    Returns:
      ccp_buy: (S, I_max+1)
    """
    S = P_price.shape[0]
    I_size = I_max + 1

    I_grid = np.arange(I_size, dtype=np.int64)

    # Inventory index transitions for c in {0,1}
    I_a0_c0 = I_grid
    I_a0_c1 = np.maximum(I_grid - 1, 0)
    I_a1_c0 = np.minimum(I_grid + 1, I_max)
    I_a1_c1 = I_grid

    # Flow utilities
    u0 = (-v * (I_grid == 0))[None, :]
    base_buy_s = u_m - alpha * price_vals - fc
    u1 = base_buy_s[:, None]

    # Waste-at-cap penalty: waste_cost * P(c=0) * 1{I == I_max}
    u1 = u1 - (waste_cost * (1.0 - lambda_c)) * (I_grid == I_max)[None, :]

    V = np.zeros((S, I_size), dtype=np.float64)

    for _ in range(max_iter):
        EV_next = P_price @ V

        cont0 = (1.0 - lambda_c) * EV_next[:, I_a0_c0] + lambda_c * EV_next[:, I_a0_c1]
        cont1 = (1.0 - lambda_c) * EV_next[:, I_a1_c0] + lambda_c * EV_next[:, I_a1_c1]

        Q0 = u0 + beta * cont0
        Q1 = u1 + beta * cont1

        V_new = logsumexp2(Q0, Q1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    EV_next = P_price @ V
    cont0 = (1.0 - lambda_c) * EV_next[:, I_a0_c0] + lambda_c * EV_next[:, I_a0_c1]
    cont1 = (1.0 - lambda_c) * EV_next[:, I_a1_c0] + lambda_c * EV_next[:, I_a1_c1]
    Q0 = u0 + beta * cont0
    Q1 = u1 + beta * cont1
    denom = logsumexp2(Q0, Q1)

    return np.exp(Q1 - denom)


def simulate_market_price_path(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
) -> np.ndarray:
    S = int(P_price.shape[0])
    start_state = int(rng.integers(0, S))
    return simulate_price_states(rng=rng, P_price=P_price, T=T, start_state=start_state)


def solve_market_ccps(
    u_m: float,
    beta_mn: np.ndarray,
    alpha_mn: np.ndarray,
    v_mn: np.ndarray,
    fc_mn: np.ndarray,
    lambda_c_mn: np.ndarray,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    N = int(beta_mn.shape[0])
    S = int(P_price.shape[0])
    ccp_buy_n_s_i = np.zeros((N, S, I_max + 1), dtype=np.float64)

    for n in range(N):
        ccp_buy_n_s_i[n] = solve_ccp_buy(
            u_m=float(u_m),
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

    return ccp_buy_n_s_i


def simulate_market_panel(
    rng: np.random.Generator,
    s_t: np.ndarray,
    ccp_buy_n_s_i: np.ndarray,
    lambda_c_mn: np.ndarray,
    I_init: np.ndarray,
    I_max: int,
) -> np.ndarray:
    """
    Simulate (N,T) purchases for one market given:
      - price states s_t (T,)
      - per-consumer CCPs ccp_buy_n_s_i (N,S,I)
      - per-consumer consumption probabilities lambda_c_mn (N,)
      - initial inventory I_init (N,)
    """
    N = int(I_init.shape[0])
    T = int(s_t.shape[0])

    a_nt = np.zeros((N, T), dtype=np.int64)

    I_curr = I_init.copy()
    c_block = simulate_consumption(
        rng=rng, N=N, T=T, lambda_c_n=lambda_c_mn.astype(np.float64)
    )

    n_idx = np.arange(N)
    for t in range(T):
        s = int(s_t[t])
        prob_buy = ccp_buy_n_s_i[n_idx, s, I_curr]
        a_t = rng.random(N) < prob_buy

        a_nt[:, t] = a_t.astype(np.int64)

        I_curr = next_inventory(I_curr, a_t, c_block[:, t], I_max)

    return a_nt


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
    Generate seller-observed stockpiling data.

    Returns:
      a_imt      (M, N, T)
      p_state_mt (M, T)
      u_m_true   (M,)
      theta_true dict of (M,N) arrays
    """
    rng = np.random.default_rng(seed)

    M = int(E_bar_true.shape[0])

    theta_true = sample_theta_true(rng=rng, M=M, N=N)

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

    beta_all = theta_true["beta"]
    alpha_all = theta_true["alpha"]
    v_all = theta_true["v"]
    fc_all = theta_true["fc"]
    lambda_all = theta_true["lambda_c"]

    p_state_mt = np.zeros((M, T), dtype=np.int64)
    a_imt = np.zeros((M, N, T), dtype=np.int64)

    for m in range(M):
        s_t = simulate_market_price_path(rng=rng, P_price=P_price, T=T)
        p_state_mt[m] = s_t

        I_init = rng.integers(0, I_max + 1, size=N, dtype=np.int64)

        beta_mn = beta_all[m]
        alpha_mn = alpha_all[m]
        v_mn = v_all[m]
        fc_mn = fc_all[m]
        lambda_c_mn = lambda_all[m]

        ccp_buy_n_s_i = solve_market_ccps(
            u_m=float(u_m_true[m]),
            beta_mn=beta_mn,
            alpha_mn=alpha_mn,
            v_mn=v_mn,
            fc_mn=fc_mn,
            lambda_c_mn=lambda_c_mn,
            I_max=I_max,
            P_price=P_price,
            price_vals=price_vals,
            waste_cost=waste_cost,
            tol=tol,
            max_iter=max_iter,
        )

        a_imt[m] = simulate_market_panel(
            rng=rng,
            s_t=s_t,
            ccp_buy_n_s_i=ccp_buy_n_s_i,
            lambda_c_mn=lambda_c_mn,
            I_init=I_init,
            I_max=I_max,
        )

    return a_imt, p_state_mt, u_m_true, theta_true
