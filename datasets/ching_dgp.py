"""
Ching-style stockpiling DGP (multi-product, seller-observed).

This DGP implements the Phase-3 model where:
  - all products j=1..J are modelled in the stockpiling layer
  - each (market, product) pair has its own exogenous Markov price-state process
  - price-state processes are simulated independently across (market, product) pairs (by construction)

Upstream truth inputs (from Phase 1–2):
  - delta_true (J,) product baseline utilities
  - E_bar_true (M,) market shocks
  - njt_true   (M,J) market-product shocks

Phase-3 market-product intercept passed into the stockpiling layer:
  u_mj_true[m, j] = delta_true[j] + E_bar_true[m] + njt_true[m, j]

Seller-observed outputs:
  - a_mnjt      (M, N, J, T) purchases (0/1, stored as int64)
  - p_state_mjt (M, J, T)    price states (int64)
  - u_mj_true   (M, J)       fixed market-product intercepts (float64)

True parameters (DGP draws):
  - per (market, product) (M,J): beta, alpha, v, fc
  - per (market, consumer) (M,N): lambda   (note: renamed from lambda_c)

Model conventions:
  - markets are independent
  - consumers are independent within each market
  - inventory and consumption are latent (unobserved by seller)
  - each product has its own latent inventory process per consumer
  - lambda_{m,n} is shared across products, but consumption draws are independent across (j,t)

Implementation note:
  This file assumes a common number of price states S across market-product pairs:
    P_price_mj    has shape (M, J, S, S)
    price_vals_mj has shape (M, J, S)
"""

from __future__ import annotations

import numpy as np

from ching.stockpiling_input_validation import validate_stockpiling_dgp_inputs


def _logit(p: float) -> float:
    """Logit transform for p in (0,1)."""
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid applied elementwise."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# Draw on an unconstrained z-scale then transform:
#   beta, lambda: sigmoid(z)
#   alpha, v, fc
theta_z_hypers: dict[str, dict[str, float]] = {
    "beta": {"mu": _logit(0.71), "sd": 0.35},
    "alpha": {"mu": float(np.log(0.27)), "sd": 0.50},
    "v": {"mu": float(np.log(0.48)), "sd": 0.50},
    "fc": {"mu": float(np.log(1.83)), "sd": 0.50},
    "lambda": {"mu": _logit(0.30), "sd": 0.35},
}


def sample_theta_true(
    rng: np.random.Generator,
    M: int,
    N: int,
    J: int,
) -> dict[str, np.ndarray]:
    """
    Sample true parameters under the target heterogeneity structure.

    Returns:
      dict with:
        - beta, alpha, v, fc: (M,J) float64
        - lambda:             (M,N) float64
    """
    # Market-product blocks (M,J)
    z_beta = rng.normal(
        theta_z_hypers["beta"]["mu"], theta_z_hypers["beta"]["sd"], size=(M, J)
    )
    z_alpha = rng.normal(
        theta_z_hypers["alpha"]["mu"], theta_z_hypers["alpha"]["sd"], size=(M, J)
    )
    z_v = rng.normal(theta_z_hypers["v"]["mu"], theta_z_hypers["v"]["sd"], size=(M, J))
    z_fc = rng.normal(
        theta_z_hypers["fc"]["mu"], theta_z_hypers["fc"]["sd"], size=(M, J)
    )

    # Market-consumer block (M,N)
    z_lambda = rng.normal(
        theta_z_hypers["lambda"]["mu"], theta_z_hypers["lambda"]["sd"], size=(M, N)
    )

    out: dict[str, np.ndarray] = {
        "beta": _sigmoid(z_beta).astype(np.float64, copy=False),
        "alpha": np.exp(z_alpha).astype(np.float64, copy=False),
        "v": np.exp(z_v).astype(np.float64, copy=False),
        "fc": np.exp(z_fc).astype(np.float64, copy=False),
        "lambda": _sigmoid(z_lambda).astype(np.float64, copy=False),
    }
    return out


def compute_u_mj(
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
) -> np.ndarray:
    """
    Compute u_mj = delta_j + E_bar_m + n_mj.

    Returns:
      ndarray (M,J) float64 via numpy promotion.
    """
    return delta_true[None, :] + E_bar_true[:, None] + njt_true


def simulate_price_states(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
    start_state: int,
) -> np.ndarray:
    """
    Simulate a length-T Markov chain over discrete price states.

    Args:
      P_price: (S,S) row-stochastic transition matrix.
      start_state: initial state in {0,...,S-1}.

    Returns:
      ndarray of shape (T,), dtype int64.
    """
    cdf = np.cumsum(P_price, axis=1)
    p_state_t = np.empty((T,), dtype=np.int64)
    p_state_t[0] = start_state

    u = rng.random(T - 1)
    for t in range(1, T):
        prev = p_state_t[t - 1]
        p_state_t[t] = np.searchsorted(cdf[prev], u[t - 1], side="right")

    return p_state_t


def simulate_market_product_price_path(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Simulate one (market, product) price-state path, with a random initial state.

    Returns:
      ndarray (T,), dtype int64.
    """
    S = int(P_price.shape[0])
    start_state = int(rng.integers(0, S))
    return simulate_price_states(rng=rng, P_price=P_price, T=T, start_state=start_state)


def simulate_consumption(
    rng: np.random.Generator,
    N: int,
    T: int,
    lambda_n: np.ndarray,
) -> np.ndarray:
    """
    Simulate latent consumption shocks c_{n,t} ~ Bernoulli(lambda_n).

    Returns:
      ndarray: boolean array of shape (N, T).
    """
    u = rng.random((N, T))
    return u < lambda_n[:, None]


def next_inventory(
    I: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    I_max: int,
) -> np.ndarray:
    """
    Inventory transition with truncation to [0, I_max].

    Transition:
      I_next = clip(I + a - c, 0, I_max)
    """
    I_next = I + a - c
    return np.clip(I_next, 0, I_max)


def solve_ccp_buy(
    u_eff: float,
    beta: float,
    alpha: float,
    v: float,
    fc: float,
    lambda_: float,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Solve the single-consumer DP for one product and return buy CCPs.

    State: (s, I) where
      - s is a discrete price state in {0,...,S-1}
      - I is inventory in {0,...,I_max}

    Returns:
      ndarray: ccp_buy[s, I] = P(a=1 | s, I), shape (S, I_max+1).
    """
    # --- State transitions (indices) ---
    S = int(P_price.shape[0])
    I_size = I_max + 1
    I_grid = np.arange(I_size, dtype=np.int64)

    # Inventory index transitions for (a in {0,1}, c in {0,1}).
    I_a0_c0 = I_grid
    I_a0_c1 = np.maximum(I_grid - 1, 0)
    I_a1_c0 = np.minimum(I_grid + 1, I_max)
    I_a1_c1 = I_grid

    # --- Flow utilities ---
    # No buy: stockout penalty if inventory is zero.
    u0 = (-v * (I_grid == 0))[None, :]

    # Buy: scaled intercept minus price disutility and fixed cost.
    base_buy_s = u_eff - alpha * price_vals - fc
    u1 = base_buy_s[:, None]

    # Waste-at-cap penalty: buying when at cap, then not consuming.
    u1 = u1 - (waste_cost * (1.0 - lambda_)) * (I_grid == I_max)[None, :]

    # --- Bellman fixed point / value iteration ---
    v_fn = np.zeros((S, I_size), dtype=np.float64)

    def _compute_q(v_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute choice-specific value functions q0(s,I), q1(s,I) given v(s,I).
        """
        ev_next = P_price @ v_value

        cont0 = (1.0 - lambda_) * ev_next[:, I_a0_c0] + lambda_ * ev_next[:, I_a0_c1]
        cont1 = (1.0 - lambda_) * ev_next[:, I_a1_c0] + lambda_ * ev_next[:, I_a1_c1]

        q0 = u0 + beta * cont0
        q1 = u1 + beta * cont1
        return q0, q1

    for _ in range(max_iter):
        q0, q1 = _compute_q(v_fn)
        v_new = np.logaddexp(q0, q1)

        if np.max(np.abs(v_new - v_fn)) < tol:
            v_fn = v_new
            break
        v_fn = v_new

    q0, q1 = _compute_q(v_fn)
    denom = np.logaddexp(q0, q1)
    return np.exp(q1 - denom)


def solve_market_product_ccps(
    u_eff: float,
    beta_mj: float,
    alpha_mj: float,
    v_mj: float,
    fc_mj: float,
    lambda_mn: np.ndarray,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Solve CCPs for all consumers for a single (market, product).

    Market-product parameters are shared across consumers; only lambda varies by consumer.

    Returns:
      ndarray: ccp_buy_n_s_i with shape (N, S, I_max+1), float64.
    """
    N = int(lambda_mn.shape[0])
    S = int(P_price.shape[0])
    ccp_buy_n_s_i = np.zeros((N, S, I_max + 1), dtype=np.float64)

    # Cast market-product scalars once; only lambda changes across consumers.
    u_eff_f = float(u_eff)
    beta_f = float(beta_mj)
    alpha_f = float(alpha_mj)
    v_f = float(v_mj)
    fc_f = float(fc_mj)
    waste_cost_f = float(waste_cost)

    for n in range(N):
        ccp_buy_n_s_i[n] = solve_ccp_buy(
            u_eff=u_eff_f,
            beta=beta_f,
            alpha=alpha_f,
            v=v_f,
            fc=fc_f,
            lambda_=float(lambda_mn[n]),
            I_max=I_max,
            P_price=P_price,
            price_vals=price_vals,
            waste_cost=waste_cost_f,
            tol=tol,
            max_iter=max_iter,
        )

    return ccp_buy_n_s_i


def simulate_market_product_panel(
    rng: np.random.Generator,
    p_state_t: np.ndarray,
    ccp_buy_n_s_i: np.ndarray,
    lambda_mn: np.ndarray,
    I_init: np.ndarray,
    I_max: int,
) -> np.ndarray:
    """
    Simulate seller-observed purchases for one (market, product).

    Args:
      p_state_t: (T,) price states for this product.
      ccp_buy_n_s_i: (N,S,I_max+1) buy probabilities by consumer/state/inventory.
      lambda_mn: (N,) consumption probabilities.
      I_init: (N,) initial inventory (latent).
      I_max: inventory cap.

    Returns:
      ndarray: a_nt of shape (N, T), dtype int64.
    """
    N = int(I_init.shape[0])
    T = int(p_state_t.shape[0])
    a_nt = np.zeros((N, T), dtype=np.int64)

    I_curr = I_init.copy()

    # Latent consumption shocks c_{n,t} (boolean); converted to 0/1 ints at use sites.
    c_nt = simulate_consumption(rng=rng, N=N, T=T, lambda_n=lambda_mn)

    n_idx = np.arange(N)
    for t in range(T):
        s = int(p_state_t[t])
        prob_buy = ccp_buy_n_s_i[n_idx, s, I_curr]
        a_bool = rng.random(N) < prob_buy

        a_int = a_bool.astype(np.int64, copy=False)
        c_int = c_nt[:, t].astype(np.int64, copy=False)

        a_nt[:, t] = a_int
        I_curr = next_inventory(I_curr, a_int, c_int, I_max)

    return a_nt


def generate_dgp(
    seed: int,
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
    N: int,
    T: int,
    I_max: int,
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate seller-observed stockpiling panel data for M markets and J products.

    Args:
      P_price_mj: (M,J,S,S) transition matrices (row-stochastic), one per (market, product).
      price_vals_mj: (M,J,S) price levels indexed by the market-product price state.

    Returns:
      a_mnjt: (M, N, J, T) int64 purchases
      p_state_mjt: (M, J, T) int64 price states
      u_mj_true: (M, J) float64 intercepts (unscaled)
      theta_true: dict of true parameters:
        - beta, alpha, v, fc: (M,J) float64
        - lambda:             (M,N) float64
    """
    rng = np.random.default_rng(seed)

    M = int(E_bar_true.shape[0])
    J = int(delta_true.shape[0])

    theta_true = sample_theta_true(rng=rng, M=M, N=N, J=J)

    validate_stockpiling_dgp_inputs(
        seed=seed,
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        N=N,
        T=T,
        I_max=I_max,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
    )

    u_mj_true = compute_u_mj(
        delta_true=delta_true, E_bar_true=E_bar_true, njt_true=njt_true
    )

    p_state_mjt = np.zeros((M, J, T), dtype=np.int64)
    a_mnjt = np.zeros((M, N, J, T), dtype=np.int64)

    theta = theta_true

    for m in range(M):
        # Per-market objects.
        lambda_mn = theta["lambda"][m]  # (N,)

        # Latent initial inventories for each consumer-product.
        I_init_nj = rng.integers(0, I_max + 1, size=(N, J), dtype=np.int64)

        for j in range(J):
            # Per-market-product objects.
            P_price = P_price_mj[m, j]
            price_vals = price_vals_mj[m, j]

            # Market-product-specific price-state path (observed by seller).
            p_state_t = simulate_market_product_price_path(
                rng=rng, P_price=P_price, T=T
            )
            p_state_mjt[m, j] = p_state_t

            # Effective intercept for this (market, product).
            u_eff = u_mj_true[m, j]

            # Market-product parameters for this (m,j).
            beta_mj = float(theta["beta"][m, j])
            alpha_mj = float(theta["alpha"][m, j])
            v_mj = float(theta["v"][m, j])
            fc_mj = float(theta["fc"][m, j])

            # Solve per-consumer CCPs (only lambda varies across consumers).
            ccp_buy_n_s_i = solve_market_product_ccps(
                u_eff=u_eff,
                beta_mj=beta_mj,
                alpha_mj=alpha_mj,
                v_mj=v_mj,
                fc_mj=fc_mj,
                lambda_mn=lambda_mn,
                I_max=I_max,
                P_price=P_price,
                price_vals=price_vals,
                waste_cost=waste_cost,
                tol=tol,
                max_iter=max_iter,
            )

            # Simulate purchases for this product in this market.
            a_mnjt[m, :, j, :] = simulate_market_product_panel(
                rng=rng,
                p_state_t=p_state_t,
                ccp_buy_n_s_i=ccp_buy_n_s_i,
                lambda_mn=lambda_mn,
                I_init=I_init_nj[:, j],
                I_max=I_max,
            )

    return a_mnjt, p_state_mjt, u_mj_true, theta_true
