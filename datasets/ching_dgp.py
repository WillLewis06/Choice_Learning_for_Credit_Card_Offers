"""
Ching-style stockpiling DGP (multi-product, seller-observed).

This DGP implements a Phase-3 model where:
  - all products j=1..J are modelled in the stockpiling layer
  - each (market, product) pair has its own exogenous Markov price-state process
  - price-state processes are simulated independently across (market, product) pairs (by construction)

Upstream truth inputs (from Phase 1–2):
  - delta_true (J,)  product baseline utilities
  - E_bar_true (M,)  market shocks
  - njt_true   (M,J) market-product shocks

Phase-3 market-product intercept passed into the stockpiling layer:
  u_mj_true[m, j] = delta_true[j] + E_bar_true[m] + njt_true[m, j]

Seller-observed outputs:
  - a_mnjt      (M, N, J, T) purchases (0/1, stored as int64)
  - p_state_mjt (M, J, T)    price states (int64)
  - u_mj_true   (M, J)       fixed market-product intercepts (unscaled, float64)

True parameters (DGP draws):
  - global: beta (scalar)
  - per product (J,): alpha, v, fc
  - per (market, consumer) (M,N): lambda

IMPORTANT:
  - There is NO "true" u_scale in the DGP.
    Any u_scale (market-level scaling of Phase-1/2 intercepts) is an estimation-only nuisance
    concept and must not be generated or applied in this simulation.

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

from ching.stockpiling_input_validation import normalize_stockpiling_dgp_inputs


def _logit(p: float) -> float:
    """Logit transform for p in (0,1)."""
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid applied elementwise."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def sample_theta_true(
    rng: np.random.Generator,
    M: int,
    N: int,
    J: int,
) -> dict[str, np.ndarray]:
    """
    Sample "true" parameters used by the DGP.

    Returns dict of ndarrays:
      - beta:   scalar float64 stored as 0-d ndarray
      - alpha:  (J,) float64
      - v:      (J,) float64
      - fc:     (J,) float64
      - lambda: (M,N) float64, in (0,1)
    """
    beta = np.asarray(rng.uniform(0.85, 0.98), dtype=np.float64)

    alpha = rng.lognormal(mean=np.log(1.0), sigma=0.25, size=(J,)).astype(np.float64)
    v = rng.lognormal(mean=np.log(2.0), sigma=0.25, size=(J,)).astype(np.float64)
    fc = rng.lognormal(mean=np.log(0.2), sigma=0.25, size=(J,)).astype(np.float64)

    # Market-consumer consumption probability.
    lam_logit = rng.normal(loc=_logit(0.25), scale=0.8, size=(M, N)).astype(np.float64)
    lam = _sigmoid(lam_logit)

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "lambda": lam,
    }


def compute_u_mj(
    delta_true: np.ndarray,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
) -> np.ndarray:
    """
    Compute market-product intercept u_mj from Phase 1–2 truth:
      u_mj = delta_j + E_bar_m + n_mj
    """
    return delta_true[None, :] + E_bar_true[:, None] + njt_true


def simulate_price_states(
    rng: np.random.Generator,
    P_price: np.ndarray,
    T: int,
    start_state: int,
) -> np.ndarray:
    """
    Simulate a discrete Markov chain of length T given transition matrix P_price (S,S).

    Returns:
      ndarray (T,), dtype int64.
    """
    S = int(P_price.shape[0])
    # Precompute row CDFs once; faster than rng.choice(..., p=row) in a tight loop.
    cdf = np.cumsum(P_price, axis=1)
    s = np.zeros((T,), dtype=np.int64)
    s[0] = int(start_state)

    u = rng.random(int(T - 1))
    for t in range(1, T):
        prev = int(s[t - 1])
        nxt = int(np.searchsorted(cdf[prev], u[t - 1], side="right"))
        # Guard against floating-point CDF rounding returning S.
        s[t] = nxt if nxt < S else (S - 1)

    return s


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

    Args:
      I: (N,) current inventory
      a: (N,) purchase indicator {0,1}
      c: (N,) consumption indicator {0,1}

    Returns:
      I_next: (N,) updated inventory in [0, I_max]
    """
    # Expect I and a to be int64. c is typically boolean; NumPy promotes it safely.
    I_next = I + a - c
    np.clip(I_next, 0, I_max, out=I_next)
    return I_next


def solve_ccp_buy(
    u_eff: float,
    beta: float,
    alpha_j: float,
    v_j: float,
    fc_j: float,
    lambda_n: float,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Solve for consumer CCPs for a single (market, product, consumer) given lambda_n.

    State: (s, i) where
      - s is price state in {0,...,S-1}
      - i is inventory in {0,...,I_max}

    Flow utilities (spec):
      U0(s,i) = - v_j * 1{i=0}
      U1(s,i) = u_eff - alpha_j * price(s) - fc_j
                - waste_cost * (1 - lambda_n) * 1{i=I_max}

    Inventory dynamics (spec):
      i' = clip(i + a - c, 0, I_max),  c ~ Bernoulli(lambda_n)

    Returns:
      ccp_buy_s_i: (S, I_max+1) float64 in [0,1].
    """
    S = int(P_price.shape[0])
    I_size = int(I_max + 1)

    # Inventory index maps (shared across s).
    inv = np.arange(I_size, dtype=np.int64)
    inv_buy = np.minimum(inv + 1, I_max)
    inv_down = np.maximum(inv - 1, 0)

    lam = float(lambda_n)
    one_minus_lam = 1.0 - lam

    # Flow utilities that do not depend on the continuation value.
    u_buy_s = (
        float(u_eff)
        - float(alpha_j) * price_vals.astype(np.float64, copy=False)
        - float(fc_j)
    )  # (S,)
    u_nb_i = -float(v_j) * (inv == 0).astype(np.float64)  # (I,)
    waste_i = (
        float(waste_cost) * one_minus_lam * (inv == I_max).astype(np.float64)
    )  # (I,)

    # Inclusive value V(s,i).
    V = np.zeros((S, I_size), dtype=np.float64)
    V_old = np.zeros_like(V)

    beta_f = float(beta)
    tol_f = float(tol)

    for _ in range(int(max_iter)):
        V_old[:] = V

        # Expected future value under price transitions:
        EV_next = P_price @ V_old  # (S, I)

        # Continuation values integrating out consumption:
        cont_buy = one_minus_lam * EV_next[:, inv_buy] + lam * EV_next[:, inv]
        cont_nb = one_minus_lam * EV_next[:, inv] + lam * EV_next[:, inv_down]

        Q_buy = (u_buy_s[:, None] - waste_i[None, :]) + beta_f * cont_buy
        Q_nb = u_nb_i[None, :] + beta_f * cont_nb

        np.logaddexp(Q_buy, Q_nb, out=V)

        diff = float(np.max(np.abs(V - V_old)))
        if diff < tol_f:
            break

    # Compute CCP_buy from converged V.
    EV = P_price @ V
    cont_buy = one_minus_lam * EV[:, inv_buy] + lam * EV[:, inv]
    cont_nb = one_minus_lam * EV[:, inv] + lam * EV[:, inv_down]

    Q_buy = (u_buy_s[:, None] - waste_i[None, :]) + beta_f * cont_buy
    Q_nb = u_nb_i[None, :] + beta_f * cont_nb

    # Stable logistic: P(buy) = 1 / (1 + exp(Q_nb - Q_buy))
    d = np.clip(Q_nb - Q_buy, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(d))


def solve_market_product_ccps(
    u_eff: float,
    beta: float,
    alpha_j: float,
    v_j: float,
    fc_j: float,
    lambda_mn: np.ndarray,
    I_max: int,
    P_price: np.ndarray,
    price_vals: np.ndarray,
    waste_cost: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    Solve CCP_buy for all consumers in a given market-product pair.

    Args:
      lambda_mn: (N,) consumer-specific consumption probabilities.

    Returns:
      ccp_buy_n_s_i: (N, S, I_max+1) float64.
    """
    N = int(lambda_mn.shape[0])
    S = int(P_price.shape[0])
    I_size = int(I_max + 1)

    ccp_buy = np.zeros((N, S, I_size), dtype=np.float64)

    for n in range(N):
        ccp_buy[n] = solve_ccp_buy(
            u_eff=float(u_eff),
            beta=float(beta),
            alpha_j=float(alpha_j),
            v_j=float(v_j),
            fc_j=float(fc_j),
            lambda_n=float(lambda_mn[n]),
            I_max=int(I_max),
            P_price=P_price,
            price_vals=price_vals,
            waste_cost=float(waste_cost),
            tol=float(tol),
            max_iter=int(max_iter),
        )

    return ccp_buy


def simulate_market_product_panel(
    rng: np.random.Generator,
    p_state_t: np.ndarray,
    ccp_buy_n_s_i: np.ndarray,
    lambda_mn: np.ndarray,
    I_init: np.ndarray,
    I_max: int,
) -> np.ndarray:
    """
    Simulate purchases a_{n,t} for one market-product pair.

    Args:
      p_state_t: (T,) int price-state path.
      ccp_buy_n_s_i: (N,S,I_max+1) buy probabilities.
      lambda_mn: (N,) consumption probabilities.
      I_init: (N,) initial inventory levels in [0, I_max].

    Returns:
      a_nt: (N,T) int64 purchases.
    """
    N = int(lambda_mn.shape[0])
    T = int(p_state_t.shape[0])
    a_nt = np.zeros((N, T), dtype=np.int64)

    # Latent consumption draws:
    c_nt = simulate_consumption(rng=rng, N=N, T=T, lambda_n=lambda_mn)

    I_n = I_init.astype(np.int64)

    idx = np.arange(N, dtype=np.int64)

    for t in range(T):
        s = int(p_state_t[t])

        # Buy decisions conditional on (s, I_n).
        buy_prob = ccp_buy_n_s_i[idx, s, I_n]
        a_t = rng.random((N,)) < buy_prob
        a_nt[:, t] = a_t

        # Inventory transition.
        I_n = next_inventory(I=I_n, a=a_nt[:, t], c=c_nt[:, t], I_max=I_max)

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

    This function performs a single boundary normalization/validation via
    normalize_stockpiling_dgp_inputs(...). After that point, all inputs are treated
    as canonical (dtypes/shapes/ranges) and the remainder of the code is simulation.

    Args:
      P_price_mj: (M,J,S,S) transition matrices (row-stochastic), one per (market, product).
      price_vals_mj: (M,J,S) price levels indexed by the market-product price state.

    Returns:
      a_mnjt: (M, N, J, T) int64 purchases
      p_state_mjt: (M, J, T) int64 price states
      u_mj_true: (M, J) float64 intercepts (unscaled)
      theta_true: dict of true parameters:
        - beta:    scalar float64 stored as a 0-d ndarray
        - alpha:   (J,) float64
        - v:       (J,) float64
        - fc:      (J,) float64
        - lambda:  (M,N) float64
    """
    norm = normalize_stockpiling_dgp_inputs(
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

    rng = np.random.default_rng(int(norm["seed"]))

    # Canonical dimensions and inputs.
    M = int(norm["M"])
    N = int(norm["N"])
    J = int(norm["J"])
    T = int(norm["T"])
    I_max = int(norm["I_max"])

    delta_true = norm["delta_true"]
    E_bar_true = norm["E_bar_true"]
    njt_true = norm["njt_true"]
    P_price_mj = norm["P_price_mj"]
    price_vals_mj = norm["price_vals_mj"]
    waste_cost = float(norm["waste_cost"])
    tol = float(norm["tol"])
    max_iter = int(norm["max_iter"])

    theta_true = sample_theta_true(rng=rng, M=M, N=N, J=J)

    u_mj_true = compute_u_mj(
        delta_true=delta_true, E_bar_true=E_bar_true, njt_true=njt_true
    )

    p_state_mjt = np.zeros((M, J, T), dtype=np.int64)
    a_mnjt = np.zeros((M, N, J, T), dtype=np.int64)

    beta = float(theta_true["beta"])
    alpha = theta_true["alpha"]
    v = theta_true["v"]
    fc = theta_true["fc"]
    lambda_true = theta_true["lambda"]

    for m in range(M):
        lambda_mn = lambda_true[m]  # (N,)

        # Latent initial inventories for each consumer-product.
        I_init_nj = rng.integers(0, I_max + 1, size=(N, J), dtype=np.int64)

        for j in range(J):
            P_price = P_price_mj[m, j]
            price_vals = price_vals_mj[m, j]

            # Market-product-specific price-state path (observed by seller).
            p_state_t = simulate_market_product_price_path(
                rng=rng, P_price=P_price, T=T
            )
            p_state_mjt[m, j] = p_state_t

            # DGP uses the unscaled Phase-1/2 intercept.
            u_eff = float(u_mj_true[m, j])

            # Solve per-consumer CCPs (only lambda varies across consumers).
            ccp_buy_n_s_i = solve_market_product_ccps(
                u_eff=u_eff,
                beta=beta,
                alpha_j=float(alpha[j]),
                v_j=float(v[j]),
                fc_j=float(fc[j]),
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
