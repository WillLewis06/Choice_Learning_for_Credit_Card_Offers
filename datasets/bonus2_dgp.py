"""
datasets/bonus2_dgp.py

Bonus Question 2 DGP: habit formation + peer effects + time features + market features
layered on top of fixed Phase-1 baseline utilities (Zhang feature-based model outputs).

This module does NOT run Phase 1. It assumes Phase-1 baseline utilities are provided as
an input array `delta` of shape (M, J), fixed over time within each market.

True DGP (no alpha):
  v_{m,i,j,t} = delta_{m,j}
               + beta_habit  * H_{m,i,j,t}
               + beta_peer   * P_{m,i,j,t}
               + beta_time^T   c_t
               + beta_market^T x_{m,t}
  v_{m,i,0,t} = 0  (outside option)

Choices y_{m,i,t} are sampled via multinomial logit with outside option.
Peer exposure uses lagged neighbor choices (t-1).
Habit stock is a decaying count of past purchases of each product.
"""

from __future__ import annotations

import numpy as np


def make_rng(seed):
    """
    Create a NumPy random generator.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    rng : np.random.Generator
    """
    return np.random.default_rng(seed)


def generate_time_features(T, kind="dow+season", season_period=28):
    """
    Generate deterministic time features c_t shared across markets.

    Minimal default:
      - day-of-week one-hot (7 dims) using t mod 7
      - seasonality sin/cos with fixed period (2 dims), optional

    Parameters
    ----------
    T : int
        Number of time steps.
    kind : str
        "dow" or "dow+season".
    season_period : int
        Period for seasonality (used when kind includes "season").

    Returns
    -------
    c : np.ndarray
        Array of shape (T, d_c).
    """
    t = np.arange(T, dtype=np.int64)

    if kind not in ("dow", "dow+season"):
        raise ValueError(f"Unknown kind={kind!r}. Expected 'dow' or 'dow+season'.")

    # Day-of-week one-hot
    dow = (t % 7).astype(np.int64)
    c_dow = np.zeros((T, 7), dtype=np.float64)
    c_dow[np.arange(T), dow] = 1.0

    if kind == "dow":
        return c_dow

    # Seasonality (sin/cos) with a fixed period
    if season_period is None or season_period <= 0:
        raise ValueError(
            "season_period must be a positive integer when using 'dow+season'."
        )

    ang = 2.0 * np.pi * (t.astype(np.float64) / float(season_period))
    c_season = np.column_stack([np.sin(ang), np.cos(ang)]).astype(np.float64)

    return np.concatenate([c_dow, c_season], axis=1)


def generate_market_features(M, T, d_x, rng, ar1_rho=0.0, ar1_sigma=1.0):
    """
    Generate market covariates x_{m,t} independently across markets.

    If ar1_rho == 0, features are i.i.d. N(0, ar1_sigma^2).
    Else, each dimension follows AR(1):
      x_t = ar1_rho * x_{t-1} + ar1_sigma * eps_t, eps_t ~ N(0, I)

    Parameters
    ----------
    M : int
        Number of markets.
    T : int
        Number of time steps.
    d_x : int
        Number of market features.
    rng : np.random.Generator
        Random generator.
    ar1_rho : float
        AR(1) coefficient.
    ar1_sigma : float
        Innovation scale.

    Returns
    -------
    x : np.ndarray
        Array of shape (M, T, d_x).
    """
    x = np.zeros((M, T, d_x), dtype=np.float64)

    if ar1_rho == 0.0:
        x[:] = rng.normal(loc=0.0, scale=ar1_sigma, size=(M, T, d_x))
        return x

    eps = rng.normal(loc=0.0, scale=ar1_sigma, size=(M, T, d_x))
    x[:, 0, :] = eps[:, 0, :]
    for t in range(1, T):
        x[:, t, :] = ar1_rho * x[:, t - 1, :] + eps[:, t, :]
    return x


def generate_sparse_network(N, expected_degree, rng):
    """
    Generate a sparse within-market social network in adjacency-list form.

    Minimal choice: fixed-K neighbors per node, sampled uniformly without replacement.
    Weights are uniform 1/K per row.

    Parameters
    ----------
    N : int
        Number of consumers in the market.
    expected_degree : int
        Number of neighbors per node (K). Clipped to [0, N-1].
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    nbrs : np.ndarray
        Neighbor indices of shape (N, K), dtype int64.
    wts : np.ndarray
        Row-normalized weights of shape (N, K), dtype float64.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    K = int(expected_degree)
    if K < 0:
        raise ValueError("expected_degree must be >= 0.")
    K = min(K, max(0, N - 1))

    nbrs = np.zeros((N, K), dtype=np.int64)
    wts = np.zeros((N, K), dtype=np.float64)

    if K == 0:
        return nbrs, wts

    for i in range(N):
        # Candidates exclude self i
        candidates = np.arange(N, dtype=np.int64)
        if N > 1:
            candidates = np.concatenate([candidates[:i], candidates[i + 1 :]])
        chosen = rng.choice(candidates, size=K, replace=False)
        nbrs[i, :] = chosen

    wts[:] = 1.0 / float(K)
    return nbrs, wts


def peer_exposure_from_prev_choice(y_prev, nbrs, wts, J):
    """
    Compute lagged peer exposure P_{i,j} from neighbors' previous choices.

    P[i, j-1] = sum_k wts[i,k] * 1{ y_prev[nbrs[i,k]] == j }, for j=1..J
    Outside option (0) contributes nothing.

    Parameters
    ----------
    y_prev : np.ndarray
        Previous choices, shape (N,), ints in {0..J}.
    nbrs : np.ndarray
        Neighbor indices, shape (N, K).
    wts : np.ndarray
        Neighbor weights, shape (N, K).
    J : int
        Number of inside products.

    Returns
    -------
    P : np.ndarray
        Peer exposure, shape (N, J).
    """
    N = y_prev.shape[0]
    K = nbrs.shape[1]
    P = np.zeros((N, J), dtype=np.float64)

    if K == 0 or J == 0:
        return P

    for i in range(N):
        for k in range(K):
            nb = nbrs[i, k]
            y_nb = int(y_prev[nb])
            if y_nb > 0:
                # map product id j in {1..J} to column j-1
                P[i, y_nb - 1] += wts[i, k]
    return P


def sample_mnl(v, rng):
    """
    Sample multinomial logit choices with an outside option of utility 0.

    Parameters
    ----------
    v : np.ndarray
        Inside-option utilities, shape (N, J).
    rng : np.random.Generator

    Returns
    -------
    y : np.ndarray
        Choices, shape (N,), ints in {0..J}.
        0 denotes outside option.
    """
    N, J = v.shape
    y = np.zeros(N, dtype=np.int64)

    if J == 0:
        return y

    # Stable computation: subtract row max.
    vmax = np.max(v, axis=1)  # shape (N,)
    ev = np.exp(v - vmax[:, None])  # shape (N, J)
    e0 = np.exp(-vmax)  # outside utility 0 shifted by -vmax
    denom = e0 + np.sum(ev, axis=1)  # shape (N,)

    # Probabilities for inside options
    p_inside = ev / denom[:, None]  # shape (N, J)
    p0 = e0 / denom  # shape (N,)

    # Sample via cumulative probabilities (minimal, readable loop)
    for i in range(N):
        u = rng.random()
        cum = p0[i]
        if u < cum:
            y[i] = 0
            continue
        # Inside options: assign smallest j such that u < p0 + sum_{r<=j} p_r
        for j in range(J):
            cum += p_inside[i, j]
            if u < cum:
                y[i] = j + 1
                break
        # If due to numeric drift no break, choose last inside option
        if y[i] == 0:
            y[i] = J
    return y


def simulate_market_panel(delta_mj, x_t, c_t, nbrs, wts, params, rng):
    """
    Simulate one market panel y_{i,t} for i=1..N, t=1..T.

    Parameters
    ----------
    delta_mj : np.ndarray
        Baseline utilities from Phase 1 for this market, shape (J,).
        Fixed over time in this DGP.
    x_t : np.ndarray
        Market features for this market, shape (T, d_x).
    c_t : np.ndarray
        Time features shared across markets, shape (T, d_c).
    nbrs : np.ndarray
        Neighbor indices, shape (N, K).
    wts : np.ndarray
        Neighbor weights, shape (N, K).
    params : dict
        True parameters:
          - beta_habit : float
          - beta_peer : float
          - beta_time : np.ndarray shape (d_c,)
          - beta_market : np.ndarray shape (d_x,)
          - rho_h : float in (0, 1]
    rng : np.random.Generator

    Returns
    -------
    y_it : np.ndarray
        Simulated choices, shape (N, T), ints in {0..J}.
    """
    delta_mj = np.asarray(delta_mj, dtype=np.float64)
    x_t = np.asarray(x_t, dtype=np.float64)
    c_t = np.asarray(c_t, dtype=np.float64)

    J = int(delta_mj.shape[0])
    T = int(x_t.shape[0])
    N = int(nbrs.shape[0])

    beta_habit = float(params["beta_habit"])
    beta_peer = float(params["beta_peer"])
    beta_time = np.asarray(params["beta_time"], dtype=np.float64)
    beta_market = np.asarray(params["beta_market"], dtype=np.float64)
    rho_h = float(params["rho_h"])

    if not (0.0 < rho_h <= 1.0):
        raise ValueError("rho_h must be in (0, 1].")
    if beta_time.shape[0] != c_t.shape[1]:
        raise ValueError("beta_time length must match c_t second dimension.")
    if beta_market.shape[0] != x_t.shape[1]:
        raise ValueError("beta_market length must match x_t second dimension.")

    y_it = np.zeros((N, T), dtype=np.int64)

    # Habit stock H[i, j] for j=1..J stored as column j-1
    H = np.zeros((N, J), dtype=np.float64)

    # Previous choices for lagged peer exposure; start at outside
    y_prev = np.zeros(N, dtype=np.int64)

    decay = 1.0 - rho_h

    for t in range(T):
        # Peer exposure from previous time step
        P = peer_exposure_from_prev_choice(y_prev, nbrs, wts, J)

        # Scalars that shift all inside options equally at time t
        time_shift = float(beta_time @ c_t[t])
        market_shift = float(beta_market @ x_t[t])

        # Deterministic inside utilities (true DGP; no alpha)
        v = (
            delta_mj[None, :]
            + beta_habit * H
            + beta_peer * P
            + time_shift
            + market_shift
        )

        y = sample_mnl(v, rng)
        y_it[:, t] = y

        # Habit update: decay then add 1 to chosen inside option
        if decay != 0.0:
            H *= decay

        for i in range(N):
            yi = int(y[i])
            if yi > 0:
                H[i, yi - 1] += 1.0

        y_prev = y

    return y_it


def simulate_bonus2_dgp(
    delta,
    N,
    T,
    d_x,
    expected_degree,
    params_true,
    seed=None,
    time_kind="dow+season",
    season_period=28,
    ar1_rho=0.0,
    ar1_sigma=1.0,
):
    """
    Simulate multi-market Bonus Q2 data with fixed Phase-1 baseline utilities.

    Parameters
    ----------
    delta : np.ndarray
        Baseline utilities from Phase 1, shape (M, J). Fixed over time.
    N : int
        Number of consumers per market (constant across markets in this minimal implementation).
    T : int
        Number of time steps.
    d_x : int
        Market feature dimension.
    expected_degree : int
        Fixed number of neighbors per consumer in each market graph.
    params_true : dict
        True parameters (see simulate_market_panel).
    seed : int | None
        Random seed.
    time_kind : str
        "dow" or "dow+season".
    season_period : int
        Period for seasonality if used.
    ar1_rho : float
        AR(1) coefficient for market features; 0.0 means i.i.d.
    ar1_sigma : float
        Innovation scale for market features.

    Returns
    -------
    out : dict
        Keys:
          - y : np.ndarray shape (M, N, T), ints in {0..J}
          - delta : np.ndarray shape (M, J)
          - x : np.ndarray shape (M, T, d_x)
          - c : np.ndarray shape (T, d_c)
          - nbrs : list of np.ndarray, each shape (N, K)
          - wts : list of np.ndarray, each shape (N, K)
          - params_true : dict (echo)
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError("delta must have shape (M, J).")
    M, J = delta.shape
    if J <= 0:
        raise ValueError("J must be positive.")
    if N <= 0 or T <= 0 or d_x < 0:
        raise ValueError("N and T must be positive; d_x must be >= 0.")

    rng = make_rng(seed)

    c = generate_time_features(T, time_kind, season_period)
    x = generate_market_features(M, T, d_x, rng, ar1_rho, ar1_sigma)

    y = np.zeros((M, N, T), dtype=np.int64)
    nbrs_list = []
    wts_list = []

    for m in range(M):
        nbrs, wts = generate_sparse_network(N, expected_degree, rng)
        nbrs_list.append(nbrs)
        wts_list.append(wts)

        y[m] = simulate_market_panel(delta[m], x[m], c, nbrs, wts, params_true, rng)

    return {
        "y": y,
        "delta": delta,
        "x": x,
        "c": c,
        "nbrs": nbrs_list,
        "wts": wts_list,
        "params_true": params_true,
    }
