"""datasets/bonus2_dgp.py

Bonus Q2 DGP (updated spec)

Conventions
-----------
- Outside option is encoded as choice 0.
- Inside products j=1..J are encoded as (j+1) in the simulated y array.

Observed time features
----------------------
- Weekend indicator: w(t) ∈ {0,1} derived from dow_t = t % 7 via w(t)=1{dow_t∈{5,6}}.
- Seasonal angle: θ(t)=2π/P * (t mod P) and Fourier basis with K harmonics.

Core model (inside options)
---------------------------
For inside options j=1..J:

  v_{m,i,j,t} =
      delta_{m,j}
    + beta_market_j[j]
    + beta_habit_j[j] * H_{m,i,j,t}
    + beta_peer_j[j]  * P_{m,i,j,t}
    + beta_dow_j[j, w(t)]
    + S_m(t)

Outside option:
  v_{m,i,0,t} = 0

Habit
-----
  H_{t+1} = decay * H_t + 1{ y_t = j }
where decay ∈ (0,1) is a known scalar passed to the DGP (and estimator).

Peer exposure
-------------
  P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{ y_{k,t-ell} = j }

Seasonality (market-only)
-------------------------
  S_m(t) = Σ_{k=1..K} [ a_mk sin(kθ(t)) + b_mk cos(kθ(t)) ]

Day-of-week (product-only; binary)
----------------------------------
  w(t) ∈ {0,1} and DOW shift is beta_dow_j[j, w(t)].

Product intercept (product-only)
--------------------------------
  beta_market_j[j] is an intercept shift beyond delta_{m,j}.

Parameter draws
---------------
All hyperparameters are read from the dict `dgp_hyperparams` with no defaults.
This file expects (minimum):
  habit_mean, habit_sd
  peer_mean, peer_sd
  mktprod_sd        (used for beta_market_j; name kept to reduce downstream edits)
  dow_prod_sd       (used for beta_dow_j over w∈{0,1})
  season_mkt_sd     (used for market seasonality a_m,b_m)

Notes
-----
- Market-level seasonality only (no product seasonality).
- Product-level DOW only (binary weekday/weekend; no market DOW).
- Product intercept shift only (no market×product intercept beyond delta).
"""

from __future__ import annotations

from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# RNG
# -----------------------------------------------------------------------------


def make_rng(seed: Any) -> np.random.Generator:
    """Create a NumPy random generator (seed may be None)."""
    return np.random.default_rng(seed)


# -----------------------------------------------------------------------------
# Observed time features
# -----------------------------------------------------------------------------


def generate_time_features(T: int, season_period_P: int, K: int):
    """Generate observed time features.

    Returns
    -------
    weekend_t : (T,) int64
        w(t) in {0,1}, where w(t)=1 for weekend and 0 for weekday.
    season_angle_t : (T,) float64
        theta(t) = 2π/P * (t mod P).
    season_sin_kt : (K,T) float64
        sin((k+1) * season_angle_t[t]).
    season_cos_kt : (K,T) float64
        cos((k+1) * season_angle_t[t]).
    dow_t : (T,) int64
        d(t) in {0..6} via t % 7 (returned for debugging).
    """
    T = int(T)
    P = int(season_period_P)
    K = int(K)

    t = np.arange(T, dtype=np.int64)
    dow_t = (t % 7).astype(np.int64)
    weekend_t = (dow_t >= 5).astype(np.int64)

    tau = (t % P).astype(np.float64)
    season_angle_t = (2.0 * np.pi / float(P) * tau).astype(np.float64)

    if K == 0:
        return (
            weekend_t,
            season_angle_t,
            np.zeros((0, T), dtype=np.float64),
            np.zeros((0, T), dtype=np.float64),
            dow_t,
        )

    k = np.arange(1, K + 1, dtype=np.float64)[:, None]  # (K,1)
    ang = k * season_angle_t[None, :]  # (K,T)
    season_sin_kt = np.sin(ang).astype(np.float64)
    season_cos_kt = np.cos(ang).astype(np.float64)

    return weekend_t, season_angle_t, season_sin_kt, season_cos_kt, dow_t


# -----------------------------------------------------------------------------
# Fourier seasonality helper
# -----------------------------------------------------------------------------


def fourier_seasonality(a_coeff, b_coeff, season_sin_kt, season_cos_kt):
    """Compute Fourier seasonality series.

    For each row r and time t:
      S[r,t] = sum_k [a_coeff[r,k] * sin_k[t] + b_coeff[r,k] * cos_k[t]]
    """
    a_coeff = np.asarray(a_coeff, dtype=np.float64)
    b_coeff = np.asarray(b_coeff, dtype=np.float64)
    season_sin_kt = np.asarray(season_sin_kt, dtype=np.float64)
    season_cos_kt = np.asarray(season_cos_kt, dtype=np.float64)

    K = a_coeff.shape[1] if a_coeff.size else 0
    if K == 0:
        R = a_coeff.shape[0] if a_coeff.ndim == 2 else 0
        T = season_sin_kt.shape[1] if season_sin_kt.ndim == 2 else 0
        return np.zeros((R, T), dtype=np.float64)

    return (a_coeff @ season_sin_kt + b_coeff @ season_cos_kt).astype(np.float64)


# -----------------------------------------------------------------------------
# Parameter draws
# -----------------------------------------------------------------------------


def sample_core_product_params(J: int, rng, dgp_hyperparams: dict):
    """Sample product-level habit and peer coefficients.

    Returns
    -------
    beta_habit_j : (J,) float64
    beta_peer_j  : (J,) float64
    """
    J = int(J)

    habit_mean = float(dgp_hyperparams["habit_mean"])
    habit_sd = float(dgp_hyperparams["habit_sd"])
    peer_mean = float(dgp_hyperparams["peer_mean"])
    peer_sd = float(dgp_hyperparams["peer_sd"])

    beta_habit_j = rng.normal(loc=habit_mean, scale=habit_sd, size=J).astype(np.float64)
    beta_peer_j = rng.normal(loc=peer_mean, scale=peer_sd, size=J).astype(np.float64)

    return beta_habit_j, beta_peer_j


def sample_product_intercepts(J: int, rng, dgp_hyperparams: dict):
    """Sample product-only intercept shifts beta_market_j (J,)."""
    J = int(J)
    mkt_sd = float(dgp_hyperparams["mktprod_sd"])  # name kept to reduce edits elsewhere
    return rng.normal(loc=0.0, scale=mkt_sd, size=J).astype(np.float64)


def sample_time_market_params(M: int, K: int, rng, dgp_hyperparams: dict):
    """Sample market-level seasonality coefficients.

    Returns
    -------
    a_m, b_m : (M,K) float64
    """
    M = int(M)
    K = int(K)

    season_mkt_sd = float(dgp_hyperparams["season_mkt_sd"])

    if K == 0:
        a_m = np.zeros((M, 0), dtype=np.float64)
        b_m = np.zeros((M, 0), dtype=np.float64)
    else:
        a_m = rng.normal(loc=0.0, scale=season_mkt_sd, size=(M, K)).astype(np.float64)
        b_m = rng.normal(loc=0.0, scale=season_mkt_sd, size=(M, K)).astype(np.float64)

    return a_m, b_m


def sample_time_product_params(J: int, rng, dgp_hyperparams: dict):
    """Sample product-level weekday/weekend effects beta_dow_j (J,2)."""
    J = int(J)
    dow_prod_sd = float(dgp_hyperparams["dow_prod_sd"])
    return rng.normal(loc=0.0, scale=dow_prod_sd, size=(J, 2)).astype(np.float64)


# -----------------------------------------------------------------------------
# Network and peer exposure
# -----------------------------------------------------------------------------


def generate_sparse_network(N: int, avg_friends: float, friends_sd: float, rng):
    """Generate adjacency list (within a market).

    Out-degree K_i ~ Normal(avg_friends, friends_sd^2), rounded, clipped to [0, N-1].
    nbrs[i] is an int64 array of out-neighbors consumer i observes.
    """
    N = int(N)
    mu = float(avg_friends)
    sd = float(friends_sd)

    nbrs = []
    all_idx = np.arange(N, dtype=np.int64)

    for i in range(N):
        k = int(np.rint(rng.normal(loc=mu, scale=sd)))
        k = max(0, min(k, N - 1))

        if k == 0:
            nbrs.append(np.zeros(0, dtype=np.int64))
            continue

        candidates = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
        chosen = rng.choice(candidates, size=k, replace=False).astype(np.int64)
        nbrs.append(chosen)

    return nbrs


def _update_recent_choice_counts_inplace(C_counts, y_vec, sign: int):
    """Update rolling inside-choice counts.

    - Outside option is encoded as 0 and does not contribute.
    - Inside product j is encoded as (j+1), so column index is (y-1).
    """
    y_vec = np.asarray(y_vec, dtype=np.int64)
    idx = np.where(y_vec > 0)[0]
    if idx.size == 0:
        return
    prod = y_vec[idx] - 1
    np.add.at(C_counts, (idx, prod), int(sign))


def advance_peer_window(peer_buf, buf_pos: int, counts, y_new):
    """Advance a length-L circular buffer and maintain per-consumer inside counts."""
    y_expired = peer_buf[buf_pos]
    _update_recent_choice_counts_inplace(counts, y_expired, sign=-1)
    _update_recent_choice_counts_inplace(counts, y_new, sign=+1)
    peer_buf[buf_pos] = y_new
    return (buf_pos + 1) % peer_buf.shape[0]


def peer_exposure_from_recent_counts(C_counts, nbrs):
    """Compute peer exposure P from recent inside-choice counts."""
    N = len(nbrs)
    P = np.zeros((N, C_counts.shape[1]), dtype=np.float64)

    for i in range(N):
        ni = nbrs[i]
        if ni.size:
            P[i, :] = C_counts[ni, :].sum(axis=0)

    return P


# -----------------------------------------------------------------------------
# Choice model (MNL with outside option)
# -----------------------------------------------------------------------------


def sample_mnl(v, rng):
    """Sample MNL choices when outside option has utility 0.

    Returns y in {0..J}, where:
      - y=0 is outside
      - y=j+1 corresponds to inside product j (0-indexed column in v)
    """
    v = np.asarray(v, dtype=np.float64)
    N, J = v.shape

    if J == 0:
        return np.zeros(N, dtype=np.int64)

    shift = np.maximum(0.0, v.max(axis=1))
    exp_inside = np.exp(v - shift[:, None])
    exp_outside = np.exp(-shift)  # outside utility is 0

    denom = exp_outside + exp_inside.sum(axis=1)
    p_outside = exp_outside / denom
    p_inside = exp_inside / denom[:, None]

    cdf = np.cumsum(np.concatenate([p_outside[:, None], p_inside], axis=1), axis=1)
    u = rng.random(size=N)

    idx = (cdf >= u[:, None]).argmax(axis=1)
    idx[u > cdf[:, -1]] = J  # numerical guard
    return idx.astype(np.int64)


# -----------------------------------------------------------------------------
# Simulation core
# -----------------------------------------------------------------------------


def simulate_one_market(
    delta_mj,
    beta_market_j,
    beta_dow_j,
    weekend_t,
    S_m,
    beta_habit_j,
    beta_peer_j,
    decay: float,
    nbrs,
    rng,
    peer_lookback_L: int,
):
    """Simulate one market panel y_{i,t}."""
    L = int(peer_lookback_L)

    delta_mj = np.asarray(delta_mj, dtype=np.float64)
    beta_market_j = np.asarray(beta_market_j, dtype=np.float64)
    beta_dow_j = np.asarray(beta_dow_j, dtype=np.float64)
    weekend_t = np.asarray(weekend_t, dtype=np.int64)
    S_m = np.asarray(S_m, dtype=np.float64)
    beta_habit_j = np.asarray(beta_habit_j, dtype=np.float64)
    beta_peer_j = np.asarray(beta_peer_j, dtype=np.float64)
    decay = float(decay)

    J = int(delta_mj.shape[0])
    T = int(weekend_t.shape[0])
    N = len(nbrs)

    y_it = np.zeros((N, T), dtype=np.int64)
    H = np.zeros((N, J), dtype=np.float64)

    peer_buf = np.zeros((L, N), dtype=np.int64)
    buf_pos = 0
    recent_counts = np.zeros((N, J), dtype=np.int32)

    for t in range(T):
        P = peer_exposure_from_recent_counts(recent_counts, nbrs)  # (N,J)
        w = int(weekend_t[t])

        base_j = delta_mj + beta_market_j + beta_dow_j[:, w] + S_m[t]
        v = base_j[None, :] + beta_habit_j[None, :] * H + beta_peer_j[None, :] * P

        y = sample_mnl(v, rng)
        y_it[:, t] = y

        # Habit update: H_{t+1} = decay * H_t + 1{y_t == j}
        H *= decay
        chosen = np.where(y > 0)[0]
        if chosen.size:
            prod = y[chosen] - 1
            np.add.at(H, (chosen, prod), 1.0)

        buf_pos = advance_peer_window(peer_buf, buf_pos, recent_counts, y)

    return y_it


def simulate_bonus2_dgp(
    delta,
    N: int,
    T: int,
    avg_friends: float,
    params_true: dict,
    decay: float,
    seed: Any,
    season_period: int,
    friends_sd: float,
    K: int,
    peer_lookback_L: int,
):
    """Simulate multi-market data for Bonus Q2 under the updated spec."""
    delta = np.asarray(delta, dtype=np.float64)
    M, J = delta.shape

    N = int(N)
    T = int(T)
    K = int(K)
    peer_lookback_L = int(peer_lookback_L)

    rng = make_rng(seed)

    weekend_t, theta, season_sin_kt, season_cos_kt, dow = generate_time_features(
        T, season_period, K
    )

    # Draw parameters
    beta_habit_j, beta_peer_j = sample_core_product_params(
        J=J, rng=rng, dgp_hyperparams=params_true
    )
    beta_market_j = sample_product_intercepts(J=J, rng=rng, dgp_hyperparams=params_true)
    beta_dow_j = sample_time_product_params(J=J, rng=rng, dgp_hyperparams=params_true)
    a_m, b_m = sample_time_market_params(M=M, K=K, rng=rng, dgp_hyperparams=params_true)

    # Precompute market seasonal series
    S_m_all = fourier_seasonality(a_m, b_m, season_sin_kt, season_cos_kt)  # (M,T)

    y = np.zeros((M, N, T), dtype=np.int64)
    nbrs_list = []

    for m in range(M):
        nbrs = generate_sparse_network(N, avg_friends, friends_sd, rng)
        nbrs_list.append(nbrs)

        y[m] = simulate_one_market(
            delta_mj=delta[m],
            beta_market_j=beta_market_j,
            beta_dow_j=beta_dow_j,
            weekend_t=weekend_t,
            S_m=S_m_all[m],
            beta_habit_j=beta_habit_j,
            beta_peer_j=beta_peer_j,
            decay=decay,
            nbrs=nbrs,
            rng=rng,
            peer_lookback_L=peer_lookback_L,
        )

    return {
        # observed / known inputs
        "y": y,
        "delta": delta,
        "w": weekend_t,
        "dow": dow,
        "theta": theta,
        "season_sin_kt": season_sin_kt,
        "season_cos_kt": season_cos_kt,
        "nbrs": nbrs_list,
        "decay": float(decay),
        # true parameters (for evaluation)
        "beta_market_j": beta_market_j,
        "beta_habit_j": beta_habit_j,
        "beta_peer_j": beta_peer_j,
        "beta_dow_j": beta_dow_j,
        "a_m": a_m,
        "b_m": b_m,
        # metadata
        "season_period": int(season_period),
        "K": int(K),
        "peer_lookback_L": int(peer_lookback_L),
        "params_true": params_true,
    }
