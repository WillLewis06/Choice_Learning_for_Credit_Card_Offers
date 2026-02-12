"""
datasets/bonus2_dgp.py

Bonus Q2 DGP (updated model)

Conventions
-----------
- Outside option is encoded as choice 0.
- Inside products j=1..J are encoded as (j+1) in the simulated y array.

Core model (inside options)
---------------------------
v_{m,i,j,t} =
    delta_{m,j}
  + beta_market_mj[m,j]
  + beta_habit_j[j] * H_{m,i,j,t}
  + beta_peer_j[j]  * P_{m,i,j,t}
  + beta_dow_m[m, d(t)] + beta_dow_j[j, d(t)]
  + S_m(t) + S_j(t)

Habit:
  H_{t+1} = decay_rate_j * H_t + 1{ y_t = j }

Peer exposure (count over last L periods of peers' purchases):
  P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{ y_{k,t-ell} = j }

Time features:
  d(t) = t mod 7
  theta(t) = 2π/P * (t mod P)
  S_m(t), S_j(t) are Fourier series with K harmonics.

Parameter draws
---------------
All hyperparameters are read from the dict `params_true` with no defaults.
Expected keys:
  habit_mean, habit_sd
  peer_mean, peer_sd
  mktprod_sd
  dow_mkt_sd, dow_prod_sd
  season_mkt_sd, season_prod_sd
  decay_rate_eps
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


def generate_time_features(T: int, season_period: int, K: int):
    """
    Returns
    -------
    dow : (T,) int64
        d(t) in {0..6} via t % 7.
    theta : (T,) float64
        2π/P * (t mod P).
    sin_k_theta : (K,T) float64
        sin((k+1)*theta_t).
    cos_k_theta : (K,T) float64
        cos((k+1)*theta_t).
    """
    T = int(T)
    P = int(season_period)
    K = int(K)

    t = np.arange(T, dtype=np.int64)
    dow = (t % 7).astype(np.int64)

    tau = (t % P).astype(np.float64)
    theta = (2.0 * np.pi / float(P) * tau).astype(np.float64)

    if K == 0:
        return (
            dow,
            theta,
            np.zeros((0, T), dtype=np.float64),
            np.zeros((0, T), dtype=np.float64),
        )

    k = np.arange(1, K + 1, dtype=np.float64)[:, None]  # (K,1)
    ang = k * theta[None, :]  # (K,T)
    sin_k_theta = np.sin(ang).astype(np.float64)
    cos_k_theta = np.cos(ang).astype(np.float64)
    return dow, theta, sin_k_theta, cos_k_theta


# -----------------------------------------------------------------------------
# Fourier seasonality helper
# -----------------------------------------------------------------------------


def fourier_seasonality(a_coeff, b_coeff, sin_k_theta, cos_k_theta):
    """
    S[r,t] = sum_k [a_coeff[r,k]*sin_k_theta[k,t] + b_coeff[r,k]*cos_k_theta[k,t]]
    """
    a_coeff = np.asarray(a_coeff, dtype=np.float64)
    b_coeff = np.asarray(b_coeff, dtype=np.float64)
    sin_k_theta = np.asarray(sin_k_theta, dtype=np.float64)
    cos_k_theta = np.asarray(cos_k_theta, dtype=np.float64)

    K = a_coeff.shape[1] if a_coeff.size else 0
    if K == 0:
        R = a_coeff.shape[0] if a_coeff.ndim == 2 else 0
        T = sin_k_theta.shape[1] if sin_k_theta.ndim == 2 else 0
        return np.zeros((R, T), dtype=np.float64)

    return (a_coeff @ sin_k_theta + b_coeff @ cos_k_theta).astype(np.float64)


# -----------------------------------------------------------------------------
# Parameter draws (hyperparameters read from params_true dict, no defaults)
# -----------------------------------------------------------------------------


def _kappa_from_average_decay_rate(mu: float) -> float:
    """kappa for Beta(kappa,1) given mean mu."""
    mu = float(mu)
    return mu / (1.0 - mu)


def sample_core_product_params(
    J: int, rng, average_decay_rate: float, params_true: dict
):
    """
    Sample:
      beta_habit_j : (J,)
      beta_peer_j  : (J,)
      decay_rate_j : (J,)  with Beta(kappa,1) skew toward 1, then clipped by decay_rate_eps
      kappa_decay  : scalar
    """
    J = int(J)

    habit_mean = float(params_true["habit_mean"])
    habit_sd = float(params_true["habit_sd"])
    peer_mean = float(params_true["peer_mean"])
    peer_sd = float(params_true["peer_sd"])
    decay_rate_eps = float(params_true["decay_rate_eps"])

    beta_habit_j = rng.normal(loc=habit_mean, scale=habit_sd, size=J).astype(np.float64)
    beta_peer_j = rng.normal(loc=peer_mean, scale=peer_sd, size=J).astype(np.float64)

    kappa = float(_kappa_from_average_decay_rate(average_decay_rate))
    u = rng.random(size=J).astype(np.float64)
    decay_rate = np.power(u, 1.0 / kappa).astype(np.float64)  # Beta(kappa,1)

    if decay_rate_eps > 0.0:
        decay_rate = np.minimum(decay_rate, 1.0 - decay_rate_eps)

    return beta_habit_j, beta_peer_j, decay_rate, kappa


def sample_market_product_intercepts(M: int, J: int, rng, params_true: dict):
    """beta_market_mj (M,J)."""
    M = int(M)
    J = int(J)
    mktprod_sd = float(params_true["mktprod_sd"])
    return rng.normal(loc=0.0, scale=mktprod_sd, size=(M, J)).astype(np.float64)


def sample_time_market_params(M: int, K: int, rng, params_true: dict):
    """
    Market-level:
      beta_dow_m : (M,7)
      a_m, b_m   : (M,K)
    """
    M = int(M)
    K = int(K)

    dow_mkt_sd = float(params_true["dow_mkt_sd"])
    season_mkt_sd = float(params_true["season_mkt_sd"])

    beta_dow_m = rng.normal(loc=0.0, scale=dow_mkt_sd, size=(M, 7)).astype(np.float64)

    if K == 0:
        a_m = np.zeros((M, 0), dtype=np.float64)
        b_m = np.zeros((M, 0), dtype=np.float64)
    else:
        a_m = rng.normal(loc=0.0, scale=season_mkt_sd, size=(M, K)).astype(np.float64)
        b_m = rng.normal(loc=0.0, scale=season_mkt_sd, size=(M, K)).astype(np.float64)

    return beta_dow_m, a_m, b_m


def sample_time_product_params(J: int, K: int, rng, params_true: dict):
    """
    Product-level:
      beta_dow_j : (J,7)
      a_j, b_j   : (J,K)
    """
    J = int(J)
    K = int(K)

    dow_prod_sd = float(params_true["dow_prod_sd"])
    season_prod_sd = float(params_true["season_prod_sd"])

    beta_dow_j = rng.normal(loc=0.0, scale=dow_prod_sd, size=(J, 7)).astype(np.float64)

    if K == 0:
        a_j = np.zeros((J, 0), dtype=np.float64)
        b_j = np.zeros((J, 0), dtype=np.float64)
    else:
        a_j = rng.normal(loc=0.0, scale=season_prod_sd, size=(J, K)).astype(np.float64)
        b_j = rng.normal(loc=0.0, scale=season_prod_sd, size=(J, K)).astype(np.float64)

    return beta_dow_j, a_j, b_j


def apply_identifiability_constraints(beta_dow_m, beta_dow_j, a_j, b_j):
    """
    Centering constraints for additive decompositions (in-place):
      - market DOW: mean across weekdays is 0 within each market
      - product DOW: mean across weekdays is 0 within each product, then mean across
        products is 0 for each weekday
      - product seasonal coeffs: centered across products for each harmonic
    """
    beta_dow_m -= beta_dow_m.mean(axis=1, keepdims=True)

    beta_dow_j -= beta_dow_j.mean(axis=1, keepdims=True)
    beta_dow_j -= beta_dow_j.mean(axis=0, keepdims=True)

    if a_j.size:
        a_j -= a_j.mean(axis=0, keepdims=True)
        b_j -= b_j.mean(axis=0, keepdims=True)


# -----------------------------------------------------------------------------
# Network and peer exposure
# -----------------------------------------------------------------------------


def generate_sparse_network(N: int, avg_friends: float, friends_sd: float, rng):
    """
    Adjacency list nbrs[i] of out-neighbors consumer i observes (within a market).
    Out-degree K_i ~ Normal(avg_friends, friends_sd^2), rounded, clipped to [0, N-1].
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
    """
    Update rolling counts (inside choices only).

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
    """
    Maintain a length-L circular buffer of past choices and per-consumer inside counts.

    counts[k,j] = number of times consumer k bought inside product (j+1) over last L periods.
    """
    y_expired = peer_buf[buf_pos]
    _update_recent_choice_counts_inplace(counts, y_expired, sign=-1)
    _update_recent_choice_counts_inplace(counts, y_new, sign=+1)
    peer_buf[buf_pos] = y_new
    return (buf_pos + 1) % peer_buf.shape[0]


def peer_exposure_from_recent_counts(C_counts, nbrs):
    """
    P[i,:] = sum_{k in nbrs[i]} C_counts[k,:]
    This equals the count of peers' inside purchases over the last L periods.
    """
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
    """
    Sample MNL choices when outside option has utility 0.

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
    dow,
    beta_market_mj,
    beta_dow_m,
    beta_dow_j,
    S_m,
    S_j,
    beta_habit_j,
    beta_peer_j,
    decay_rate_j,
    nbrs,
    rng,
    peer_lookback_L: int,
):
    """
    Simulate one market panel y_{i,t}.

    Choices:
      - outside encoded as 0
      - inside product j encoded as (j+1)
    """
    L = int(peer_lookback_L)

    delta_mj = np.asarray(delta_mj, dtype=np.float64)
    beta_market_mj = np.asarray(beta_market_mj, dtype=np.float64)
    beta_dow_m = np.asarray(beta_dow_m, dtype=np.float64)
    beta_dow_j = np.asarray(beta_dow_j, dtype=np.float64)
    S_m = np.asarray(S_m, dtype=np.float64)
    S_j = np.asarray(S_j, dtype=np.float64)
    beta_habit_j = np.asarray(beta_habit_j, dtype=np.float64)
    beta_peer_j = np.asarray(beta_peer_j, dtype=np.float64)
    decay_rate_j = np.asarray(decay_rate_j, dtype=np.float64)

    J = int(delta_mj.shape[0])
    T = int(dow.shape[0])
    N = len(nbrs)

    y_it = np.zeros((N, T), dtype=np.int64)
    H = np.zeros((N, J), dtype=np.float64)

    peer_buf = np.zeros((L, N), dtype=np.int64)
    buf_pos = 0
    recent_counts = np.zeros((N, J), dtype=np.int32)

    for t in range(T):
        P = peer_exposure_from_recent_counts(recent_counts, nbrs)  # (N,J)
        d = int(dow[t])

        base_j = (
            delta_mj
            + beta_market_mj
            + (beta_dow_m[d] + beta_dow_j[:, d])
            + S_m[t]
            + S_j[:, t]
        )
        v = base_j[None, :] + beta_habit_j[None, :] * H + beta_peer_j[None, :] * P

        y = sample_mnl(v, rng)
        y_it[:, t] = y

        # Habit update: H_{t+1} = decay * H_t + 1{y_t == j}
        H *= decay_rate_j[None, :]
        chosen = np.where(y > 0)[0]  # outside encoded as 0
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
    average_decay_rate: float,
    seed: Any,
    season_period: int,
    friends_sd: float,
    K: int,
    peer_lookback_L: int,
):
    """
    Simulate multi-market data for Bonus Q2.

    Notes
    -----
    - No input validation is performed here (handled elsewhere).
    - All hyperparameters are read from params_true with no defaults.
    """
    delta = np.asarray(delta, dtype=np.float64)
    M, J = delta.shape

    N = int(N)
    T = int(T)
    K = int(K)
    peer_lookback_L = int(peer_lookback_L)

    rng = make_rng(seed)

    # Time features
    dow, theta, sin_k_theta, cos_k_theta = generate_time_features(
        T=T, season_period=season_period, K=K
    )

    # Draw parameters
    beta_habit_j, beta_peer_j, decay_rate_j, kappa_decay = sample_core_product_params(
        J=J, rng=rng, average_decay_rate=average_decay_rate, params_true=params_true
    )
    beta_market_mj = sample_market_product_intercepts(
        M=M, J=J, rng=rng, params_true=params_true
    )
    beta_dow_m, a_m, b_m = sample_time_market_params(
        M=M, K=K, rng=rng, params_true=params_true
    )
    beta_dow_j, a_j, b_j = sample_time_product_params(
        J=J, K=K, rng=rng, params_true=params_true
    )

    apply_identifiability_constraints(
        beta_dow_m=beta_dow_m, beta_dow_j=beta_dow_j, a_j=a_j, b_j=b_j
    )

    # Precompute seasonal series
    S_m_all = fourier_seasonality(a_m, b_m, sin_k_theta, cos_k_theta)  # (M,T)
    S_j_all = fourier_seasonality(a_j, b_j, sin_k_theta, cos_k_theta)  # (J,T)

    # Simulate markets
    y = np.zeros((M, N, T), dtype=np.int64)
    nbrs_list = []

    for m in range(M):
        nbrs = generate_sparse_network(N, avg_friends, friends_sd, rng)
        nbrs_list.append(nbrs)

        y[m] = simulate_one_market(
            delta_mj=delta[m],
            dow=dow,
            beta_market_mj=beta_market_mj[m],
            beta_dow_m=beta_dow_m[m],
            beta_dow_j=beta_dow_j,
            S_m=S_m_all[m],
            S_j=S_j_all,
            beta_habit_j=beta_habit_j,
            beta_peer_j=beta_peer_j,
            decay_rate_j=decay_rate_j,
            nbrs=nbrs,
            rng=rng,
            peer_lookback_L=peer_lookback_L,
        )

    return {
        # observed / known inputs
        "y": y,
        "delta": delta,
        "dow": dow,
        "theta": theta,
        "sin_k_theta": sin_k_theta,
        "cos_k_theta": cos_k_theta,
        "nbrs": nbrs_list,
        # true parameters (for evaluation)
        "beta_habit_j": beta_habit_j,
        "beta_peer_j": beta_peer_j,
        "decay_rate_j": decay_rate_j,
        "beta_market_mj": beta_market_mj,
        "beta_dow_m": beta_dow_m,
        "beta_dow_j": beta_dow_j,
        "a_m": a_m,
        "b_m": b_m,
        "a_j": a_j,
        "b_j": b_j,
        # metadata
        "kappa_decay": float(kappa_decay),
        "average_decay_rate": float(average_decay_rate),
        "season_period": int(season_period),
        "K": int(K),
        "peer_lookback_L": int(peer_lookback_L),
        "params_true": params_true,
    }
