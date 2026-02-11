"""
datasets/bonus2_dgp.py

Bonus Question 2 DGP (current):

Inputs from Phase 1:
- Fixed product set and fixed baseline utilities delta_{m,j} (provided as input).

Within each market:
- Habit formation (per consumer-product).
- Peer effects via a sparse social network (lagged neighbor choices).
- Time effects:
    (i) consumer-specific day-of-week effects (7 values per consumer),
    (ii) mean-zero seasonal sinusoid.
- Fixed market features x_m (constant over time).

True DGP (no alpha):
  v_{m,i,j,t} = delta_{m,j}
               + beta_habit  * H_{m,i,j,t}
               + beta_peer   * P_{m,i,j,t}
               + beta_dow_{m,i, dow(t)}
               + beta_season * s_t
               + beta_market^T x_m
  v_{m,i,0,t} = 0  (outside option)

Network degrees:
- For each consumer i, sample K_i ~ Normal(avg_friends, friends_sd^2), round to int,
  clip to [0, N-1]. Sample K_i distinct neighbors uniformly from {0..N-1}\{i}.
  Weights are uniform 1/K_i when K_i > 0.
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
        Random generator.
    """
    return np.random.default_rng(seed)


def generate_time_features(T, season_period=28):
    """
    Generate observed time features known to the estimator.

    Two features:
      - dow_t: day-of-week index in {0..6}, where dow_t = t % 7
      - s_t: mean-zero seasonal sinusoid sin(2π t / P)

    Parameters
    ----------
    T : int
        Number of time steps.
    season_period : int
        Period P for the sinusoid (must be positive).

    Returns
    -------
    dow : np.ndarray
        Day-of-week index, shape (T,), dtype int64.
    s : np.ndarray
        Seasonal sinusoid, shape (T,), dtype float64.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if season_period is None or season_period <= 0:
        raise ValueError("season_period must be a positive integer.")

    t = np.arange(T, dtype=np.int64)
    dow = (t % 7).astype(np.int64)

    ang = 2.0 * np.pi * (t.astype(np.float64) / float(season_period))
    s = np.sin(ang).astype(np.float64)

    return dow, s


def generate_market_features(M, d_x, rng):
    """
    Generate fixed market features x_m that do not vary over time.

    Parameters
    ----------
    M : int
        Number of markets.
    d_x : int
        Dimension of market features (>= 0).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    x : np.ndarray
        Market features, shape (M, d_x), dtype float64.
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if d_x < 0:
        raise ValueError("d_x must be >= 0.")
    if d_x == 0:
        return np.zeros((M, 0), dtype=np.float64)
    return rng.normal(loc=0.0, scale=1.0, size=(M, d_x)).astype(np.float64)


def generate_consumer_dow_effects(N, rng, sd=0.5):
    """
    Generate consumer-specific day-of-week effects beta_dow_i[d] for d=0..6.

    Effects are independent across consumers and across days:
      beta_dow_i[d] ~ Normal(0, sd^2)

    Parameters
    ----------
    N : int
        Number of consumers.
    rng : np.random.Generator
        Random generator.
    sd : float
        Standard deviation of day-of-week effects.

    Returns
    -------
    beta_dow : np.ndarray
        Consumer-specific day-of-week effects, shape (N, 7), dtype float64.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if sd < 0.0:
        raise ValueError("sd must be >= 0.")
    if sd == 0.0:
        return np.zeros((N, 7), dtype=np.float64)
    return rng.normal(loc=0.0, scale=sd, size=(N, 7)).astype(np.float64)


def generate_sparse_network(N, avg_friends, friends_sd, rng):
    """
    Generate a sparse within-market social network in adjacency-list form, with
    heterogeneous degree per consumer.

    For each consumer i:
      K_i ~ Normal(avg_friends, friends_sd^2), rounded to int and clipped to [0, N-1].
      Sample K_i distinct neighbors uniformly from {0..N-1} \\ {i}.
      Assign weights uniformly 1/K_i when K_i > 0.

    This yields a directed network: i listing k as a neighbor does not imply k lists i.

    Parameters
    ----------
    N : int
        Number of consumers.
    avg_friends : float
        Mean number of friends.
    friends_sd : float
        Standard deviation for number of friends (>= 0).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    nbrs : list[np.ndarray]
        Length N. nbrs[i] has shape (K_i,) with neighbor indices (int64).
    wts : list[np.ndarray]
        Length N. wts[i] has shape (K_i,) with weights summing to 1 when K_i>0 (float64).
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if friends_sd < 0.0:
        raise ValueError("friends_sd must be >= 0.")

    mu = float(avg_friends)
    sd = float(friends_sd)

    nbrs = []
    wts = []

    for i in range(N):
        k = int(np.rint(rng.normal(loc=mu, scale=sd)))
        if k < 0:
            k = 0
        if k > N - 1:
            k = N - 1

        if k == 0:
            nbrs.append(np.zeros(0, dtype=np.int64))
            wts.append(np.zeros(0, dtype=np.float64))
            continue

        candidates = np.arange(N, dtype=np.int64)
        candidates = np.concatenate([candidates[:i], candidates[i + 1 :]])
        chosen = rng.choice(candidates, size=k, replace=False)

        nbrs.append(chosen.astype(np.int64))
        wts.append(np.full(k, 1.0 / float(k), dtype=np.float64))

    return nbrs, wts


def peer_exposure_from_prev_choice(y_prev, nbrs, wts, J):
    """
    Compute lagged peer exposure P_{i,j} from neighbors' previous choices.

    For j=1..J:
      P[i, j-1] = sum_k wts[i][k] * 1{ y_prev[nbrs[i][k]] == j }
    Outside option (0) contributes nothing.

    Parameters
    ----------
    y_prev : np.ndarray
        Previous choices, shape (N,), ints in {0..J}.
    nbrs : list[np.ndarray]
        Neighbors; nbrs[i] has shape (K_i,).
    wts : list[np.ndarray]
        Weights; wts[i] has shape (K_i,).
    J : int
        Number of inside products.

    Returns
    -------
    P : np.ndarray
        Peer exposure, shape (N, J), dtype float64.
    """
    N = int(y_prev.shape[0])
    P = np.zeros((N, J), dtype=np.float64)

    if J == 0:
        return P

    for i in range(N):
        ni = nbrs[i]
        wi = wts[i]
        for k in range(int(ni.shape[0])):
            nb = int(ni[k])
            y_nb = int(y_prev[nb])
            if y_nb > 0:
                P[i, y_nb - 1] += float(wi[k])

    return P


def sample_mnl(v, rng):
    """
    Sample multinomial logit choices with an outside option utility fixed at 0.

    Parameters
    ----------
    v : np.ndarray
        Inside-option utilities, shape (N, J).
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    y : np.ndarray
        Choices, shape (N,), ints in {0..J}.
        0 denotes the outside option.
    """
    N, J = v.shape
    y = np.zeros(N, dtype=np.int64)

    if J == 0:
        return y

    vmax_inside = np.max(v, axis=1)
    shift = np.maximum(0.0, vmax_inside)

    ev = np.exp(v - shift[:, None])
    e0 = np.exp(-shift)
    denom = e0 + np.sum(ev, axis=1)

    p0 = e0 / denom
    p_inside = ev / denom[:, None]

    for i in range(N):
        u = rng.random()
        cum = p0[i]
        if u < cum:
            y[i] = 0
            continue
        for j in range(J):
            cum += p_inside[i, j]
            if u < cum:
                y[i] = j + 1
                break
        if y[i] == 0:
            y[i] = J

    return y


def simulate_market_panel(
    delta_mj, x_m, dow_t, s_t, beta_dow_i, nbrs, wts, params, rng
):
    """
    Simulate one market panel y_{i,t} given fixed baseline utilities and network.

    Implements:
      v_{i,j,t} = delta_j
                  + beta_habit  * H_{i,j,t}
                  + beta_peer   * P_{i,j,t}
                  + beta_dow_i[ dow_t[t] ]
                  + beta_season * s_t[t]
                  + beta_market^T x_m
      v_{i,0,t} = 0

    Parameters
    ----------
    delta_mj : np.ndarray
        Baseline utilities for this market, shape (J,).
    x_m : np.ndarray
        Fixed market features, shape (d_x,).
    dow_t : np.ndarray
        Day-of-week indices, shape (T,), ints in {0..6}.
    s_t : np.ndarray
        Seasonal sinusoid, shape (T,), float.
    beta_dow_i : np.ndarray
        Consumer-specific day-of-week effects, shape (N, 7).
    nbrs : list[np.ndarray]
        Neighbor lists; length N.
    wts : list[np.ndarray]
        Weight lists; length N.
    params : dict
        True parameters:
          - beta_habit : float
          - beta_peer : float
          - beta_season : float
          - beta_market : np.ndarray shape (d_x,)
          - rho_h : float in (0, 1]
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    y_it : np.ndarray
        Simulated choices, shape (N, T), ints in {0..J}.
    """
    delta_mj = np.asarray(delta_mj, dtype=np.float64)
    x_m = np.asarray(x_m, dtype=np.float64)
    dow_t = np.asarray(dow_t, dtype=np.int64)
    s_t = np.asarray(s_t, dtype=np.float64)
    beta_dow_i = np.asarray(beta_dow_i, dtype=np.float64)

    J = int(delta_mj.shape[0])
    T = int(dow_t.shape[0])
    N = len(nbrs)

    if s_t.shape[0] != T:
        raise ValueError("dow_t and s_t must have the same length.")
    if beta_dow_i.shape != (N, 7):
        raise ValueError("beta_dow_i must have shape (N, 7).")
    if len(wts) != N:
        raise ValueError("nbrs and wts must have the same length.")

    beta_habit = float(params["beta_habit"])
    beta_peer = float(params["beta_peer"])
    beta_season = float(params["beta_season"])
    beta_market = np.asarray(params["beta_market"], dtype=np.float64)
    rho_h = float(params["rho_h"])

    if not (0.0 < rho_h <= 1.0):
        raise ValueError("rho_h must be in (0, 1].")
    if beta_market.shape[0] != x_m.shape[0]:
        raise ValueError("beta_market length must match x_m length.")

    market_shift = float(beta_market @ x_m)

    y_it = np.zeros((N, T), dtype=np.int64)
    H = np.zeros((N, J), dtype=np.float64)
    y_prev = np.zeros(N, dtype=np.int64)
    decay = 1.0 - rho_h

    for t in range(T):
        P = peer_exposure_from_prev_choice(y_prev, nbrs, wts, J)

        dow_idx = int(dow_t[t])
        if dow_idx < 0 or dow_idx > 6:
            raise ValueError("dow_t must contain only values in {0..6}.")
        dow_shift = beta_dow_i[:, dow_idx]  # shape (N,)

        time_shift = beta_season * float(s_t[t])

        v = (
            delta_mj[None, :]
            + beta_habit * H
            + beta_peer * P
            + dow_shift[:, None]
            + time_shift
            + market_shift
        )

        y = sample_mnl(v, rng)
        y_it[:, t] = y

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
    avg_friends,
    params_true,
    seed=None,
    season_period=28,
    friends_sd=2.0,
):
    """
    Simulate multi-market data for the Bonus Q2 true model.

    Notes:
    - delta is fixed from Phase 1 and time-invariant.
    - x_m is fixed per market (known to estimator).
    - dow_t and s_t are known to the estimator.
    - beta_dow_{m,i,d} is consumer-specific and independent across i and d in the DGP.
    - Social network degrees K_i are sampled from Normal(avg_friends, friends_sd^2) and clipped.

    Parameters
    ----------
    delta : np.ndarray
        Phase-1 baseline utilities, shape (M, J). Fixed over time.
    N : int
        Number of consumers per market (constant across markets).
    T : int
        Number of time steps.
    d_x : int
        Market feature dimension (fixed features per market).
    avg_friends : float
        Mean number of friends per consumer (Normal mean for K_i).
    params_true : dict
        True model parameters:
          - beta_habit : float
          - beta_peer : float
          - beta_season : float
          - beta_market : np.ndarray shape (d_x,)
          - rho_h : float in (0, 1]
        Optional DGP-only:
          - beta_dow_sd : float (default 0.5)
    seed : int | None
        Random seed.
    season_period : int
        Period for seasonal sinusoid s_t.
    friends_sd : float
        Std dev for number of friends (Normal sd), >= 0.

    Returns
    -------
    out : dict
        Keys:
          - y : np.ndarray shape (M, N, T), ints in {0..J}
          - delta : np.ndarray shape (M, J) (echo)
          - x : np.ndarray shape (M, d_x) fixed market features
          - dow : np.ndarray shape (T,) day-of-week indices
          - s : np.ndarray shape (T,) seasonal sinusoid
          - beta_dow : np.ndarray shape (M, N, 7) consumer-specific dow effects
          - nbrs : list (len M) of list (len N) of neighbor index arrays
          - wts : list (len M) of list (len N) of neighbor weight arrays
          - params_true : dict (echo)
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError("delta must have shape (M, J).")
    M, J = delta.shape
    if M <= 0 or J <= 0:
        raise ValueError("delta must have positive shape (M, J).")
    if N <= 0:
        raise ValueError("N must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if d_x < 0:
        raise ValueError("d_x must be >= 0.")

    rng = make_rng(seed)

    dow, s = generate_time_features(T, season_period)
    x = generate_market_features(M, d_x, rng)

    beta_dow_sd = float(params_true.get("beta_dow_sd", 0.5))
    beta_dow = np.zeros((M, N, 7), dtype=np.float64)
    for m in range(M):
        beta_dow[m] = generate_consumer_dow_effects(N, rng, beta_dow_sd)

    y = np.zeros((M, N, T), dtype=np.int64)
    nbrs_list = []
    wts_list = []

    for m in range(M):
        nbrs, wts = generate_sparse_network(N, avg_friends, friends_sd, rng)
        nbrs_list.append(nbrs)
        wts_list.append(wts)

        y[m] = simulate_market_panel(
            delta[m],
            x[m],
            dow,
            s,
            beta_dow[m],
            nbrs,
            wts,
            params_true,
            rng,
        )

    return {
        "y": y,
        "delta": delta,
        "x": x,
        "dow": dow,
        "s": s,
        "beta_dow": beta_dow,
        "nbrs": nbrs_list,
        "wts": wts_list,
        "params_true": params_true,
    }
