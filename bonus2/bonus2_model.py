"""
bonus2_model.py

TensorFlow core mechanics for Bonus Q2 (updated spec):
  - Habit + peer effects with deterministic state construction from observed choices
  - Product-only intercept
  - Product-only weekday/weekend (binary) shift
  - Market-only seasonality (Fourier)
  - Known scalar habit decay (passed to model; not estimated)

Observed inputs (typically passed by posterior/estimator):
  y_mit          (M,N,T) int32  choices; 0=outside, c=j+1=inside product j
  delta_mj       (M,J)   f64    Phase-1 baseline utilities (fixed)
  weekend_t            (T,)    int32  weekend indicator in {0,1} (1=weekend, 0=weekday)
  season_sin_kt  (K,T)   f64    sin((k+1)*theta_t[t]) basis
  season_cos_kt  (K,T)   f64    cos((k+1)*theta_t[t]) basis
  peer_adj_m     tuple length M of tf.SparseTensor (N,N) adjacency (known networks)
  L              scalar int32   peer lookback window length (known hyperparameter)
  decay          scalar f64     habit decay in (0,1) (known input)

Parameters (theta), all float64:
  beta_market_j  (J,)    product intercept shift beyond delta_mj
  beta_habit_j   (J,)    habit sensitivity
  beta_peer_j    (J,)    peer sensitivity
  beta_dow_j     (J,2)   product weekday/weekend shifts
  a_m, b_m       (M,K)   market seasonal Fourier coefficients

Utility (inside options j=1..J; outside has utility 0):
  v_{m,i,j,t} =
      delta_{m,j}
    + beta_market_j[j]
    + beta_habit_j[j] * H_{m,i,j,t}
    + beta_peer_j[j]  * P_{m,i,j,t}
    + beta_dow_j[j, weekend_t[t]]
    + S_m[m,t]

Seasonality (market-only):
  S_m[m,t] = sum_k a_m[m,k]*sin_k[t] + b_m[m,k]*cos_k[t]

Habit stock (known scalar decay):
  H_{t+1} = decay * H_t + x_t
where x_t = 1{y_t == j}.

Peer exposure (lookback L):
  P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{ y_{k,t-ell} == j }.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

PeerAdjacency = tuple[tf.SparseTensor, ...]


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Map unconstrained parameters z[*] to theta[*].

    This updated spec has no constrained parameters in theta (decay is known input),
    so this is an identity mapping.

    Required z keys and shapes (all float64):
      z_beta_market_j  (J,)   -> beta_market_j
      z_beta_habit_j   (J,)   -> beta_habit_j
      z_beta_peer_j    (J,)   -> beta_peer_j
      z_beta_dow_j     (J,2)  -> beta_dow_j
      z_a_m            (M,K)  -> a_m
      z_b_m            (M,K)  -> b_m
    """
    return {
        "beta_market_j": z["z_beta_market_j"],
        "beta_habit_j": z["z_beta_habit_j"],
        "beta_peer_j": z["z_beta_peer_j"],
        "beta_dow_j": z["z_beta_dow_j"],
        "a_m": z["z_a_m"],
        "b_m": z["z_b_m"],
    }


# =============================================================================
# Social network adjacency (known / observed)
# =============================================================================


def neighbors_to_sparse_adj(nbrs_i: Sequence[Sequence[int]], N: int) -> tf.SparseTensor:
    """Build a sparse adjacency matrix A of shape (N,N) with A[i,k]=1 if k in nbrs_i[i]."""
    rows: list[int] = []
    cols: list[int] = []

    for i in range(int(N)):
        for k in nbrs_i[i]:
            rows.append(i)
            cols.append(int(k))

    if not rows:
        indices = tf.zeros((0, 2), dtype=tf.int64)
        values = tf.zeros((0,), dtype=tf.float64)
    else:
        indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
        values = tf.ones((len(rows),), dtype=tf.float64)

    A = tf.SparseTensor(indices=indices, values=values, dense_shape=(int(N), int(N)))
    return tf.sparse.reorder(A)


def build_peer_adjacency(
    nbrs_m: Sequence[Sequence[Sequence[int]]], N: int
) -> PeerAdjacency:
    """Convert per-market neighbor sets into per-market sparse adjacency tensors."""
    return tuple(
        neighbors_to_sparse_adj(nbrs_i=nbrs_m[m], N=N) for m in range(len(nbrs_m))
    )


# =============================================================================
# Deterministic state construction from observed y
# =============================================================================


def _inside_choice_onehot(y_mit: tf.Tensor, J: tf.Tensor) -> tf.Tensor:
    """Build inside-choice one-hot indicators.

    Args:
      y_mit: (M,N,T) int32, 0=outside, c=j+1=inside product j
      J: scalar int32 (number of inside products)

    Returns:
      x_mntj: (M,N,T,J) float64, x[...,t,j]=1{y[...,t]==j+1}
    """
    J_int = tf.cast(J, tf.int32)

    # Map {0,1..J} -> {0,0..J-1} and mask out outside choices.
    y0 = tf.maximum(y_mit - 1, 0)
    onehot = tf.one_hot(y0, depth=J_int, dtype=tf.float64)
    mask = tf.cast(y_mit > 0, tf.float64)[..., None]
    return onehot * mask


def compute_habit_stock_from_onehot(
    x_mntj: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Compute habit stocks H_{m,i,j,t} used in utility at each time t.

    Recurrence:
      H_{t+1} = decay * H_t + x_t

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      decay: scalar float64

    Returns:
      H_mntj: (M,N,T,J) float64, where H[...,t,:] is the pre-choice stock at time t.
    """
    decay = tf.cast(decay, tf.float64)
    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])  # (T,M,N,J)

    M = tf.shape(x_mntj)[0]
    N = tf.shape(x_mntj)[1]
    J_int = tf.shape(x_mntj)[3]

    H0 = tf.zeros((M, N, J_int), dtype=tf.float64)

    def _step(H_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return H_curr * decay + x_curr

    H_next_t = tf.scan(_step, x_t, initializer=H0)  # (T,M,N,J) gives H_{t+1}

    # Pre-choice stock at time t is H_t; we have H_{t+1} from scan.
    H_curr_t = tf.concat([H0[None, ...], H_next_t[:-1, ...]], axis=0)  # (T,M,N,J)
    return tf.transpose(H_curr_t, perm=[1, 2, 0, 3])  # (M,N,T,J)


def rolling_window_counts(x_mntj: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """For each t, compute counts over the previous L periods (excluding current t).

      C[...,t,:] = sum_{u=max(0,t-L)}^{t-1} x[...,u,:]

    Args:
      x_mntj: (M,N,T,J) float64
      L: scalar int32

    Returns:
      C_mntj: (M,N,T,J) float64
    """
    L_int = tf.cast(L, tf.int32)
    T = tf.shape(x_mntj)[2]

    S = tf.cumsum(x_mntj, axis=2)  # inclusive prefix
    Z = tf.zeros_like(S[:, :, :1, :])
    S_pad = tf.concat([Z, S], axis=2)  # (M,N,T+1,J)

    t_idx = tf.range(T, dtype=tf.int32)
    start_idx = tf.maximum(0, t_idx - L_int)

    # sum_{u < t} x_u is S_pad[..., t]
    S_end = tf.gather(S_pad, t_idx, axis=2)
    S_start = tf.gather(S_pad, start_idx, axis=2)

    return S_end - S_start


def compute_peer_exposure_from_onehot(
    x_mntj: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
) -> tf.Tensor:
    """Compute peer exposures P_{m,i,j,t} using known per-market adjacency matrices.

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      peer_adj_m: tuple length M of tf.SparseTensor (N,N)
      L: scalar int32

    Returns:
      P_mntj: (M,N,T,J) float64
    """
    C_mntj = rolling_window_counts(x_mntj=x_mntj, L=L)  # (M,N,T,J)

    N = tf.shape(C_mntj)[1]
    T = tf.shape(C_mntj)[2]
    J_int = tf.shape(C_mntj)[3]

    P_list: list[tf.Tensor] = []
    for m in range(len(peer_adj_m)):
        A = peer_adj_m[m]
        C_m = C_mntj[m]  # (N,T,J)
        C_flat = tf.reshape(C_m, (N, -1))
        P_flat = tf.sparse.sparse_dense_matmul(A, C_flat)
        P_list.append(tf.reshape(P_flat, (N, T, J_int)))

    return tf.stack(P_list, axis=0)  # (M,N,T,J)


# =============================================================================
# Seasonality and utilities
# =============================================================================


def fourier_seasonality_tf(
    a_coeff: tf.Tensor,
    b_coeff: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
) -> tf.Tensor:
    """Compute Fourier seasonal series.

      S[r,t] = sum_k a[r,k]*season_sin_kt[k,t] + b[r,k]*season_cos_kt[k,t]

    Shapes:
      a_coeff, b_coeff: (R,K)
      season_sin_kt, season_cos_kt: (K,T)

    Returns:
      S: (R,T)
    """
    return tf.linalg.matmul(a_coeff, season_sin_kt) + tf.linalg.matmul(
        b_coeff, season_cos_kt
    )


def utilities_mntj_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Compute inside-option utilities v_{m,i,j,t} (no outside option column).

    Returns:
      v_mntj: (M,N,T,J) float64
    """
    y_mit = tf.convert_to_tensor(y_mit, dtype=tf.int32)
    delta_mj = tf.convert_to_tensor(delta_mj, dtype=tf.float64)
    weekend_t = tf.convert_to_tensor(weekend_t, dtype=tf.int32)
    decay = tf.cast(decay, tf.float64)

    # Market-only seasonality: S_m is (M,T)
    S_m = fourier_seasonality_tf(
        tf.convert_to_tensor(theta["a_m"], dtype=tf.float64),
        tf.convert_to_tensor(theta["b_m"], dtype=tf.float64),
        tf.convert_to_tensor(season_sin_kt, dtype=tf.float64),
        tf.convert_to_tensor(season_cos_kt, dtype=tf.float64),
    )

    # Product-only binary DOW: gather along w-axis (2)
    beta_dow_j = tf.convert_to_tensor(theta["beta_dow_j"], dtype=tf.float64)  # (J,2)
    dow_j_jt = tf.gather(beta_dow_j, weekend_t, axis=1)  # (J,T)
    dow_term = tf.transpose(dow_j_jt)[None, :, :]  # (1,T,J) -> broadcast over M

    # Product-only intercept: broadcast to (M,J)
    beta_market_j = tf.convert_to_tensor(
        theta["beta_market_j"], dtype=tf.float64
    )  # (J,)
    base_mj = delta_mj + beta_market_j[None, :]  # (M,J)

    # Base utility without habit/peer: (M,T,J)
    base_mtj = base_mj[:, None, :] + S_m[:, :, None] + dow_term

    # Deterministic states from observed y.
    J_int = tf.shape(delta_mj)[1]
    x_mntj = _inside_choice_onehot(y_mit=y_mit, J=J_int)  # (M,N,T,J)

    H_mntj = compute_habit_stock_from_onehot(x_mntj=x_mntj, decay=decay)
    P_mntj = compute_peer_exposure_from_onehot(
        x_mntj=x_mntj, peer_adj_m=peer_adj_m, L=L
    )

    beta_habit = tf.convert_to_tensor(theta["beta_habit_j"], dtype=tf.float64)[
        None, None, None, :
    ]
    beta_peer = tf.convert_to_tensor(theta["beta_peer_j"], dtype=tf.float64)[
        None, None, None, :
    ]

    return base_mtj[:, None, :, :] + beta_habit * H_mntj + beta_peer * P_mntj


# =============================================================================
# Log-likelihood and prediction
# =============================================================================


def loglik_mnt_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Per-(market, consumer, time) log-likelihood contributions under MNL with outside option.

    Returns:
      ll_mnt: (M,N,T) float64
    """
    y_mit = tf.convert_to_tensor(y_mit, dtype=tf.int32)

    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        weekend_t=weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        L=L,
        decay=decay,
    )

    zeros_outside = tf.zeros_like(v_mntj[..., :1])
    logits = tf.concat([zeros_outside, v_mntj], axis=3)  # (M,N,T,J+1)

    logp = tf.nn.log_softmax(logits, axis=3)
    return tf.gather(logp, y_mit, axis=3, batch_dims=3)


def loglik_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Total log-likelihood scalar (sum over M,N,T)."""
    return tf.reduce_sum(
        loglik_mnt_from_theta(
            theta=theta,
            y_mit=y_mit,
            delta_mj=delta_mj,
            weekend_t=weekend_t,
            season_sin_kt=season_sin_kt,
            season_cos_kt=season_cos_kt,
            peer_adj_m=peer_adj_m,
            L=L,
            decay=decay,
        )
    )


def predict_choice_probs_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Predict choice probabilities under MNL with outside option.

    Returns:
      p_mntc: (M,N,T,J+1) float64, where c=0 is outside and c=j+1 is inside product j.
    """
    y_mit = tf.convert_to_tensor(y_mit, dtype=tf.int32)

    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        weekend_t=weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        L=L,
        decay=decay,
    )

    zeros_outside = tf.zeros_like(v_mntj[..., :1])
    logits = tf.concat([zeros_outside, v_mntj], axis=3)
    return tf.nn.softmax(logits, axis=3)
