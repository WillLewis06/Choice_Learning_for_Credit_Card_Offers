"""
bonus2_model.py

Pure TensorFlow core mechanics for Bonus Q2 (habit + peer + DOW + seasonality) with MNL choices.

This module is the "core model" layer. It contains:
  - Parameter transforms (z -> theta)
  - Deterministic state construction from observed choices:
      * habit stock H_{m,i,j,t}
      * peer exposure P_{m,i,j,t} using known social networks
  - Utility construction and MNL log-likelihood / prediction probabilities

Observed inputs (typically passed in an `inputs` dict by the posterior/estimator):
  y_mit          (M,N,T) int32  choices; 0=outside, j+1=inside product j
  delta_mj       (M,J)   f64    Phase-1 baseline utilities (fixed)
  dow_t          (T,)    int32  weekday index in {0..6}
  season_sin_kt  (K,T)   f64    sin((k+1)*season_angle_t[t]) basis
  season_cos_kt  (K,T)   f64    cos((k+1)*season_angle_t[t]) basis
  peer_adj_m     tuple of length M, each a tf.SparseTensor (N,N) adjacency
  L              scalar int32   peer lookback window length

Parameters (theta) are expected to be float64 tensors:
  beta_market_mj (M,J)
  beta_habit_j   (J,)
  beta_peer_j    (J,)
  decay_rate_j   (J,) in (0,1)
  beta_dow_m     (M,7)
  beta_dow_j     (J,7)
  a_m, b_m       (M,K)
  a_j, b_j       (J,K)

Utility (inside options j=1..J, with outside option utility 0):
  v_{m,i,j,t} =
      delta_{m,j}
    + beta_market_mj[m,j]
    + beta_habit_j[j] * H_{m,i,j,t}
    + beta_peer_j[j]  * P_{m,i,j,t}
    + beta_dow_m[m, dow_t] + beta_dow_j[j, dow_t]
    + S_m[m,t] + S_j[j,t]

Choice model:
  P(y=j | v) = exp(v_j) / (1 + sum_r exp(v_r)),  outside j=0 has v_0=0.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

PeerAdjacency = tuple[tf.SparseTensor, ...]


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Map unconstrained parameters z[*] to constrained theta[*].

    Expected dtypes:
      - All z tensors are float64.

    Required z keys and shapes:
      z_beta_market_mj  (M,J) -> beta_market_mj  real
      z_beta_habit_j    (J,)  -> beta_habit_j    real
      z_beta_peer_j     (J,)  -> beta_peer_j     real
      z_decay_rate_j    (J,)  -> decay_rate_j    in (0,1)
      z_beta_dow_m      (M,7) -> beta_dow_m      real
      z_beta_dow_j      (J,7) -> beta_dow_j      real
      z_a_m             (M,K) -> a_m             real
      z_b_m             (M,K) -> b_m             real
      z_a_j             (J,K) -> a_j             real
      z_b_j             (J,K) -> b_j             real

    Returns:
      dict[str, tf.Tensor] theta.
    """
    return {
        "beta_market_mj": z["z_beta_market_mj"],
        "beta_habit_j": z["z_beta_habit_j"],
        "beta_peer_j": z["z_beta_peer_j"],
        "decay_rate_j": tf.math.sigmoid(z["z_decay_rate_j"]),
        "beta_dow_m": z["z_beta_dow_m"],
        "beta_dow_j": z["z_beta_dow_j"],
        "a_m": z["z_a_m"],
        "b_m": z["z_b_m"],
        "a_j": z["z_a_j"],
        "b_j": z["z_b_j"],
    }


def _ensure_theta(
    theta: dict[str, tf.Tensor],
    decay_rate_eps: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Apply identifiability constraints and optional decay clipping.

    This enforces (in theta-space):
      - beta_dow_m: mean across weekdays is 0 within each market
      - beta_dow_j: mean across weekdays is 0 within each product, then mean across products is 0 per weekday
      - a_j, b_j: centered across products per harmonic
      - decay_rate_j: clipped to [eps, 1-eps] (eps may be 0)

    Expected dtypes:
      - theta[*] are float64
      - decay_rate_eps is float64 (or castable to float64)
    """
    eps = tf.cast(decay_rate_eps, tf.float64)

    th = dict(theta)
    th["decay_rate_j"] = tf.clip_by_value(th["decay_rate_j"], eps, 1.0 - eps)

    beta_dow_m, beta_dow_j, a_j, b_j = apply_identifiability_constraints_tf(
        beta_dow_m=th["beta_dow_m"],
        beta_dow_j=th["beta_dow_j"],
        a_j=th["a_j"],
        b_j=th["b_j"],
    )
    th["beta_dow_m"] = beta_dow_m
    th["beta_dow_j"] = beta_dow_j
    th["a_j"] = a_j
    th["b_j"] = b_j

    return th


def apply_identifiability_constraints_tf(
    beta_dow_m: tf.Tensor,
    beta_dow_j: tf.Tensor,
    a_j: tf.Tensor,
    b_j: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Center additive decomposition components.

    - beta_dow_m: center within market over weekdays
    - beta_dow_j: center within product over weekdays, then center across products per weekday
    - a_j, b_j: center across products per harmonic

    Expected dtypes: float64.
    """
    beta_dow_m = beta_dow_m - tf.reduce_mean(beta_dow_m, axis=1, keepdims=True)

    beta_dow_j = beta_dow_j - tf.reduce_mean(beta_dow_j, axis=1, keepdims=True)
    beta_dow_j = beta_dow_j - tf.reduce_mean(beta_dow_j, axis=0, keepdims=True)

    # These ops are safe with shape (J,0) when K == 0.
    a_j = a_j - tf.reduce_mean(a_j, axis=0, keepdims=True)
    b_j = b_j - tf.reduce_mean(b_j, axis=0, keepdims=True)

    return beta_dow_m, beta_dow_j, a_j, b_j


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
      y_mit: (M,N,T) int32, 0=outside, j+1=inside j
      J: scalar int32

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
    decay_rate_j: tf.Tensor,
) -> tf.Tensor:
    """Compute habit stocks H_{m,i,j,t} used in utility at each time t.

    Recurrence:
      H_{t+1} = decay_rate_j * H_t + x_t

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      decay_rate_j: (J,) float64

    Returns:
      H_mntj: (M,N,T,J) float64, where H[...,t,:] is the pre-choice stock at time t.
    """
    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])  # (T,M,N,J)

    M = tf.shape(x_mntj)[0]
    N = tf.shape(x_mntj)[1]
    J_int = tf.shape(x_mntj)[3]

    H0 = tf.zeros((M, N, J_int), dtype=tf.float64)

    def _step(H_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return H_curr * decay_rate_j[None, None, :] + x_curr

    H_next_t = tf.scan(_step, x_t, initializer=H0)  # (T,M,N,J) gives H_{t+1}

    # Pre-choice stock at time t is H_t; we have H_{t+1} from scan.
    H_curr_t = tf.concat([H0[None, ...], H_next_t[:-1, ...]], axis=0)  # (T,M,N,J)
    return tf.transpose(H_curr_t, perm=[1, 2, 0, 3])  # (M,N,T,J)


def rolling_window_counts(x_mntj: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """For each t, compute counts over the previous L periods (excluding current t).

      C[...,t,:] = sum_{u=max(0,t-L)}^{t-1} x[...,u,:]

    Args:
      x_mntj: (M,N,T,J) float64
      L: scalar int

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

    Definition:
      P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{y_{k,t-ell} == j}

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      peer_adj_m: tuple length M of tf.SparseTensor (N,N)
      L: scalar int

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
    dow_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay_rate_eps: tf.Tensor,
) -> tf.Tensor:
    """Compute inside-option utilities v_{m,i,j,t} (no outside option column).

    Returns:
      v_mntj: (M,N,T,J) float64
    """
    theta = _ensure_theta(theta=theta, decay_rate_eps=decay_rate_eps)

    y_mit = tf.convert_to_tensor(y_mit, dtype=tf.int32)
    delta_mj = tf.convert_to_tensor(delta_mj, dtype=tf.float64)
    dow_t = tf.convert_to_tensor(dow_t, dtype=tf.int32)

    # Seasonality: S_m (M,T), S_j (J,T)
    S_m = fourier_seasonality_tf(
        theta["a_m"], theta["b_m"], season_sin_kt, season_cos_kt
    )
    S_j = fourier_seasonality_tf(
        theta["a_j"], theta["b_j"], season_sin_kt, season_cos_kt
    )

    # DOW terms: gather along weekday axis.
    dow_m_mt = tf.gather(theta["beta_dow_m"], dow_t, axis=1)  # (M,T)
    dow_j_jt = tf.gather(theta["beta_dow_j"], dow_t, axis=1)  # (J,T)

    dow_mtj = dow_m_mt[:, :, None] + tf.transpose(dow_j_jt)[None, :, :]  # (M,T,J)
    season_mtj = S_m[:, :, None] + tf.transpose(S_j)[None, :, :]  # (M,T,J)

    base_mj = delta_mj + theta["beta_market_mj"]  # (M,J)
    base_mtj = base_mj[:, None, :] + dow_mtj + season_mtj  # (M,T,J)

    # Deterministic states from observed y.
    J_int = tf.shape(delta_mj)[1]
    x_mntj = _inside_choice_onehot(y_mit=y_mit, J=J_int)  # (M,N,T,J)
    H_mntj = compute_habit_stock_from_onehot(
        x_mntj=x_mntj, decay_rate_j=theta["decay_rate_j"]
    )
    P_mntj = compute_peer_exposure_from_onehot(
        x_mntj=x_mntj, peer_adj_m=peer_adj_m, L=L
    )

    beta_habit = theta["beta_habit_j"][None, None, None, :]
    beta_peer = theta["beta_peer_j"][None, None, None, :]

    return base_mtj[:, None, :, :] + beta_habit * H_mntj + beta_peer * P_mntj


# =============================================================================
# Log-likelihood and prediction
# =============================================================================


def loglik_mnt_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay_rate_eps: tf.Tensor,
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
        dow_t=dow_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        L=L,
        decay_rate_eps=decay_rate_eps,
    )

    zeros_outside = tf.zeros_like(v_mntj[..., :1])
    logits = tf.concat([zeros_outside, v_mntj], axis=3)  # (M,N,T,J+1)

    logp = tf.nn.log_softmax(logits, axis=3)
    return tf.gather(logp, y_mit, axis=3, batch_dims=3)


def loglik_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay_rate_eps: tf.Tensor,
) -> tf.Tensor:
    """Total log-likelihood scalar (sum over M,N,T)."""
    return tf.reduce_sum(
        loglik_mnt_from_theta(
            theta=theta,
            y_mit=y_mit,
            delta_mj=delta_mj,
            dow_t=dow_t,
            season_sin_kt=season_sin_kt,
            season_cos_kt=season_cos_kt,
            peer_adj_m=peer_adj_m,
            L=L,
            decay_rate_eps=decay_rate_eps,
        )
    )


def predict_choice_probs_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    decay_rate_eps: tf.Tensor,
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
        dow_t=dow_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        L=L,
        decay_rate_eps=decay_rate_eps,
    )

    zeros_outside = tf.zeros_like(v_mntj[..., :1])
    logits = tf.concat([zeros_outside, v_mntj], axis=3)
    return tf.nn.softmax(logits, axis=3)
