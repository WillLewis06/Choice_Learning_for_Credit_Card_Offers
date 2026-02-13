"""
bonus2_model.py

Pure TensorFlow core mechanics for Bonus Q2 (habit + peer + DOW + seasonality) with MNL choices.

This module is the "core model" layer (analogous to ching/stockpiling_model.py). It contains:
  - Parameter transforms (z -> theta)
  - Deterministic state construction from observed choices:
      * habit stock H_{m,i,j,t}
      * peer exposure P_{m,i,j,t} using known social networks
  - Utility construction and MNL log-likelihood / prediction probabilities

Observed inputs (typically passed in an `inputs` dict by the posterior/estimator):
  y_mit        (M,N,T) int   choices; 0=outside, j+1=inside product j
  delta_mj     (M,J)   f64   Phase-1 baseline utilities (fixed)
  dow_t        (T,)    int   weekday index in {0..6}
  sin_k_theta  (K,T)   f64   sin((k+1)*theta(t))
  cos_k_theta  (K,T)   f64   cos((k+1)*theta(t))
  peer_adj_m   tuple of length M, each a tf.SparseTensor (N,N) adjacency
  L            scalar int    peer lookback window length

Parameters (theta):
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

from typing import Iterable, Sequence

import tensorflow as tf

PeerAdjacency = tuple[tf.SparseTensor, ...]

ONE_F64 = tf.constant(1.0, dtype=tf.float64)


# =============================================================================
# Parameter transforms
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """
    Map unconstrained parameters z[*] to constrained theta[*].

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
        "beta_market_mj": tf.cast(z["z_beta_market_mj"], tf.float64),
        "beta_habit_j": tf.cast(z["z_beta_habit_j"], tf.float64),
        "beta_peer_j": tf.cast(z["z_beta_peer_j"], tf.float64),
        "decay_rate_j": tf.math.sigmoid(tf.cast(z["z_decay_rate_j"], tf.float64)),
        "beta_dow_m": tf.cast(z["z_beta_dow_m"], tf.float64),
        "beta_dow_j": tf.cast(z["z_beta_dow_j"], tf.float64),
        "a_m": tf.cast(z["z_a_m"], tf.float64),
        "b_m": tf.cast(z["z_b_m"], tf.float64),
        "a_j": tf.cast(z["z_a_j"], tf.float64),
        "b_j": tf.cast(z["z_b_j"], tf.float64),
    }


def _ensure_theta(
    theta: dict[str, tf.Tensor],
    eps_decay: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """
    Ensure float64 and apply identifiability constraints consistent with the DGP.

    This enforces (in theta-space):
      - beta_dow_m: mean across weekdays is 0 within each market
      - beta_dow_j: mean across weekdays is 0 within each product, then mean across products is 0 per weekday
      - a_j, b_j: centered across products per harmonic
      - decay_rate_j: clipped to [eps, 1-eps] if eps > 0
    """
    th = {k: tf.cast(v, tf.float64) for k, v in theta.items()}

    # Optional numerical guard for decay rates.
    eps_decay = tf.cast(eps_decay, tf.float64)
    if eps_decay.dtype != tf.float64:
        eps_decay = tf.cast(eps_decay, tf.float64)

    if tf.reduce_any(eps_decay > 0.0):
        th["decay_rate_j"] = tf.clip_by_value(
            th["decay_rate_j"], eps_decay, ONE_F64 - eps_decay
        )

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
    """
    TF version of the DGP centering constraints for additive decompositions.

    - beta_dow_m: center within market over weekdays
    - beta_dow_j: center within product over weekdays, then center across products per weekday
    - a_j, b_j: center across products per harmonic (if K>0)
    """
    beta_dow_m = tf.cast(beta_dow_m, tf.float64)
    beta_dow_j = tf.cast(beta_dow_j, tf.float64)
    a_j = tf.cast(a_j, tf.float64)
    b_j = tf.cast(b_j, tf.float64)

    beta_dow_m = beta_dow_m - tf.reduce_mean(beta_dow_m, axis=1, keepdims=True)

    beta_dow_j = beta_dow_j - tf.reduce_mean(beta_dow_j, axis=1, keepdims=True)
    beta_dow_j = beta_dow_j - tf.reduce_mean(beta_dow_j, axis=0, keepdims=True)

    # K may be 0; these ops are safe with shape (J,0).
    a_j = a_j - tf.reduce_mean(a_j, axis=0, keepdims=True)
    b_j = b_j - tf.reduce_mean(b_j, axis=0, keepdims=True)

    return beta_dow_m, beta_dow_j, a_j, b_j


# =============================================================================
# Social network adjacency (known / observed)
# =============================================================================


def neighbors_to_sparse_adj(
    nbrs_i: Sequence[Sequence[int] | tf.Tensor], N: int
) -> tf.SparseTensor:
    """
    Build a sparse adjacency matrix A of shape (N,N) with A[i,k]=1 if k in nbrs_i[i].

    nbrs_i[i] is the out-neighbor set observed by consumer i (no self edges assumed).
    """
    rows: list[int] = []
    cols: list[int] = []

    for i in range(int(N)):
        ni = nbrs_i[i]
        if isinstance(ni, tf.Tensor):
            ni_list = [int(x) for x in ni.numpy().tolist()]
        else:
            ni_list = [int(x) for x in ni]

        for k in ni_list:
            rows.append(i)
            cols.append(k)

    if len(rows) == 0:
        indices = tf.zeros((0, 2), dtype=tf.int64)
        values = tf.zeros((0,), dtype=tf.float64)
    else:
        indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
        values = tf.ones((len(rows),), dtype=tf.float64)

    A = tf.SparseTensor(indices=indices, values=values, dense_shape=(int(N), int(N)))
    return tf.sparse.reorder(A)


def build_peer_adjacency(
    nbrs_m: Sequence[Sequence[Sequence[int] | tf.Tensor]], N: int
) -> PeerAdjacency:
    """
    Convert per-market neighbor sets into per-market sparse adjacency tensors.

    Args:
      nbrs_m: length M; each element is nbrs_i for that market, length N
      N: number of consumers

    Returns:
      peer_adj_m: tuple length M of tf.SparseTensor (N,N)
    """
    return tuple(
        neighbors_to_sparse_adj(nbrs_i=nbrs_m[m], N=N) for m in range(len(nbrs_m))
    )


# =============================================================================
# Deterministic state construction from observed y
# =============================================================================


def _inside_choice_onehot(y_mit: tf.Tensor, J: tf.Tensor) -> tf.Tensor:
    """
    Build inside-choice one-hot indicators.

    Args:
      y_mit: (M,N,T) int, 0=outside, j+1=inside j
      J: scalar int

    Returns:
      x_mntj: (M,N,T,J) float64, x[...,t,j]=1{y[...,t]==j+1}
    """
    y_mit = tf.convert_to_tensor(y_mit)
    y_int = tf.cast(y_mit, tf.int32)

    J_int = tf.cast(J, tf.int32)
    y0 = tf.maximum(y_int - 1, 0)
    onehot = tf.one_hot(y0, depth=J_int, dtype=tf.float64)
    mask = tf.cast(y_int > 0, tf.float64)[..., None]
    return onehot * mask


def compute_habit_stock_from_choices(
    y_mit: tf.Tensor,
    decay_rate_j: tf.Tensor,
    J: tf.Tensor,
) -> tf.Tensor:
    """
    Compute habit stocks H_{m,i,j,t} used in utility at each time t.

    Recurrence:
      H_{t+1} = decay_rate_j * H_t + 1{y_t == j}

    Returns:
      H_mntj: (M,N,T,J) float64, where H[...,t,:] is the pre-choice stock at time t.
    """
    decay_rate_j = tf.cast(decay_rate_j, tf.float64)
    x_mntj = _inside_choice_onehot(y_mit=y_mit, J=J)  # (M,N,T,J)

    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])  # (T,M,N,J)
    M = tf.shape(x_mntj)[0]
    N = tf.shape(x_mntj)[1]
    J_int = tf.shape(x_mntj)[3]

    H0 = tf.zeros((M, N, J_int), dtype=tf.float64)

    def _step(H_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return H_curr * decay_rate_j[None, None, :] + x_curr

    H_next_t = tf.scan(_step, x_t, initializer=H0)  # (T,M,N,J) gives H_{t+1}

    H_curr_t = tf.concat(
        [H0[None, ...], H_next_t[:-1, ...]], axis=0
    )  # (T,M,N,J) gives H_t
    return tf.transpose(H_curr_t, perm=[1, 2, 0, 3])  # (M,N,T,J)


def rolling_window_counts(x_mntj: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """
    For each t, compute counts over the previous L periods (excluding current t):

      C[...,t,:] = sum_{u=max(0,t-L)}^{t-1} x[...,u,:]

    Args:
      x_mntj: (M,N,T,J) float64
      L: scalar int

    Returns:
      C_mntj: (M,N,T,J) float64
    """
    x_mntj = tf.cast(x_mntj, tf.float64)
    L_int = tf.cast(L, tf.int32)
    T = tf.shape(x_mntj)[2]

    S = tf.cumsum(x_mntj, axis=2)  # inclusive
    Z = tf.zeros_like(S[:, :, :1, :])
    S_pad = tf.concat([Z, S], axis=2)  # (M,N,T+1,J), prefix sums

    t_idx = tf.range(T, dtype=tf.int32)  # 0..T-1
    start_idx = tf.maximum(0, t_idx - L_int)

    S_end = tf.gather(S_pad, t_idx, axis=2)  # sum_{u< t} x_u
    S_start = tf.gather(S_pad, start_idx, axis=2)  # sum_{u< start} x_u

    return S_end - S_start


def compute_peer_exposure_from_adj(
    y_mit: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    J: tf.Tensor,
) -> tf.Tensor:
    """
    Compute peer exposures P_{m,i,j,t} using known per-market adjacency matrices.

    Definition:
      P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{y_{k,t-ell} == j}

    Returns:
      P_mntj: (M,N,T,J) float64
    """
    x_mntj = _inside_choice_onehot(y_mit=y_mit, J=J)  # (M,N,T,J)
    C_mntj = rolling_window_counts(x_mntj=x_mntj, L=L)  # (M,N,T,J)

    N = tf.shape(C_mntj)[1]
    T = tf.shape(C_mntj)[2]
    J_int = tf.shape(C_mntj)[3]

    P_list: list[tf.Tensor] = []
    for m in range(len(peer_adj_m)):
        A = peer_adj_m[m]
        C_m = C_mntj[m]  # (N,T,J)
        C_flat = tf.reshape(C_m, (N, -1))  # (N, T*J)
        P_flat = tf.sparse.sparse_dense_matmul(A, C_flat)  # (N, T*J)
        P_list.append(tf.reshape(P_flat, (N, T, J_int)))

    return tf.stack(P_list, axis=0)  # (M,N,T,J)


# =============================================================================
# Seasonality and utilities
# =============================================================================


def fourier_seasonality_tf(
    a_coeff: tf.Tensor,
    b_coeff: tf.Tensor,
    sin_k_theta: tf.Tensor,
    cos_k_theta: tf.Tensor,
) -> tf.Tensor:
    """
    Compute Fourier seasonal series:

      S[r,t] = sum_k a[r,k]*sin[k,t] + b[r,k]*cos[k,t]

    Shapes:
      a_coeff, b_coeff: (R,K)
      sin_k_theta, cos_k_theta: (K,T)

    Returns:
      S: (R,T)
    """
    a_coeff = tf.cast(a_coeff, tf.float64)
    b_coeff = tf.cast(b_coeff, tf.float64)
    sin_k_theta = tf.cast(sin_k_theta, tf.float64)
    cos_k_theta = tf.cast(cos_k_theta, tf.float64)
    return tf.linalg.matmul(a_coeff, sin_k_theta) + tf.linalg.matmul(
        b_coeff, cos_k_theta
    )


def utilities_mntj_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    sin_k_theta: tf.Tensor,
    cos_k_theta: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    eps_decay: tf.Tensor,
) -> tf.Tensor:
    """
    Compute inside-option utilities v_{m,i,j,t} (no outside option column).

    Returns:
      v_mntj: (M,N,T,J) float64
    """
    theta = _ensure_theta(theta=theta, eps_decay=eps_decay)

    y_mit = tf.convert_to_tensor(y_mit)
    delta_mj = tf.cast(delta_mj, tf.float64)
    dow_t = tf.cast(dow_t, tf.int32)

    M = tf.shape(delta_mj)[0]
    J_int = tf.shape(delta_mj)[1]

    # Seasonality: S_m (M,T), S_j (J,T)
    S_m = fourier_seasonality_tf(theta["a_m"], theta["b_m"], sin_k_theta, cos_k_theta)
    S_j = fourier_seasonality_tf(theta["a_j"], theta["b_j"], sin_k_theta, cos_k_theta)

    # DOW terms: gather along weekday axis.
    dow_m_mt = tf.gather(theta["beta_dow_m"], dow_t, axis=1)  # (M,T)
    dow_j_jt = tf.gather(theta["beta_dow_j"], dow_t, axis=1)  # (J,T)

    dow_mtj = (
        dow_m_mt[:, :, None] + tf.transpose(dow_j_jt, perm=[1, 0])[None, :, :]
    )  # (M,T,J)
    season_mtj = S_m[:, :, None] + tf.transpose(S_j, perm=[1, 0])[None, :, :]  # (M,T,J)

    base_mj = delta_mj + theta["beta_market_mj"]  # (M,J)
    base_mtj = base_mj[:, None, :] + dow_mtj + season_mtj  # (M,T,J)

    # Deterministic states from observed y:
    H_mntj = compute_habit_stock_from_choices(
        y_mit=y_mit, decay_rate_j=theta["decay_rate_j"], J=J_int
    )
    P_mntj = compute_peer_exposure_from_adj(
        y_mit=y_mit, peer_adj_m=peer_adj_m, L=L, J=J_int
    )

    beta_habit = theta["beta_habit_j"][None, None, None, :]  # (1,1,1,J)
    beta_peer = theta["beta_peer_j"][None, None, None, :]  # (1,1,1,J)

    v_mntj = (
        base_mtj[:, None, :, :] + beta_habit * H_mntj + beta_peer * P_mntj
    )  # (M,N,T,J)
    return v_mntj


# =============================================================================
# Log-likelihood and prediction
# =============================================================================


def loglik_mnt_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    sin_k_theta: tf.Tensor,
    cos_k_theta: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    eps_decay: tf.Tensor,
) -> tf.Tensor:
    """
    Per-(market, consumer, time) log-likelihood contributions under MNL with outside option.

    Returns:
      ll_mnt: (M,N,T) float64
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        dow_t=dow_t,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    )

    v_mntj = tf.cast(v_mntj, tf.float64)
    M = tf.shape(v_mntj)[0]
    N = tf.shape(v_mntj)[1]
    T = tf.shape(v_mntj)[2]

    zeros_outside = tf.zeros((M, N, T, 1), dtype=tf.float64)
    logits = tf.concat([zeros_outside, v_mntj], axis=3)  # (M,N,T,J+1)

    logp = tf.nn.log_softmax(logits, axis=3)  # (M,N,T,J+1)
    y_int = tf.cast(tf.convert_to_tensor(y_mit), tf.int32)  # (M,N,T)

    return tf.gather(logp, y_int, axis=3, batch_dims=3)  # (M,N,T)


def loglik_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    sin_k_theta: tf.Tensor,
    cos_k_theta: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    eps_decay: tf.Tensor,
) -> tf.Tensor:
    """
    Total log-likelihood scalar (sum over M,N,T).
    """
    ll_mnt = loglik_mnt_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        dow_t=dow_t,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    )
    return tf.reduce_sum(ll_mnt)


def predict_choice_probs_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    dow_t: tf.Tensor,
    sin_k_theta: tf.Tensor,
    cos_k_theta: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    L: tf.Tensor,
    eps_decay: tf.Tensor,
) -> tf.Tensor:
    """
    Predict choice probabilities under MNL with outside option.

    Returns:
      p_mntc: (M,N,T,J+1) float64, where c=0 is outside and c=j+1 is inside product j.
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        dow_t=dow_t,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    )

    v_mntj = tf.cast(v_mntj, tf.float64)
    M = tf.shape(v_mntj)[0]
    N = tf.shape(v_mntj)[1]
    T = tf.shape(v_mntj)[2]

    zeros_outside = tf.zeros((M, N, T, 1), dtype=tf.float64)
    logits = tf.concat([zeros_outside, v_mntj], axis=3)
    return tf.nn.softmax(logits, axis=3)
