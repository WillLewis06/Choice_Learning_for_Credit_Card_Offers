"""
bonus2_model.py

TensorFlow kernels for the Bonus Q2 simulation/estimator.

Model components implemented here
- Outside-option multinomial logit: choice c=0 is outside, c=j+1 is inside product j.
- Deterministic states built from observed choices:
  - Habit stock per consumer-product with known scalar decay in (0,1):
      H_{t+1} = decay * H_t + 1{y_t = j}
    Utilities use the pre-choice stock H_t.
  - Peer exposure per consumer-product over a lookback window L >= 1:
      P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{y_{k,t-ell} = j}
- Additive utility terms for inside options:
  - Phase-1 baseline utilities delta_mj (fixed input)
  - Product intercept shift beta_intercept_j
  - Product weekend shift beta_weekend_jw with w in {0,1}
  - Market seasonality S_m(t) via Fourier coefficients (a_mk, b_mk) and provided basis

Input validation is intentionally not performed in this file. All tensors are assumed to have
the documented shapes and dtypes by construction.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

PeerAdjacency = tuple[tf.SparseTensor, ...]


# =============================================================================
# Parameter mapping
# =============================================================================


def unconstrained_to_theta(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Rename unconstrained parameter tensors z[*] to structural tensors theta[*].

    Required z keys (float64):
      z_beta_intercept_j  (J,)
      z_beta_habit_j      (J,)
      z_beta_peer_j       (J,)
      z_beta_weekend_jw   (J,2)  (w=0 weekday, w=1 weekend)
      z_a_m               (M,K)
      z_b_m               (M,K)
    """
    return {
        "beta_intercept_j": z["z_beta_intercept_j"],
        "beta_habit_j": z["z_beta_habit_j"],
        "beta_peer_j": z["z_beta_peer_j"],
        "beta_weekend_jw": z["z_beta_weekend_jw"],
        "a_m": z["z_a_m"],
        "b_m": z["z_b_m"],
    }


# =============================================================================
# Social network adjacency (known input)
# =============================================================================


def neighbors_to_sparse_adj(
    neighbors_i: Sequence[Sequence[int]],
    n_consumers: int,
) -> tf.SparseTensor:
    """Build a sparse adjacency matrix A of shape (N,N) with A[i,k]=1 if k is a neighbor of i."""
    rows: list[int] = []
    cols: list[int] = []

    for i in range(int(n_consumers)):
        for k in neighbors_i[i]:
            rows.append(i)
            cols.append(int(k))

    if rows:
        indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
        values = tf.ones((len(rows),), dtype=tf.float64)
    else:
        indices = tf.zeros((0, 2), dtype=tf.int64)
        values = tf.zeros((0,), dtype=tf.float64)

    adj = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=(int(n_consumers), int(n_consumers)),
    )
    return tf.sparse.reorder(adj)


def build_peer_adjacency(
    neighbors_m: Sequence[Sequence[Sequence[int]]],
    n_consumers: int,
) -> PeerAdjacency:
    """Convert per-market neighbor lists into per-market sparse adjacency tensors."""
    return tuple(
        neighbors_to_sparse_adj(
            neighbors_i=neighbors_m[m],
            n_consumers=n_consumers,
        )
        for m in range(len(neighbors_m))
    )


# =============================================================================
# Deterministic state construction from observed choices
# =============================================================================


def _inside_choice_onehot(y_mit: tf.Tensor, n_products: tf.Tensor) -> tf.Tensor:
    """Inside-choice indicators.

    Args:
      y_mit: (M,N,T) int32, 0=outside, c=j+1 for inside product j
      n_products: scalar int32, number of inside products J

    Returns:
      x_mntj: (M,N,T,J) float64, x[...,t,j]=1{y[...,t]==j+1}
    """
    # Map {0,1..J} -> {0,0..J-1} and mask out outside choices.
    y0 = tf.maximum(y_mit - 1, 0)
    onehot = tf.one_hot(y0, depth=n_products, dtype=tf.float64)
    outside_mask = tf.cast(y_mit > 0, tf.float64)[..., None]
    return onehot * outside_mask


def habit_stock_pre_choice(x_mntj: tf.Tensor, decay: tf.Tensor) -> tf.Tensor:
    """Pre-choice habit stock H_t for each period t.

    Recurrence:
      H_{t+1} = decay * H_t + x_t

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      decay: scalar float64 in (0,1)

    Returns:
      H_mntj: (M,N,T,J) float64, H[...,t,:] equals H_t (pre-choice at t)
    """
    # Scan over time to produce H_{t+1}; then shift to obtain H_t.
    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])  # (T,M,N,J)

    m = tf.shape(x_mntj)[0]
    n = tf.shape(x_mntj)[1]
    j = tf.shape(x_mntj)[3]

    h0 = tf.zeros((m, n, j), dtype=tf.float64)

    def step(h_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return h_curr * decay + x_curr

    h_next_t = tf.scan(step, x_t, initializer=h0)  # (T,M,N,J) = H_{t+1}

    # Pre-choice stock at time t is H_t; prepend H_0 and drop the last H_{T}.
    h_curr_t = tf.concat([h0[None, ...], h_next_t[:-1, ...]], axis=0)  # (T,M,N,J)
    return tf.transpose(h_curr_t, perm=[1, 2, 0, 3])  # (M,N,T,J)


def rolling_lookback_sum(x_mntj: tf.Tensor, lookback: tf.Tensor) -> tf.Tensor:
    """Sum over the previous L periods (excluding the current period).

    Definition:
      c[...,t,:] = sum_{u=max(0,t-L)}^{t-1} x[...,u,:]

    Args:
      x_mntj: (M,N,T,J) float64
      lookback: scalar int32, L

    Returns:
      c_mntj: (M,N,T,J) float64
    """
    t = tf.shape(x_mntj)[2]

    # Inclusive prefix sums; pad with a leading zero so prefix at index t is sum_{u < t}.
    prefix = tf.cumsum(x_mntj, axis=2)
    prefix = tf.concat(
        [tf.zeros_like(prefix[:, :, :1, :]), prefix], axis=2
    )  # (M,N,T+1,J)

    t_idx = tf.range(t, dtype=tf.int32)
    start_idx = tf.maximum(0, t_idx - lookback)

    prefix_end = tf.gather(prefix, t_idx, axis=2)  # sum_{u < t} x_u
    prefix_start = tf.gather(prefix, start_idx, axis=2)  # sum_{u < start} x_u

    return prefix_end - prefix_start


def peer_exposure_from_onehot(
    x_mntj: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
) -> tf.Tensor:
    """Peer exposure counts from known within-market adjacency and lookback L.

    Args:
      x_mntj: (M,N,T,J) float64 inside-choice indicators
      peer_adj_m: tuple length M of tf.SparseTensor (N,N), float64 values
      lookback: scalar int32, L

    Returns:
      p_mntj: (M,N,T,J) float64
    """
    counts_mntj = rolling_lookback_sum(x_mntj=x_mntj, lookback=lookback)  # (M,N,T,J)

    n = tf.shape(counts_mntj)[1]
    t = tf.shape(counts_mntj)[2]
    j = tf.shape(counts_mntj)[3]

    per_market: list[tf.Tensor] = []
    for m in range(len(peer_adj_m)):
        # Flatten (T,J) so each adjacency multiplication aggregates neighbors for every (t,j) cell.
        counts_flat = tf.reshape(counts_mntj[m], (n, -1))  # (N, T*J)
        exposed_flat = tf.sparse.sparse_dense_matmul(
            peer_adj_m[m], counts_flat
        )  # (N, T*J)
        per_market.append(tf.reshape(exposed_flat, (n, t, j)))  # (N,T,J)

    return tf.stack(per_market, axis=0)  # (M,N,T,J)


def build_deterministic_states(
    y_mit: tf.Tensor,
    n_products: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build deterministic state tensors (x, H, P) from observed choices.

    Returns:
      x_mntj: (M,N,T,J) float64  inside-choice indicators
      H_mntj: (M,N,T,J) float64  pre-choice habit stocks
      P_mntj: (M,N,T,J) float64  peer exposure counts
    """
    x_mntj = _inside_choice_onehot(y_mit=y_mit, n_products=n_products)
    h_mntj = habit_stock_pre_choice(x_mntj=x_mntj, decay=decay)
    p_mntj = peer_exposure_from_onehot(
        x_mntj=x_mntj,
        peer_adj_m=peer_adj_m,
        lookback=lookback,
    )
    return x_mntj, h_mntj, p_mntj


# =============================================================================
# Seasonality and utilities
# =============================================================================


def market_fourier_seasonality(
    a_mk: tf.Tensor,
    b_mk: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
) -> tf.Tensor:
    """Market seasonal component S_m(t) from Fourier coefficients and basis.

    Shapes:
      a_mk, b_mk: (M,K)
      season_sin_kt, season_cos_kt: (K,T)

    Returns:
      s_mt: (M,T) float64
    """
    return tf.linalg.matmul(a_mk, season_sin_kt) + tf.linalg.matmul(b_mk, season_cos_kt)


def utilities_mntj_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Inside-option utilities v_{m,i,j,t} (no outside-option column).

    Args:
      theta: parameter dictionary (float64 tensors)
      y_mit: (M,N,T) int32
      delta_mj: (M,J) float64
      is_weekend_t: (T,) int32 in {0,1}
      season_sin_kt, season_cos_kt: (K,T) float64
      peer_adj_m: tuple length M of sparse adjacency (N,N)
      lookback: scalar int32, L
      decay: scalar float64

    Returns:
      v_mntj: (M,N,T,J) float64
    """
    # Seasonal utility component per market and time.
    season_mt = market_fourier_seasonality(
        a_mk=theta["a_m"],
        b_mk=theta["b_m"],
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
    )  # (M,T)

    # Weekend effect is product-specific: beta_weekend_jw[j,w].
    weekend_jt = tf.gather(theta["beta_weekend_jw"], is_weekend_t, axis=1)  # (J,T)
    weekend_tj = tf.transpose(weekend_jt)  # (T,J)

    # Base utility (no habit/peer) for each market-time-product: (M,T,J).
    base_mj = delta_mj + theta["beta_intercept_j"][None, :]  # (M,J)
    base_mtj = base_mj[:, None, :] + season_mt[:, :, None] + weekend_tj[None, :, :]

    # Deterministic states from observed choices.
    n_products = tf.shape(delta_mj)[1]
    _, h_mntj, p_mntj = build_deterministic_states(
        y_mit=y_mit,
        n_products=n_products,
        peer_adj_m=peer_adj_m,
        lookback=lookback,
        decay=decay,
    )

    beta_habit = theta["beta_habit_j"][None, None, None, :]  # broadcast over (M,N,T)
    beta_peer = theta["beta_peer_j"][None, None, None, :]

    return base_mtj[:, None, :, :] + beta_habit * h_mntj + beta_peer * p_mntj


# =============================================================================
# Log-likelihood and prediction
# =============================================================================


def loglik_mnt_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Per-(market, consumer, time) log-likelihood contributions under MNL with outside option.

    Returns:
      ll_mnt: (M,N,T) float64
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        lookback=lookback,
        decay=decay,
    )

    # Add an outside-option logit of 0 and evaluate log-softmax.
    logits = tf.concat([tf.zeros_like(v_mntj[..., :1]), v_mntj], axis=3)  # (M,N,T,J+1)
    logp = tf.nn.log_softmax(logits, axis=3)

    # Gather log-probability for the realized choice y_mit along the choice axis.
    return tf.gather(logp, y_mit, axis=3, batch_dims=3)


def loglik_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Total log-likelihood scalar (sum over markets, consumers, and time)."""
    return tf.reduce_sum(
        loglik_mnt_from_theta(
            theta=theta,
            y_mit=y_mit,
            delta_mj=delta_mj,
            is_weekend_t=is_weekend_t,
            season_sin_kt=season_sin_kt,
            season_cos_kt=season_cos_kt,
            peer_adj_m=peer_adj_m,
            lookback=lookback,
            decay=decay,
        )
    )


def predict_choice_probs_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tf.Tensor:
    """Choice probabilities under MNL with outside option.

    Returns:
      p_mntc: (M,N,T,J+1) float64, c=0 is outside and c=j+1 is inside product j.
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        peer_adj_m=peer_adj_m,
        lookback=lookback,
        decay=decay,
    )

    logits = tf.concat([tf.zeros_like(v_mntj[..., :1]), v_mntj], axis=3)
    return tf.nn.softmax(logits, axis=3)
