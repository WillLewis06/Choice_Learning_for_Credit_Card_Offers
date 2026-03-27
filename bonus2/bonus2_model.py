"""
bonus2_model.py

TensorFlow model kernels for the Bonus Q2 simulation/estimator.

Design
- Deterministic state construction from observed choices is separated from repeated posterior evaluation.
- The repeated likelihood path consumes precomputed state tensors:
    H_mntj  pre-choice habit stock
    P_mntj  peer exposure count
- Pure tensor kernels on the repeated path are compiled with jit_compile=True.

Model components
- Outside-option multinomial logit:
    choice c = 0 is outside
    choice c = j + 1 is inside product j
- Deterministic states from observed choices:
    H_{t+1} = decay * H_t + 1{y_t = j}
    P_{i,j,t} = sum_{k in N(i)} sum_{ell=1..L} 1{y_{k,t-ell} = j}
- Additive inside-option utility:
    delta_mj
    + beta_intercept_j
    + beta_weekend_jw[j, w(t)]
    + S_m(t)
    + beta_habit_j * H_mntj
    + beta_peer_j  * P_mntj

Input validation is intentionally not performed here.
All tensors are assumed to have correct shapes and dtypes.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

PeerAdjacency = tuple[tf.SparseTensor, ...]
TensorDict = dict[str, tf.Tensor]


# =============================================================================
# Parameter mapping
# =============================================================================


def unconstrained_to_theta(z: TensorDict) -> TensorDict:
    """Map unconstrained sampler state z[*] to structural parameter tensors theta[*].

    Required z keys (float64):
      z_beta_intercept_j  (J,)
      z_beta_habit_j      (J,)
      z_beta_peer_j       (J,)
      z_beta_weekend_jw   (J, 2)   where w=0 weekday, w=1 weekend
      z_a_m               (M, K)
      z_b_m               (M, K)
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
# Known social network input
# =============================================================================


def neighbors_to_sparse_adj(
    neighbors_i: Sequence[Sequence[int]],
    n_consumers: int,
) -> tf.SparseTensor:
    """Build sparse adjacency A of shape (N, N) with A[i, k] = 1 if k is a neighbor of i."""
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

    adjacency = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=(int(n_consumers), int(n_consumers)),
    )
    return tf.sparse.reorder(adjacency)


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


@tf.function(jit_compile=True, reduce_retracing=True)
def _inside_choice_onehot(y_mit: tf.Tensor, n_products: tf.Tensor) -> tf.Tensor:
    """Convert observed choices to inside-product one-hot indicators.

    Args:
      y_mit: (M, N, T) int32, where 0=outside and c=j+1 is inside product j
      n_products: scalar int32, number of inside products J

    Returns:
      x_mntj: (M, N, T, J) float64 with x[..., t, j] = 1{y[..., t] == j+1}
    """
    y0 = tf.maximum(y_mit - 1, 0)
    onehot = tf.one_hot(y0, depth=n_products, dtype=tf.float64)
    inside_mask = tf.cast(y_mit > 0, tf.float64)[..., None]
    return onehot * inside_mask


@tf.function(jit_compile=True, reduce_retracing=True)
def habit_stock_pre_choice(x_mntj: tf.Tensor, decay: tf.Tensor) -> tf.Tensor:
    """Build pre-choice habit stock H_t for each period.

    Recurrence:
      H_{t+1} = decay * H_t + x_t

    Args:
      x_mntj: (M, N, T, J) float64 inside-choice indicators
      decay: scalar float64

    Returns:
      H_mntj: (M, N, T, J) float64 where H[..., t, :] is the pre-choice stock at t
    """
    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])  # (T, M, N, J)

    m = tf.shape(x_mntj)[0]
    n = tf.shape(x_mntj)[1]
    j = tf.shape(x_mntj)[3]

    h0 = tf.zeros((m, n, j), dtype=tf.float64)

    def step(h_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return decay * h_curr + x_curr

    h_next_t = tf.scan(step, x_t, initializer=h0)  # (T, M, N, J) = H_{t+1}
    h_curr_t = tf.concat([h0[None, ...], h_next_t[:-1, ...]], axis=0)  # (T, M, N, J)

    return tf.transpose(h_curr_t, perm=[1, 2, 0, 3])  # (M, N, T, J)


@tf.function(jit_compile=True, reduce_retracing=True)
def rolling_lookback_sum(x_mntj: tf.Tensor, lookback: tf.Tensor) -> tf.Tensor:
    """Sum over the previous L periods, excluding the current period.

    Definition:
      c[..., t, :] = sum_{u=max(0, t-L)}^{t-1} x[..., u, :]

    Args:
      x_mntj: (M, N, T, J) float64
      lookback: scalar int32, L

    Returns:
      c_mntj: (M, N, T, J) float64
    """
    t = tf.shape(x_mntj)[2]

    prefix = tf.cumsum(x_mntj, axis=2)
    prefix = tf.concat(
        [tf.zeros_like(prefix[:, :, :1, :]), prefix], axis=2
    )  # (M, N, T+1, J)

    t_idx = tf.range(t, dtype=tf.int32)
    start_idx = tf.maximum(0, t_idx - lookback)

    prefix_end = tf.gather(prefix, t_idx, axis=2)
    prefix_start = tf.gather(prefix, start_idx, axis=2)

    return prefix_end - prefix_start


def peer_exposure_from_onehot(
    x_mntj: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
) -> tf.Tensor:
    """Build peer exposure counts from known within-market adjacency and lookback L.

    Args:
      x_mntj: (M, N, T, J) float64 inside-choice indicators
      peer_adj_m: tuple length M of sparse adjacency tensors, each of shape (N, N)
      lookback: scalar int32, L

    Returns:
      P_mntj: (M, N, T, J) float64
    """
    counts_mntj = rolling_lookback_sum(x_mntj=x_mntj, lookback=lookback)  # (M, N, T, J)

    n = tf.shape(counts_mntj)[1]
    t = tf.shape(counts_mntj)[2]
    j = tf.shape(counts_mntj)[3]

    per_market: list[tf.Tensor] = []
    for m in range(len(peer_adj_m)):
        counts_flat = tf.reshape(counts_mntj[m], (n, -1))  # (N, T*J)
        exposed_flat = tf.sparse.sparse_dense_matmul(
            peer_adj_m[m], counts_flat
        )  # (N, T*J)
        per_market.append(tf.reshape(exposed_flat, (n, t, j)))  # (N, T, J)

    return tf.stack(per_market, axis=0)  # (M, N, T, J)


def build_deterministic_states(
    y_mit: tf.Tensor,
    n_products: tf.Tensor,
    peer_adj_m: PeerAdjacency,
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build deterministic state tensors once from observed choices.

    Args:
      y_mit: (M, N, T) int32
      n_products: scalar int32, number of inside products J
      peer_adj_m: tuple length M of sparse adjacency tensors, each of shape (N, N)
      lookback: scalar int32, L
      decay: scalar float64

    Returns:
      x_mntj: (M, N, T, J) float64 inside-choice indicators
      H_mntj: (M, N, T, J) float64 pre-choice habit stock
      P_mntj: (M, N, T, J) float64 peer exposure count
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


@tf.function(jit_compile=True, reduce_retracing=True)
def market_fourier_seasonality(
    a_mk: tf.Tensor,
    b_mk: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
) -> tf.Tensor:
    """Compute market seasonal component S_m(t) from Fourier coefficients and basis.

    Args:
      a_mk: (M, K) float64
      b_mk: (M, K) float64
      season_sin_kt: (K, T) float64
      season_cos_kt: (K, T) float64

    Returns:
      season_mt: (M, T) float64
    """
    return tf.linalg.matmul(a_mk, season_sin_kt) + tf.linalg.matmul(b_mk, season_cos_kt)


@tf.function(jit_compile=True, reduce_retracing=True)
def utilities_mntj_from_theta(
    theta: TensorDict,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute inside-option utilities v_{m,i,t,j}.

    Args:
      theta: structural parameter dictionary with keys
        beta_intercept_j  (J,)
        beta_habit_j      (J,)
        beta_peer_j       (J,)
        beta_weekend_jw   (J, 2)
        a_m               (M, K)
        b_m               (M, K)
      delta_mj: (M, J) float64
      is_weekend_t: (T,) int32 in {0, 1}
      season_sin_kt: (K, T) float64
      season_cos_kt: (K, T) float64
      h_mntj: (M, N, T, J) float64
      p_mntj: (M, N, T, J) float64

    Returns:
      v_mntj: (M, N, T, J) float64
    """
    season_mt = market_fourier_seasonality(
        a_mk=theta["a_m"],
        b_mk=theta["b_m"],
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
    )  # (M, T)

    weekend_jt = tf.gather(theta["beta_weekend_jw"], is_weekend_t, axis=1)  # (J, T)
    weekend_tj = tf.transpose(weekend_jt, perm=[1, 0])  # (T, J)

    base_mj = delta_mj + theta["beta_intercept_j"][None, :]  # (M, J)
    base_mtj = (
        base_mj[:, None, :] + season_mt[:, :, None] + weekend_tj[None, :, :]
    )  # (M, T, J)

    beta_habit = theta["beta_habit_j"][None, None, None, :]  # (1, 1, 1, J)
    beta_peer = theta["beta_peer_j"][None, None, None, :]  # (1, 1, 1, J)

    return base_mtj[:, None, :, :] + beta_habit * h_mntj + beta_peer * p_mntj


# =============================================================================
# Log-likelihood and prediction
# =============================================================================


@tf.function(jit_compile=True, reduce_retracing=True)
def loglik_mnt_from_theta(
    theta: TensorDict,
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute per-(market, consumer, time) log-likelihood contributions.

    Args:
      theta: structural parameter dictionary
      y_mit: (M, N, T) int32 with 0=outside and c=j+1 for inside product j
      delta_mj: (M, J) float64
      is_weekend_t: (T,) int32 in {0, 1}
      season_sin_kt: (K, T) float64
      season_cos_kt: (K, T) float64
      h_mntj: (M, N, T, J) float64
      p_mntj: (M, N, T, J) float64

    Returns:
      ll_mnt: (M, N, T) float64
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
    )

    logits = tf.concat(
        [tf.zeros_like(v_mntj[..., :1]), v_mntj], axis=3
    )  # (M, N, T, J+1)
    logp = tf.nn.log_softmax(logits, axis=3)
    return tf.gather(logp, y_mit, axis=3, batch_dims=3)


@tf.function(jit_compile=True, reduce_retracing=True)
def loglik_from_theta(
    theta: TensorDict,
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute total log-likelihood summed over markets, consumers, and time."""
    ll_mnt = loglik_mnt_from_theta(
        theta=theta,
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
    )
    return tf.reduce_sum(ll_mnt)


@tf.function(jit_compile=True, reduce_retracing=True)
def predict_choice_probs_from_theta(
    theta: TensorDict,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute MNL choice probabilities with the outside option included.

    Args:
      theta: structural parameter dictionary
      delta_mj: (M, J) float64
      is_weekend_t: (T,) int32 in {0, 1}
      season_sin_kt: (K, T) float64
      season_cos_kt: (K, T) float64
      h_mntj: (M, N, T, J) float64
      p_mntj: (M, N, T, J) float64

    Returns:
      probs_mntc: (M, N, T, J+1) float64 where c=0 is outside and c=j+1 is inside product j
    """
    v_mntj = utilities_mntj_from_theta(
        theta=theta,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
    )

    logits = tf.concat([tf.zeros_like(v_mntj[..., :1]), v_mntj], axis=3)
    return tf.nn.softmax(logits, axis=3)
