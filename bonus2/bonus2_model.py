"""TensorFlow kernels for the Bonus Q2 model.

This module builds deterministic state tensors from observed choices and
provides compiled utility, likelihood, and prediction kernels. Input
validation is handled elsewhere.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf


def neighbors_to_sparse_adj(
    neighbors_by_consumer: Sequence[Sequence[int]],
    n_consumers: int,
) -> tf.SparseTensor:
    """Build a sparse adjacency matrix from neighbor lists."""
    rows: list[int] = []
    cols: list[int] = []

    for i in range(n_consumers):
        for k in neighbors_by_consumer[i]:
            rows.append(i)
            cols.append(k)

    if rows:
        indices = tf.constant(list(zip(rows, cols)), dtype=tf.int64)
        values = tf.ones((len(rows),), dtype=tf.float64)
    else:
        indices = tf.zeros((0, 2), dtype=tf.int64)
        values = tf.zeros((0,), dtype=tf.float64)

    return tf.sparse.reorder(
        tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(n_consumers, n_consumers),
        )
    )


def build_peer_adjacency(
    neighbors_m: Sequence[Sequence[Sequence[int]]],
    n_consumers: int,
) -> tuple[tf.SparseTensor, ...]:
    """Build one sparse adjacency matrix per market."""
    return tuple(
        neighbors_to_sparse_adj(
            neighbors_by_consumer=neighbors_m[m],
            n_consumers=n_consumers,
        )
        for m in range(len(neighbors_m))
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def _inside_choice_onehot(y_mit: tf.Tensor, n_products: tf.Tensor) -> tf.Tensor:
    """Convert observed choices to inside-product one-hot indicators."""
    inside_index = tf.maximum(y_mit - 1, 0)
    inside_onehot = tf.one_hot(inside_index, depth=n_products, dtype=tf.float64)
    inside_mask = tf.cast(y_mit > 0, tf.float64)[..., None]
    return inside_onehot * inside_mask


@tf.function(jit_compile=True, reduce_retracing=True)
def habit_stock_pre_choice(x_mntj: tf.Tensor, decay: tf.Tensor) -> tf.Tensor:
    """Build the pre-choice habit stock.

    The state follows

        H_{t+1} = decay * H_t + x_t

    and the returned tensor contains the pre-choice stock H_t at each period t.
    """
    x_t = tf.transpose(x_mntj, perm=[2, 0, 1, 3])
    h0 = tf.zeros(
        (tf.shape(x_mntj)[0], tf.shape(x_mntj)[1], tf.shape(x_mntj)[3]),
        dtype=tf.float64,
    )

    def step(h_curr: tf.Tensor, x_curr: tf.Tensor) -> tf.Tensor:
        return decay * h_curr + x_curr

    h_next_t = tf.scan(step, x_t, initializer=h0)
    h_curr_t = tf.concat([h0[None, ...], h_next_t[:-1, ...]], axis=0)
    return tf.transpose(h_curr_t, perm=[1, 2, 0, 3])


@tf.function(jit_compile=True, reduce_retracing=True)
def rolling_lookback_sum(x_mntj: tf.Tensor, lookback: tf.Tensor) -> tf.Tensor:
    """Sum over the previous lookback periods, excluding the current period."""
    num_periods = tf.shape(x_mntj)[2]

    prefix = tf.cumsum(x_mntj, axis=2)
    prefix = tf.concat([tf.zeros_like(prefix[:, :, :1, :]), prefix], axis=2)

    end_idx = tf.range(num_periods, dtype=tf.int32)
    start_idx = tf.maximum(0, end_idx - lookback)

    prefix_end = tf.gather(prefix, end_idx, axis=2)
    prefix_start = tf.gather(prefix, start_idx, axis=2)
    return prefix_end - prefix_start


def peer_exposure_from_onehot(
    x_mntj: tf.Tensor,
    peer_adj_m: tuple[tf.SparseTensor, ...],
    lookback: tf.Tensor,
) -> tf.Tensor:
    """Build peer exposure counts from inside-choice indicators."""
    counts_mntj = rolling_lookback_sum(x_mntj=x_mntj, lookback=lookback)

    num_consumers = tf.shape(counts_mntj)[1]
    num_periods = tf.shape(counts_mntj)[2]
    n_products = tf.shape(counts_mntj)[3]

    exposed_by_market: list[tf.Tensor] = []
    for m in range(len(peer_adj_m)):
        counts_flat = tf.reshape(counts_mntj[m], (num_consumers, -1))
        exposed_flat = tf.sparse.sparse_dense_matmul(peer_adj_m[m], counts_flat)
        exposed_by_market.append(
            tf.reshape(exposed_flat, (num_consumers, num_periods, n_products))
        )

    return tf.stack(exposed_by_market, axis=0)


def build_deterministic_states(
    y_mit: tf.Tensor,
    n_products: tf.Tensor,
    peer_adj_m: tuple[tf.SparseTensor, ...],
    lookback: tf.Tensor,
    decay: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Build deterministic state tensors from observed choices."""
    x_mntj = _inside_choice_onehot(y_mit=y_mit, n_products=n_products)
    h_mntj = habit_stock_pre_choice(x_mntj=x_mntj, decay=decay)
    p_mntj = peer_exposure_from_onehot(
        x_mntj=x_mntj,
        peer_adj_m=peer_adj_m,
        lookback=lookback,
    )
    return h_mntj, p_mntj


@tf.function(jit_compile=True, reduce_retracing=True)
def market_fourier_seasonality(
    a_mk: tf.Tensor,
    b_mk: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
) -> tf.Tensor:
    """Compute the market seasonal component from Fourier coefficients."""
    return tf.linalg.matmul(a_mk, season_sin_kt) + tf.linalg.matmul(b_mk, season_cos_kt)


@tf.function(jit_compile=True, reduce_retracing=True)
def utilities_mntj_from_theta(
    theta: dict[str, tf.Tensor],
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute inside-option utilities."""
    season_mt = market_fourier_seasonality(
        a_mk=theta["a_m"],
        b_mk=theta["b_m"],
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
    )

    weekend_jt = tf.gather(theta["beta_weekend_jw"], is_weekend_t, axis=1)
    weekend_tj = tf.transpose(weekend_jt, perm=[1, 0])

    base_mj = delta_mj + theta["beta_intercept_j"][None, :]
    base_mtj = base_mj[:, None, :] + season_mt[:, :, None] + weekend_tj[None, :, :]

    return (
        base_mtj[:, None, :, :]
        + theta["beta_habit_j"][None, None, None, :] * h_mntj
        + theta["beta_peer_j"][None, None, None, :] * p_mntj
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def loglik_mnt_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute per-observation log-likelihood contributions."""
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
    logp = tf.nn.log_softmax(logits, axis=3)
    return tf.gather(logp, y_mit, axis=3, batch_dims=3)


@tf.function(jit_compile=True, reduce_retracing=True)
def loglik_from_theta(
    theta: dict[str, tf.Tensor],
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute the total log-likelihood."""
    return tf.reduce_sum(
        loglik_mnt_from_theta(
            theta=theta,
            y_mit=y_mit,
            delta_mj=delta_mj,
            is_weekend_t=is_weekend_t,
            season_sin_kt=season_sin_kt,
            season_cos_kt=season_cos_kt,
            h_mntj=h_mntj,
            p_mntj=p_mntj,
        )
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def predict_choice_probs_from_theta(
    theta: dict[str, tf.Tensor],
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    h_mntj: tf.Tensor,
    p_mntj: tf.Tensor,
) -> tf.Tensor:
    """Compute choice probabilities with the outside option included."""
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
