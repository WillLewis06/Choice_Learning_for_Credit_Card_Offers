"""
bonus2/bonus2_updates.py

RW–MH updates for the Bonus Q2 model.

Design goal: keep the updates simple and legible, and avoid proposing large
high-dimensional blocks in a single MH decision.

Key change vs the previous version:
  - Market-leading and product-leading "time" blocks are updated row-wise:
      * market blocks: update one market row at a time (index m)
      * product blocks: update one product row at a time (index j)
      * market×product intercept: update one market row at a time (index m)

This file intentionally does not preserve backward compatibility with the
previous block-wise update signatures.

Conventions:
  - All tensors are tf.float64.
  - `k` is a scalar tf.float64 proposal scale for the update being performed.
  - `z` is a dict[str, tf.Tensor] holding *unconstrained* sampler state blocks:
      z_beta_market_mj : (M, J)
      z_beta_habit_j   : (J,)
      z_beta_peer_j    : (J,)
      z_decay_rate_j   : (J,)
      z_beta_dow_m     : (M, 7)
      z_beta_dow_j     : (J, 7)
      z_a_m, z_b_m     : (M, K)
      z_a_j, z_b_j     : (J, K)

Returned values:
  - Each update returns (new_block_tensor, accepted), where accepted is a scalar
    boolean (the MH decision for that proposal).
"""

from __future__ import annotations

import tensorflow as tf

from toolbox.mcmc_kernels import rw_mh_step

from bonus2.bonus2_posterior import (
    logpost_z_a_j_all,
    logpost_z_a_m_all,
    logpost_z_b_j_all,
    logpost_z_b_m_all,
    logpost_z_beta_dow_j_all,
    logpost_z_beta_dow_m_all,
    logpost_z_beta_habit_j_all,
    logpost_z_beta_market_mj_all,
    logpost_z_beta_peer_j_all,
    logpost_z_decay_rate_j_all,
)


def _with_block(
    z: dict[str, tf.Tensor], key: str, value: tf.Tensor
) -> dict[str, tf.Tensor]:
    """Return a shallow copy of z with one block replaced."""
    z2 = dict(z)
    z2[key] = value
    return z2


def _scatter_row_2d(
    block_2d: tf.Tensor, row_index: tf.Tensor, row_value: tf.Tensor
) -> tf.Tensor:
    """
    Replace block_2d[row_index, :] with row_value and return the new 2D tensor.

    Args:
      block_2d: (R, C)
      row_index: scalar int tensor
      row_value: (C,)
    """
    row_index = tf.cast(row_index, tf.int32)
    idx = tf.reshape(row_index, [1, 1])  # (1,1)
    upd = tf.expand_dims(row_value, axis=0)  # (1,C)
    return tf.tensor_scatter_nd_update(block_2d, idx, upd)


def _rw_mh_update_block(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    z_key: str,
    logpost_fn,
):
    """
    RW–MH update for an entire block z[z_key] as one proposal.

    Returns:
      (block_new, accepted) where accepted is a scalar bool.
    """
    x0 = z[z_key]

    def logp(x_t: tf.Tensor) -> tf.Tensor:
        z_t = _with_block(z, z_key, x_t)
        return logpost_fn(z=z_t, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(x0, logp, k, rng)


def _rw_mh_update_2d_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    z_key: str,
    logpost_fn,
    row_index: tf.Tensor,
):
    """
    RW–MH update for a single row of a 2D block z[z_key].

    Returns:
      (block_new, accepted) where accepted is a scalar bool.
    """
    block0 = z[z_key]
    row0 = block0[row_index]

    def logp(row_t: tf.Tensor) -> tf.Tensor:
        block_t = _scatter_row_2d(block0, row_index, row_t)
        z_t = _with_block(z, z_key, block_t)
        return logpost_fn(z=z_t, inputs=inputs, sigma_z=sigma_z)

    row_new, accepted = rw_mh_step(row0, logp, k, rng)
    block_new = _scatter_row_2d(block0, row_index, row_new)
    return block_new, accepted


# =============================================================================
# Vector blocks (small; update as one block for simplicity)
# =============================================================================


@tf.function(reduce_retracing=True)
def update_z_beta_habit_j(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_habit_j (J,) as a single RW–MH proposal."""
    return _rw_mh_update_block(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_habit_j",
        logpost_fn=logpost_z_beta_habit_j_all,
    )


@tf.function(reduce_retracing=True)
def update_z_beta_peer_j(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_peer_j (J,) as a single RW–MH proposal."""
    return _rw_mh_update_block(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_peer_j",
        logpost_fn=logpost_z_beta_peer_j_all,
    )


@tf.function(reduce_retracing=True)
def update_z_decay_rate_j(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """
    Update z_decay_rate_j (J,) as a single RW–MH proposal.

    Note: the posterior ignores sigma_z for this block (Beta prior in constrained space).
    """
    return _rw_mh_update_block(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_decay_rate_j",
        logpost_fn=logpost_z_decay_rate_j_all,
    )


# =============================================================================
# Market-leading blocks (row-wise by market m)
# =============================================================================


@tf.function(reduce_retracing=True)
def update_z_beta_market_mj_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    m: tf.Tensor,
):
    """Update one market row of z_beta_market_mj: z_beta_market_mj[m, :] (J,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_market_mj",
        logpost_fn=logpost_z_beta_market_mj_all,
        row_index=m,
    )


@tf.function(reduce_retracing=True)
def update_z_beta_dow_m_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    m: tf.Tensor,
):
    """Update one market row of z_beta_dow_m: z_beta_dow_m[m, :] (7,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_dow_m",
        logpost_fn=logpost_z_beta_dow_m_all,
        row_index=m,
    )


@tf.function(reduce_retracing=True)
def update_z_a_m_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    m: tf.Tensor,
):
    """Update one market row of z_a_m: z_a_m[m, :] (K,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_a_m",
        logpost_fn=logpost_z_a_m_all,
        row_index=m,
    )


@tf.function(reduce_retracing=True)
def update_z_b_m_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    m: tf.Tensor,
):
    """Update one market row of z_b_m: z_b_m[m, :] (K,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_b_m",
        logpost_fn=logpost_z_b_m_all,
        row_index=m,
    )


# =============================================================================
# Product-leading blocks (row-wise by product j)
# =============================================================================


@tf.function(reduce_retracing=True)
def update_z_beta_dow_j_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    j: tf.Tensor,
):
    """Update one product row of z_beta_dow_j: z_beta_dow_j[j, :] (7,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_dow_j",
        logpost_fn=logpost_z_beta_dow_j_all,
        row_index=j,
    )


@tf.function(reduce_retracing=True)
def update_z_a_j_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    j: tf.Tensor,
):
    """Update one product row of z_a_j: z_a_j[j, :] (K,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_a_j",
        logpost_fn=logpost_z_a_j_all,
        row_index=j,
    )


@tf.function(reduce_retracing=True)
def update_z_b_j_row(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    j: tf.Tensor,
):
    """Update one product row of z_b_j: z_b_j[j, :] (K,)."""
    return _rw_mh_update_2d_row(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_b_j",
        logpost_fn=logpost_z_b_j_all,
        row_index=j,
    )
