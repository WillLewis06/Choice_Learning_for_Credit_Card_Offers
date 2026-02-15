"""
bonus2/bonus2_updates.py

RW–MH updates for the Bonus Q2 model.

Design goals:
  - Keep updates simple and legible.
  - Update each parameter block as a single RW–MH proposal (one MH decision per block).
  - No row-wise update machinery.

Conventions:
  - All tensors are tf.float64 unless otherwise noted.
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
  - Each update returns (new_block_tensor, accepted), where accepted is a scalar bool.
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


def _rw_mh_update_block_with_sigma(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    z_key: str,
    logpost_fn,
):
    """RW–MH update for an entire block z[z_key] (Normal-prior blocks)."""
    x0 = z[z_key]

    def logp(x_t: tf.Tensor) -> tf.Tensor:
        z_t = _with_block(z, z_key, x_t)
        return logpost_fn(z=z_t, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(x0, logp, k, rng)


def _rw_mh_update_block_no_sigma(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
    z_key: str,
    logpost_fn,
):
    """RW–MH update for an entire block z[z_key] (blocks whose posterior does not take sigma_z)."""
    x0 = z[z_key]

    def logp(x_t: tf.Tensor) -> tf.Tensor:
        z_t = _with_block(z, z_key, x_t)
        return logpost_fn(z=z_t, inputs=inputs)

    return rw_mh_step(x0, logp, k, rng)


# =============================================================================
# Vector blocks
# =============================================================================


def update_z_beta_habit_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_habit_j (J,) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_habit_j",
        logpost_fn=logpost_z_beta_habit_j_all,
    )


def update_z_beta_peer_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_peer_j (J,) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_peer_j",
        logpost_fn=logpost_z_beta_peer_j_all,
    )


def update_z_decay_rate_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_decay_rate_j (J,) as a single RW–MH proposal (Beta prior via posterior; no sigma_z)."""
    return _rw_mh_update_block_no_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        z=z,
        z_key="z_decay_rate_j",
        logpost_fn=logpost_z_decay_rate_j_all,
    )


# =============================================================================
# 2D blocks (whole-block updates only)
# =============================================================================


def update_z_beta_market_mj(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_market_mj (M,J) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_market_mj",
        logpost_fn=logpost_z_beta_market_mj_all,
    )


def update_z_beta_dow_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_dow_m (M,7) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_dow_m",
        logpost_fn=logpost_z_beta_dow_m_all,
    )


def update_z_beta_dow_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_beta_dow_j (J,7) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_beta_dow_j",
        logpost_fn=logpost_z_beta_dow_j_all,
    )


def update_z_a_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_a_m (M,K) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_a_m",
        logpost_fn=logpost_z_a_m_all,
    )


def update_z_b_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_b_m (M,K) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_b_m",
        logpost_fn=logpost_z_b_m_all,
    )


def update_z_a_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_a_j (J,K) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_a_j",
        logpost_fn=logpost_z_a_j_all,
    )


def update_z_b_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z: dict[str, tf.Tensor],
):
    """Update z_b_j (J,K) as a single RW–MH proposal."""
    return _rw_mh_update_block_with_sigma(
        rng=rng,
        k=k,
        inputs=inputs,
        sigma_z=sigma_z,
        z=z,
        z_key="z_b_j",
        logpost_fn=logpost_z_b_j_all,
    )
