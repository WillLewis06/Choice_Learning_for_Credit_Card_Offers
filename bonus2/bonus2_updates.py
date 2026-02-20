"""
bonus2_updates.py

Random-walk Metropolis-Hastings updates for Bonus Q2 unconstrained parameter blocks z.

This module implements a symmetric Gaussian random-walk proposal:
  z' = z + step_size[z_key] * eps,   eps ~ Normal(0, I)

Acceptance uses the block log-target:
  log_target_block = loglik(z) + logprior(z_block)
where priors for other blocks are omitted because they cancel in MH ratios.

No input validation is performed here. Configs and external inputs are expected to be validated
in bonus2_input_validation.py before tensors reach this module.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from bonus2.bonus2_posterior import PosteriorInputs, log_target_block_normal_prior


def _rw_mh_update_block(
    z: dict[str, tf.Tensor],
    z_key: str,
    inputs: PosteriorInputs,
    step_size: tf.Tensor,
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Propose and accept/reject a RW-MH update for a single z block.

    Args:
      z: dictionary of unconstrained parameter blocks (float64 tensors).
      z_key: key of the block to update.
      inputs: likelihood inputs.
      step_size: scalar float64 step size for this block.
      sigma_z: prior scale dictionary keyed by z keys; sigma_z[z_key] broadcastable to z[z_key].
      rng: tf.random.Generator used for proposals and accept/reject.

    Returns:
      z_block_next: updated block tensor (same shape as z[z_key]).
      accepted: scalar float64 in {0,1}.
    """
    z_curr = z[z_key]

    # Symmetric Gaussian random-walk proposal.
    eps = rng.normal(shape=tf.shape(z_curr), dtype=z_curr.dtype)
    z_prop = z_curr + step_size * eps

    # Evaluate block log-targets at current and proposed states.
    logt_curr = log_target_block_normal_prior(
        z=z, z_key=z_key, inputs=inputs, sigma_z=sigma_z
    )

    z_prop_dict = dict(z)
    z_prop_dict[z_key] = z_prop
    logt_prop = log_target_block_normal_prior(
        z=z_prop_dict, z_key=z_key, inputs=inputs, sigma_z=sigma_z
    )

    # Accept with probability min(1, exp(logt_prop - logt_curr)).
    log_alpha = logt_prop - logt_curr

    # Guard against log(0) from RNG edge cases.
    u = tf.maximum(
        rng.uniform(shape=(), dtype=z_curr.dtype), tf.constant(1e-12, z_curr.dtype)
    )
    log_u = tf.math.log(u)

    accept = tf.cast(log_u < log_alpha, tf.int32)
    z_next = tf.where(accept > 0, z_prop, z_curr)
    return z_next, accept


def update_z_block(
    z: dict[str, tf.Tensor],
    z_key: str,
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update dispatcher for the requested z_key."""
    return _rw_mh_update_block(
        z=z,
        z_key=z_key,
        inputs=inputs,
        step_size=step_size_z[z_key],
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_beta_intercept_j(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_beta_intercept_j (J,)."""
    return update_z_block(
        z=z,
        z_key="z_beta_intercept_j",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_beta_habit_j(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_beta_habit_j (J,)."""
    return update_z_block(
        z=z,
        z_key="z_beta_habit_j",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_beta_peer_j(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_beta_peer_j (J,)."""
    return update_z_block(
        z=z,
        z_key="z_beta_peer_j",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_beta_weekend_jw(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_beta_weekend_jw (J,2)."""
    return update_z_block(
        z=z,
        z_key="z_beta_weekend_jw",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_a_m(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_a_m (M,K)."""
    return update_z_block(
        z=z,
        z_key="z_a_m",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )


def update_z_b_m(
    z: dict[str, tf.Tensor],
    inputs: PosteriorInputs,
    step_size_z: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for z_b_m (M,K)."""
    return update_z_block(
        z=z,
        z_key="z_b_m",
        inputs=inputs,
        step_size_z=step_size_z,
        sigma_z=sigma_z,
        rng=rng,
    )
