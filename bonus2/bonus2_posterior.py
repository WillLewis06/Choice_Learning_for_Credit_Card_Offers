"""
bonus2_posterior.py

Posterior utilities for Bonus Q2.

This module provides:
- Log-likelihood wrappers that delegate all model mechanics to bonus2_model.
- Elementwise Normal(0, sigma^2) priors on unconstrained parameter blocks z (dropping constants).
- Block log-targets for random-walk Metropolis updates.

No input validation is performed here. Configs and external inputs are expected to be validated
in bonus2_input_validation.py before tensors reach this module.

Required `inputs` keys:
  y_mit          (M,N,T)   int32    choices; 0=outside, c=j+1 for inside product j
  delta_mj       (M,J)     float64  Phase-1 baseline utilities (fixed)
  is_weekend_t   (T,)      int32    indicator in {0,1}
  season_sin_kt  (K,T)     float64  seasonal basis
  season_cos_kt  (K,T)     float64  seasonal basis
  peer_adj_m     tuple[M]  tf.SparseTensor (N,N) within-market adjacency, float64 values
  lookback       scalar    int32    peer lookback window length L
  decay          scalar    float64  known habit decay in (0,1)

Expected `z` keys (all float64):
  z_beta_intercept_j  (J,)
  z_beta_habit_j      (J,)
  z_beta_peer_j       (J,)
  z_beta_weekend_jw   (J,2)
  z_a_m               (M,K)
  z_b_m               (M,K)
"""

from __future__ import annotations

from typing import TypedDict

import tensorflow as tf

from bonus2 import bonus2_model as model


class PosteriorInputs(TypedDict):
    """Tensor inputs required by the Bonus Q2 likelihood."""

    y_mit: tf.Tensor
    delta_mj: tf.Tensor
    is_weekend_t: tf.Tensor
    season_sin_kt: tf.Tensor
    season_cos_kt: tf.Tensor
    peer_adj_m: model.PeerAdjacency
    lookback: tf.Tensor
    decay: tf.Tensor


Z_KEYS: tuple[str, ...] = (
    "z_beta_intercept_j",
    "z_beta_habit_j",
    "z_beta_peer_j",
    "z_beta_weekend_jw",
    "z_a_m",
    "z_b_m",
)


# =============================================================================
# Priors (up to additive constants)
# =============================================================================


def logprior_normal_sum(z_block: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Return sum of elementwise Normal(0, sigma^2) log prior (dropping constants)."""
    return tf.reduce_sum(tf.constant(-0.5, tf.float64) * tf.square(z_block / sigma))


# =============================================================================
# Likelihood via core model
# =============================================================================


def loglik_mnt(z: dict[str, tf.Tensor], inputs: PosteriorInputs) -> tf.Tensor:
    """Per-(market, consumer, time) log-likelihood contributions (M,N,T)."""
    theta = model.unconstrained_to_theta(z)
    return model.loglik_mnt_from_theta(
        theta=theta,
        y_mit=inputs["y_mit"],
        delta_mj=inputs["delta_mj"],
        is_weekend_t=inputs["is_weekend_t"],
        season_sin_kt=inputs["season_sin_kt"],
        season_cos_kt=inputs["season_cos_kt"],
        peer_adj_m=inputs["peer_adj_m"],
        lookback=inputs["lookback"],
        decay=inputs["decay"],
    )


def loglik(z: dict[str, tf.Tensor], inputs: PosteriorInputs) -> tf.Tensor:
    """Scalar total log-likelihood (sum over markets, consumers, and time)."""
    return tf.reduce_sum(loglik_mnt(z=z, inputs=inputs))


# =============================================================================
# Block log-targets for RW-MH
# =============================================================================


def log_target_block_normal_prior(
    z: dict[str, tf.Tensor],
    z_key: str,
    inputs: PosteriorInputs,
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Return scalar block log-target: log-likelihood + prior for the specified block.

    This target is suitable for Metropolis updates of a single block z[z_key].
    Priors for other blocks are omitted because they cancel in MH acceptance ratios.
    """
    ll = loglik(z=z, inputs=inputs)
    lp = logprior_normal_sum(z_block=z[z_key], sigma=sigma_z[z_key])
    return ll + lp
