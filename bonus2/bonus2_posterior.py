"""
bonus2_posterior.py

Thin TensorFlow posterior wrapper for Bonus Q2 (habit + peer + DOW + seasonality) MNL model.

This module contains:
  - Priors on unconstrained sampler variables z_*
      * Normal priors for most blocks (elementwise, up to additive constants)
      * Beta(kappa_decay, 1) prior for decay_rate_j in constrained space with
        change-of-variables when sampling in z-space via decay_rate_j = sigmoid(z_decay_rate_j)
  - Likelihood wrapper loglik_mnt(z, inputs) that calls bonus2_model
  - Scalar log-posterior functions per parameter block for RW-MH updates

All core mechanics (parameter transforms, habit recursion, peer exposure, utilities, MNL) live in:
  bonus2.bonus2_model

Required `inputs` keys:
  y_mit          (M,N,T)   int32  choices; 0=outside, j+1=inside product j
  delta_mj       (M,J)     f64    Phase-1 baseline utilities (fixed)
  dow_t          (T,)      int32  weekday index in {0..6}
  season_sin_kt  (K,T)     f64    sin((k+1)*season_angle_t[t])
  season_cos_kt  (K,T)     f64    cos((k+1)*season_angle_t[t])
  peer_adj_m     tuple[M]  tf.SparseTensor (N,N) known within-market adjacency
  L              scalar    int32  peer lookback window length
  decay_rate_eps scalar    f64    numeric guard for decay_rate clipping (can be 0.0)
  kappa_decay    scalar    f64    Beta prior shape for decay_rate_j ~ Beta(kappa_decay, 1)

Expected `z` keys:
  z_beta_market_mj  (M,J)
  z_beta_habit_j    (J,)
  z_beta_peer_j     (J,)
  z_decay_rate_j    (J,)
  z_beta_dow_m      (M,7)
  z_beta_dow_j      (J,7)
  z_a_m             (M,K)
  z_b_m             (M,K)
  z_a_j             (J,K)
  z_b_j             (J,K)

All z blocks and prior scales sigma_z[*] are expected to be float64 tensors.
"""

from __future__ import annotations

import tensorflow as tf

from bonus2 import bonus2_model as model

_EPS_SAFE = tf.constant(1e-12, dtype=tf.float64)
_NEG_HALF = tf.constant(-0.5, dtype=tf.float64)


# =============================================================================
# Priors (up to additive constants)
# =============================================================================


def logprior_normal(z_block: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Elementwise Normal(0, sigma^2) log prior (dropping additive constants)."""
    return _NEG_HALF * tf.square(z_block / sigma)


def logprior_decay_beta_kappa1_on_z(
    z_decay_rate_j: tf.Tensor,
    kappa_decay: tf.Tensor,
    decay_rate_eps: tf.Tensor,
) -> tf.Tensor:
    """
    Log prior for z_decay_rate_j when decay_rate_j ~ Beta(kappa_decay, 1) in constrained space,
    and decay_rate_j = sigmoid(z_decay_rate_j) in the sampler.

    Up to constants:
      log p(z) = kappa_decay * log(decay) + log(1 - decay),
    where decay = sigmoid(z).
    """
    eps = tf.maximum(decay_rate_eps, _EPS_SAFE)
    decay = tf.math.sigmoid(z_decay_rate_j)
    decay = tf.clip_by_value(decay, eps, 1.0 - eps)
    return kappa_decay * tf.math.log(decay) + tf.math.log1p(-decay)


# =============================================================================
# Likelihood via core model
# =============================================================================


def loglik_mnt(z: dict[str, tf.Tensor], inputs: dict[str, tf.Tensor]) -> tf.Tensor:
    """Per-(market, consumer, time) log-likelihood contributions (M,N,T)."""
    theta = model.unconstrained_to_theta(z)
    return model.loglik_mnt_from_theta(
        theta=theta,
        y_mit=inputs["y_mit"],
        delta_mj=inputs["delta_mj"],
        dow_t=inputs["dow_t"],
        season_sin_kt=inputs["season_sin_kt"],
        season_cos_kt=inputs["season_cos_kt"],
        peer_adj_m=inputs["peer_adj_m"],
        L=inputs["L"],
        decay_rate_eps=inputs["decay_rate_eps"],
    )


def loglik(z: dict[str, tf.Tensor], inputs: dict[str, tf.Tensor]) -> tf.Tensor:
    """Scalar total log-likelihood (sum over M,N,T)."""
    return tf.reduce_sum(loglik_mnt(z=z, inputs=inputs))


# =============================================================================
# Scalar log-posterior per block (for RW-MH)
# =============================================================================


def _logpost_normal_block(
    z: dict[str, tf.Tensor],
    z_key: str,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior for a block with elementwise Normal prior."""
    ll = loglik(z=z, inputs=inputs)
    lp = tf.reduce_sum(logprior_normal(z[z_key], sigma_z[z_key]))
    return ll + lp


def logpost_z_beta_market_mj_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_market_mj", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_habit_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_habit_j", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_peer_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_peer_j", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_dow_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_dow_m", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_dow_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_dow_j", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_a_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(z=z, z_key="z_a_m", inputs=inputs, sigma_z=sigma_z)


def logpost_z_b_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(z=z, z_key="z_b_m", inputs=inputs, sigma_z=sigma_z)


def logpost_z_a_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(z=z, z_key="z_a_j", inputs=inputs, sigma_z=sigma_z)


def logpost_z_b_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(z=z, z_key="z_b_j", inputs=inputs, sigma_z=sigma_z)


def logpost_z_decay_rate_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior for z_decay_rate_j with Beta(kappa_decay, 1) prior in constrained space."""
    ll = loglik(z=z, inputs=inputs)
    lp = tf.reduce_sum(
        logprior_decay_beta_kappa1_on_z(
            z_decay_rate_j=z["z_decay_rate_j"],
            kappa_decay=inputs["kappa_decay"],
            decay_rate_eps=inputs["decay_rate_eps"],
        )
    )
    return ll + lp
