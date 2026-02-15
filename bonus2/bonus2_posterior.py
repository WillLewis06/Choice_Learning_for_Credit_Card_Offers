"""
bonus2_posterior.py

Thin TensorFlow posterior wrapper for Bonus Q2 under the UPDATED spec:

  v_{m,i,j,t} =
      delta_{m,j}
    + beta_market_j[j]
    + beta_habit_j[j] * H_{m,i,j,t}
    + beta_peer_j[j]  * P_{m,i,j,t}
    + beta_dow_j[j, weekend_t[t]]
    + S_m[m,t]

where:
  H_{m,i,j,t+1} = decay * H_{m,i,j,t} + 1{ y_{m,i,t} = j }
  S_m[m,t] = sum_k a_m[m,k] * sin(k*theta_t) + b_m[m,k] * cos(k*theta_t)

All core mechanics (state construction, utilities, MNL loglik) live in:
  bonus2.bonus2_model

Required `inputs` keys:
  y_mit          (M,N,T)   int32  choices; 0=outside, j+1=inside product j
  delta_mj       (M,J)     f64    Phase-1 baseline utilities (fixed)
  weekend_t      (T,)      int32  weekday/weekend indicator in {0,1}
  season_sin_kt  (K,T)     f64    sin((k+1)*theta_t[t]) basis
  season_cos_kt  (K,T)     f64    cos((k+1)*theta_t[t]) basis
  peer_adj_m     tuple[M]  tf.SparseTensor (N,N) known within-market adjacency
  L              scalar    int32  peer lookback window length
  decay          scalar    f64    known habit decay in (0,1), passed through

Expected `z` keys (all float64):
  z_beta_market_j  (J,)
  z_beta_habit_j   (J,)
  z_beta_peer_j    (J,)
  z_beta_dow_j     (J,2)
  z_a_m            (M,K)
  z_b_m            (M,K)

Priors:
  - Elementwise Normal(0, sigma_z[z_key]^2) on each z block (dropping constants).
"""

from __future__ import annotations

import tensorflow as tf

from bonus2 import bonus2_model as model

_NEG_HALF = tf.constant(-0.5, dtype=tf.float64)


# =============================================================================
# Priors (up to additive constants)
# =============================================================================


def logprior_normal(z_block: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Elementwise Normal(0, sigma^2) log prior (dropping additive constants)."""
    return _NEG_HALF * tf.square(z_block / sigma)


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
        weekend_t=inputs["weekend_t"],
        season_sin_kt=inputs["season_sin_kt"],
        season_cos_kt=inputs["season_cos_kt"],
        peer_adj_m=inputs["peer_adj_m"],
        L=inputs["L"],
        decay=inputs["decay"],
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


def logpost_z_beta_market_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    return _logpost_normal_block(
        z=z, z_key="z_beta_market_j", inputs=inputs, sigma_z=sigma_z
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
