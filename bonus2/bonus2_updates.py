"""
bonus2_updates.py

Random-walk Metropolis-Hastings updates for Bonus Q2 (updated spec).

State: a dict `z` of unconstrained parameters (float64 tensors).

Updated z blocks:
  z_beta_market_j  (J,)
  z_beta_habit_j   (J,)
  z_beta_peer_j    (J,)
  z_beta_dow_j     (J,2)     product weekday/weekend effects
  z_a_m            (M,K)     market seasonality coefficients (sin)
  z_b_m            (M,K)     market seasonality coefficients (cos)

All blocks use a Normal(0, sigma_z[z_key]^2) prior (constants dropped).
Therefore all update blocks share the same RW-MH structure with sigma_z.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from bonus2.bonus2_posterior import (
    logpost_z_a_m_all,
    logpost_z_b_m_all,
    logpost_z_beta_dow_j_all,
    logpost_z_beta_habit_j_all,
    logpost_z_beta_market_j_all,
    logpost_z_beta_peer_j_all,
)

LogPostFn = Callable[
    [dict[str, tf.Tensor], dict[str, tf.Tensor], dict[str, tf.Tensor]], tf.Tensor
]


def _rw_mh_update_block_with_sigma(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    step_size: tf.Tensor,
    rng: tf.random.Generator,
    z_key: str,
    logpost_fn: LogPostFn,
):
    """RW-MH update for a single block z[z_key] with Normal prior scale sigma_z[z_key]."""
    step_size = tf.cast(step_size, tf.float64)

    current = z[z_key]
    eps = rng.normal(shape=tf.shape(current), dtype=tf.float64) * step_size
    proposal = current + eps

    z_prop = dict(z)
    z_prop[z_key] = proposal

    lp_curr = logpost_fn(z=z, inputs=inputs, sigma_z=sigma_z)
    lp_prop = logpost_fn(z=z_prop, inputs=inputs, sigma_z=sigma_z)

    log_u = tf.math.log(rng.uniform(shape=(), dtype=tf.float64))
    accept = log_u < (lp_prop - lp_curr)

    z_next = tf.cond(accept, lambda: proposal, lambda: current)
    z[z_key] = z_next

    return z, accept


def update_z_beta_market_j(z, inputs, sigma_z, step_size, rng):
    """Update product intercepts z_beta_market_j (J,)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_beta_market_j",
        logpost_fn=logpost_z_beta_market_j_all,
    )


def update_z_beta_habit_j(z, inputs, sigma_z, step_size, rng):
    """Update habit sensitivities z_beta_habit_j (J,)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_beta_habit_j",
        logpost_fn=logpost_z_beta_habit_j_all,
    )


def update_z_beta_peer_j(z, inputs, sigma_z, step_size, rng):
    """Update peer sensitivities z_beta_peer_j (J,)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_beta_peer_j",
        logpost_fn=logpost_z_beta_peer_j_all,
    )


def update_z_beta_dow_j(z, inputs, sigma_z, step_size, rng):
    """Update product weekday/weekend effects z_beta_dow_j (J,2)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_beta_dow_j",
        logpost_fn=logpost_z_beta_dow_j_all,
    )


def update_z_a_m(z, inputs, sigma_z, step_size, rng):
    """Update market seasonality sine coefficients z_a_m (M,K)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_a_m",
        logpost_fn=logpost_z_a_m_all,
    )


def update_z_b_m(z, inputs, sigma_z, step_size, rng):
    """Update market seasonality cosine coefficients z_b_m (M,K)."""
    return _rw_mh_update_block_with_sigma(
        z=z,
        inputs=inputs,
        sigma_z=sigma_z,
        step_size=step_size,
        rng=rng,
        z_key="z_b_m",
        logpost_fn=logpost_z_b_m_all,
    )
