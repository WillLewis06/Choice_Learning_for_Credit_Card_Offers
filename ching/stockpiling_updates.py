"""
ching/stockpiling_updates.py

Block-wise MCMC updates for the Phase-3 (Ching-style) stockpiling estimator.

This module mirrors the Lu pattern:
  - one function per parameter block update
  - each update holds the other blocks fixed, builds the full z-dict, evaluates the
    appropriate block-shaped log-posterior view, and calls the RW-MH kernel.

All tensors are assumed to be tf.float64 and already validated upstream.
"""

from __future__ import annotations

import tensorflow as tf

from toolbox.mcmc_kernels import rw_mh_step

from ching.stockpiling_posterior import (
    logpost_z_alpha_mj,
    logpost_z_beta_mj,
    logpost_z_fc_mj,
    logpost_z_lambda_mn,
    logpost_z_u_scale_m,
    logpost_z_v_mj,
)


def _pack_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Build the full z dict expected by stockpiling_posterior views."""
    return {
        "z_beta": z_beta,
        "z_alpha": z_alpha,
        "z_v": z_v,
        "z_fc": z_fc,
        "z_lambda": z_lambda,
        "z_u_scale": z_u_scale,
    }


@tf.function(reduce_retracing=True)
def update_z_beta_mj(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_beta (shape (M,J))."""
    k = tf.cast(k, tf.float64)

    def logp(z_beta_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta_t, z_alpha, z_v, z_fc, z_lambda, z_u_scale)
        return logpost_z_beta_mj(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_alpha_mj(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_alpha (shape (M,J))."""
    k = tf.cast(k, tf.float64)

    def logp(z_alpha_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha_t, z_v, z_fc, z_lambda, z_u_scale)
        return logpost_z_alpha_mj(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_alpha, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_v_mj(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_v (shape (M,J))."""
    k = tf.cast(k, tf.float64)

    def logp(z_v_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v_t, z_fc, z_lambda, z_u_scale)
        return logpost_z_v_mj(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_v, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_fc_mj(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_fc (shape (M,J))."""
    k = tf.cast(k, tf.float64)

    def logp(z_fc_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v, z_fc_t, z_lambda, z_u_scale)
        return logpost_z_fc_mj(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_fc, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_lambda_mn(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_lambda (shape (M,N))."""
    k = tf.cast(k, tf.float64)

    def logp(z_lambda_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v, z_fc, z_lambda_t, z_u_scale)
        return logpost_z_lambda_mn(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_lambda, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_u_scale_m(
    *,
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """RW-MH update for z_u_scale (shape (M,)). Estimator-only block."""
    k = tf.cast(k, tf.float64)

    def logp(z_u_scale_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v, z_fc, z_lambda, z_u_scale_t)
        return logpost_z_u_scale_m(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_u_scale, logp, k, rng)
