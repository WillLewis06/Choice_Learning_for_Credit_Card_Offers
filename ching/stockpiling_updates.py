"""
ching/stockpiling_updates.py

RW-MH updates for the Phase-3 (Ching-style) stockpiling estimator.

This version implements a JOINT RW-MH update for (z_beta, z_alpha, z_fc) at each
(market, product) site using a single accept/reject decision per (m,j).

Other blocks remain single-block elementwise RW-MH:
  - z_v       (M,J)
  - z_lambda  (M,N)
  - z_u_scale (M,)   (estimator-only)

All tensors are assumed to be tf.float64 and already validated upstream.
"""

from __future__ import annotations

import tensorflow as tf

from toolbox.mcmc_kernels import rw_mh_step, rw_mh_step_joint

from ching.stockpiling_posterior import (
    logpost_z_beta_alpha_fc_mj,
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
    """Build the full `z` dict expected by `ching.stockpiling_posterior` views."""
    return {
        "z_beta": z_beta,
        "z_alpha": z_alpha,
        "z_v": z_v,
        "z_fc": z_fc,
        "z_lambda": z_lambda,
        "z_u_scale": z_u_scale,
    }


@tf.function(reduce_retracing=True)
def update_z_beta_alpha_fc_mj(
    rng: tf.random.Generator,
    k_beta: tf.Tensor,
    k_alpha: tf.Tensor,
    k_fc: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_lambda: tf.Tensor,
    z_u_scale: tf.Tensor,
):
    """JOINT RW-MH update for (z_beta, z_alpha, z_fc), each of shape (M,J).

    This performs a sitewise joint update over the last dimension K=3:
      theta0[m,j,:] = [z_beta[m,j], z_alpha[m,j], z_fc[m,j]]
    with one accept/reject decision per (m,j), so all three coordinates move together.

    Proposal scales:
      - diagonal scales with k = [k_beta, k_alpha, k_fc]  (shape (3,))

    Returns:
      z_beta_new:  (M,J)
      z_alpha_new: (M,J)
      z_fc_new:    (M,J)
      accepted:    (M,J) boolean (sitewise)
    """
    k_beta = tf.convert_to_tensor(k_beta, dtype=tf.float64)
    k_alpha = tf.convert_to_tensor(k_alpha, dtype=tf.float64)
    k_fc = tf.convert_to_tensor(k_fc, dtype=tf.float64)

    theta0 = tf.stack([z_beta, z_alpha, z_fc], axis=-1)  # (M,J,3)
    k_vec = tf.stack([k_beta, k_alpha, k_fc], axis=0)  # (3,)

    def logp(theta_t: tf.Tensor) -> tf.Tensor:
        z_beta_t = theta_t[..., 0]
        z_alpha_t = theta_t[..., 1]
        z_fc_t = theta_t[..., 2]
        z = _pack_z(z_beta_t, z_alpha_t, z_v, z_fc_t, z_lambda, z_u_scale)
        return logpost_z_beta_alpha_fc_mj(z=z, inputs=inputs, sigma_z=sigma_z)  # (M,J)

    theta_new, accepted = rw_mh_step_joint(theta0, logp, k_vec, rng)  # (M,J,3), (M,J)

    z_beta_new = theta_new[..., 0]
    z_alpha_new = theta_new[..., 1]
    z_fc_new = theta_new[..., 2]

    return z_beta_new, z_alpha_new, z_fc_new, accepted


@tf.function(reduce_retracing=True)
def update_z_v_mj(
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
    """RW-MH update for the `z_v` block (shape (M, J))."""

    def logp(z_v_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v_t, z_fc, z_lambda, z_u_scale)
        return logpost_z_v_mj(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_v, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_lambda_mn(
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
    """RW-MH update for the `z_lambda` block (shape (M, N))."""

    def logp(z_lambda_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v, z_fc, z_lambda_t, z_u_scale)
        return logpost_z_lambda_mn(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_lambda, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_u_scale_m(
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
    """RW-MH update for the estimator-only `z_u_scale` block (shape (M,))."""

    def logp(z_u_scale_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(z_beta, z_alpha, z_v, z_fc, z_lambda, z_u_scale_t)
        return logpost_z_u_scale_m(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_u_scale, logp, k, rng)


# -----------------------------------------------------------------------------
# Legacy single-block updates (commented out for easy revert)
# -----------------------------------------------------------------------------
#
# from ching.stockpiling_posterior import (
#     logpost_z_alpha_mj,
#     logpost_z_beta_mj,
#     logpost_z_fc_mj,
# )
#
#
# @tf.function(reduce_retracing=True)
# def update_z_beta_mj(
#     rng: tf.random.Generator,
#     k: tf.Tensor,
#     inputs: dict[str, tf.Tensor],
#     sigma_z: dict[str, tf.Tensor],
#     z_beta: tf.Tensor,
#     z_alpha: tf.Tensor,
#     z_v: tf.Tensor,
#     z_fc: tf.Tensor,
#     z_lambda: tf.Tensor,
#     z_u_scale: tf.Tensor,
# ):
#     def logp(z_beta_t: tf.Tensor) -> tf.Tensor:
#         z = _pack_z(z_beta_t, z_alpha, z_v, z_fc, z_lambda, z_u_scale)
#         return logpost_z_beta_mj(z=z, inputs=inputs, sigma_z=sigma_z)
#     return rw_mh_step(z_beta, logp, k, rng)
#
#
# @tf.function(reduce_retracing=True)
# def update_z_alpha_mj(
#     rng: tf.random.Generator,
#     k: tf.Tensor,
#     inputs: dict[str, tf.Tensor],
#     sigma_z: dict[str, tf.Tensor],
#     z_beta: tf.Tensor,
#     z_alpha: tf.Tensor,
#     z_v: tf.Tensor,
#     z_fc: tf.Tensor,
#     z_lambda: tf.Tensor,
#     z_u_scale: tf.Tensor,
# ):
#     def logp(z_alpha_t: tf.Tensor) -> tf.Tensor:
#         z = _pack_z(z_beta, z_alpha_t, z_v, z_fc, z_lambda, z_u_scale)
#         return logpost_z_alpha_mj(z=z, inputs=inputs, sigma_z=sigma_z)
#     return rw_mh_step(z_alpha, logp, k, rng)
#
#
# @tf.function(reduce_retracing=True)
# def update_z_fc_mj(
#     rng: tf.random.Generator,
#     k: tf.Tensor,
#     inputs: dict[str, tf.Tensor],
#     sigma_z: dict[str, tf.Tensor],
#     z_beta: tf.Tensor,
#     z_alpha: tf.Tensor,
#     z_v: tf.Tensor,
#     z_fc: tf.Tensor,
#     z_lambda: tf.Tensor,
#     z_u_scale: tf.Tensor,
# ):
#     def logp(z_fc_t: tf.Tensor) -> tf.Tensor:
#         z = _pack_z(z_beta, z_alpha, z_v, z_fc_t, z_lambda, z_u_scale)
#         return logpost_z_fc_mj(z=z, inputs=inputs, sigma_z=sigma_z)
#     return rw_mh_step(z_fc, logp, k, rng)
