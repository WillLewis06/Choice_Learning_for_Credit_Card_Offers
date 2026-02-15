"""
MCMC update kernels for the stockpiling (Ching-style) model.

This module implements modular random-walk Metropolis-Hastings (RW-MH) updates for
unconstrained parameter blocks:

  - z_beta     : (1,)  scalar discount factor (mapped to beta in (0,1))
  - z_alpha    : (J,)  per-product price sensitivity (mapped to alpha > 0)
  - z_v        : (J,)  per-product stockout penalty (mapped to v > 0)
  - z_fc       : (J,)  per-product fixed purchase cost (mapped to fc > 0)
  - z_u_scale  : (M,)  per-market scale on upstream utilities (mapped to u_scale > 0)

Key conventions:
  - lambda_mn is KNOWN / FIXED data and is carried in inputs['lambda_mn'].
    It is not part of theta and is not updated here.
  - u_scale is an estimable block, but can be frozen in a test environment by
    skipping update_z_u_scale_m in the estimator/orchestration layer.

Likelihood/prior evaluation:
  - The likelihood is computed via loglik_mnj_from_theta(theta, inputs), returning
    (M,N,J) contributions.
  - The prior is applied directly on the unconstrained z-block via logprior_normal.
  - Each update returns a vector logp matching the block shape, enabling elementwise
    RW-MH accept/reject (rw_mh_step).
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf

from ching.stockpiling_model import unconstrained_to_theta
from ching.stockpiling_posterior import (
    StockpilingInputs,
    loglik_mnj_from_theta,
    logprior_normal,
)
from toolbox.mcmc_kernels import rw_mh_step


def _pack_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """
    Pack unconstrained parameters into a single z-dict.

    Returns:
      dict with keys: z_beta, z_alpha, z_v, z_fc, z_u_scale
    """
    return {
        "z_beta": tf.cast(z_beta, tf.float64),
        "z_alpha": tf.cast(z_alpha, tf.float64),
        "z_v": tf.cast(z_v, tf.float64),
        "z_fc": tf.cast(z_fc, tf.float64),
        "z_u_scale": tf.cast(z_u_scale, tf.float64),
    }


def update_z_beta_scalar(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z: Dict[str, tf.Tensor],
    k_beta: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    RW-MH update for scalar z_beta.

    Shapes:
      z_beta: (1,)

    Returns:
      z_beta_new: (1,)
      accepted: (1,) bool
    """

    def logp_beta(z_beta_prop_1: tf.Tensor) -> tf.Tensor:
        z_beta_prop = tf.reshape(tf.cast(z_beta_prop_1, tf.float64), tf.shape(z_beta))
        z_prop = _pack_z(z_beta_prop, z_alpha, z_v, z_fc, z_u_scale)
        theta = unconstrained_to_theta(z_prop)

        ll_mnj = loglik_mnj_from_theta(theta=theta, inputs=inputs)  # (M,N,J)
        ll = tf.reduce_sum(ll_mnj)
        lp = tf.reduce_sum(logprior_normal(z=z_beta_prop, sigma_z=sigma_z["z_beta"]))
        return tf.reshape(ll + lp, tf.shape(z_beta_prop))

    z_prop, accepted = rw_mh_step(z_beta, logp_beta, k_beta, rng)
    z_new = tf.where(accepted, z_prop, z_beta)
    return z_new, accepted


@tf.function(reduce_retracing=True)
def update_z_alpha_j(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z: Dict[str, tf.Tensor],
    k_alpha: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    RW-MH update for per-product z_alpha.

    Shapes:
      z_alpha: (J,)

    Returns:
      z_alpha_new: (J,)
      accepted: (J,) bool
    """

    def logp_alpha(z_alpha_prop: tf.Tensor) -> tf.Tensor:
        z_alpha_prop = tf.cast(z_alpha_prop, tf.float64)
        z_prop = _pack_z(z_beta, z_alpha_prop, z_v, z_fc, z_u_scale)
        theta = unconstrained_to_theta(z_prop)

        ll_mnj = loglik_mnj_from_theta(theta=theta, inputs=inputs)  # (M,N,J)
        ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
        lp_j = logprior_normal(z=z_alpha_prop, sigma_z=sigma_z["z_alpha"])  # (J,)
        return ll_j + lp_j

    z_prop, accepted = rw_mh_step(z_alpha, logp_alpha, k_alpha, rng)
    z_new = tf.where(accepted, z_prop, z_alpha)
    return z_new, accepted


@tf.function(reduce_retracing=True)
def update_z_v_j(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z: Dict[str, tf.Tensor],
    k_v: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    RW-MH update for per-product z_v.

    Shapes:
      z_v: (J,)

    Returns:
      z_v_new: (J,)
      accepted: (J,) bool
    """

    def logp_v(z_v_prop: tf.Tensor) -> tf.Tensor:
        z_v_prop = tf.cast(z_v_prop, tf.float64)
        z_prop = _pack_z(z_beta, z_alpha, z_v_prop, z_fc, z_u_scale)
        theta = unconstrained_to_theta(z_prop)

        ll_mnj = loglik_mnj_from_theta(theta=theta, inputs=inputs)  # (M,N,J)
        ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
        lp_j = logprior_normal(z=z_v_prop, sigma_z=sigma_z["z_v"])  # (J,)
        return ll_j + lp_j

    z_prop, accepted = rw_mh_step(z_v, logp_v, k_v, rng)
    z_new = tf.where(accepted, z_prop, z_v)
    return z_new, accepted


@tf.function(reduce_retracing=True)
def update_z_fc_j(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z: Dict[str, tf.Tensor],
    k_fc: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    RW-MH update for per-product z_fc.

    Shapes:
      z_fc: (J,)

    Returns:
      z_fc_new: (J,)
      accepted: (J,) bool
    """

    def logp_fc(z_fc_prop: tf.Tensor) -> tf.Tensor:
        z_fc_prop = tf.cast(z_fc_prop, tf.float64)
        z_prop = _pack_z(z_beta, z_alpha, z_v, z_fc_prop, z_u_scale)
        theta = unconstrained_to_theta(z_prop)

        ll_mnj = loglik_mnj_from_theta(theta=theta, inputs=inputs)  # (M,N,J)
        ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
        lp_j = logprior_normal(z=z_fc_prop, sigma_z=sigma_z["z_fc"])  # (J,)
        return ll_j + lp_j

    z_prop, accepted = rw_mh_step(z_fc, logp_fc, k_fc, rng)
    z_new = tf.where(accepted, z_prop, z_fc)
    return z_new, accepted


@tf.function(reduce_retracing=True)
def update_z_u_scale_m(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z: Dict[str, tf.Tensor],
    k_u_scale: tf.Tensor,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    RW-MH update for per-market z_u_scale.

    Shapes:
      z_u_scale: (M,)

    Returns:
      z_u_scale_new: (M,)
      accepted: (M,) bool
    """

    def logp_u_scale(z_u_scale_prop: tf.Tensor) -> tf.Tensor:
        z_u_scale_prop = tf.cast(z_u_scale_prop, tf.float64)
        z_prop = _pack_z(z_beta, z_alpha, z_v, z_fc, z_u_scale_prop)
        theta = unconstrained_to_theta(z_prop)

        ll_mnj = loglik_mnj_from_theta(theta=theta, inputs=inputs)  # (M,N,J)
        ll_m = tf.reduce_sum(ll_mnj, axis=[1, 2])  # (M,)
        lp_m = logprior_normal(z=z_u_scale_prop, sigma_z=sigma_z["z_u_scale"])  # (M,)
        return ll_m + lp_m

    z_prop, accepted = rw_mh_step(z_u_scale, logp_u_scale, k_u_scale, rng)
    z_new = tf.where(accepted, z_prop, z_u_scale)
    return z_new, accepted
