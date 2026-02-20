"""
MCMC update kernels for the stockpiling (Ching-style) model.

This module implements random-walk Metropolis-Hastings (RW-MH) updates for the
unconstrained parameter blocks:

  - z_beta     : (1,)  scalar discount factor (mapped to beta in (0,1))
  - z_alpha    : (J,)  per-product price sensitivity (mapped to alpha > 0)
  - z_v        : (J,)  per-product stockout penalty (mapped to v > 0)
  - z_fc       : (J,)  per-product fixed purchase cost (mapped to fc > 0)
  - z_u_scale  : (M,)  per-market utility scale (mapped to u_scale > 0)

Conventions:
  - All z-blocks are float64 tensors and remain float64 throughout.
  - lambda_mn is fixed input data in inputs["lambda_mn"] and is not updated here.
  - Priors are placed on the unconstrained z-blocks via logprior_normal(z, sigma_z_block).
  - loglik_mnj_from_theta(theta, inputs) returns (M,N,J) log-likelihood contributions.

Elementwise RW-MH:
  - For z_alpha, z_v, z_fc: the posterior factorizes across products conditional on the
    other blocks because each product j only affects the j-th likelihood slice.
  - For z_u_scale: the posterior factorizes across markets conditional on the other blocks
    because u_scale_m only affects market m likelihood terms.
"""

from __future__ import annotations

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
) -> dict[str, tf.Tensor]:
    """Pack unconstrained parameters into a single z-dict."""
    return {
        "z_beta": z_beta,
        "z_alpha": z_alpha,
        "z_v": z_v,
        "z_fc": z_fc,
        "z_u_scale": z_u_scale,
    }


def _theta_from_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Map unconstrained z-blocks to constrained theta dict."""
    return unconstrained_to_theta(_pack_z(z_beta, z_alpha, z_v, z_fc, z_u_scale))


def _ll_mnj_from_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
) -> tf.Tensor:
    """Compute (M,N,J) log-likelihood contributions for the theta implied by z."""
    theta = _theta_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
    return loglik_mnj_from_theta(theta=theta, inputs=inputs)


def _logp_beta_block(
    z_beta_prop: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_beta: tf.Tensor,
) -> tf.Tensor:
    """Log posterior for scalar z_beta, returned as shape (1,)."""
    ll_mnj = _ll_mnj_from_z(
        z_beta_prop, z_alpha, z_v, z_fc, z_u_scale, inputs
    )  # (M,N,J)
    ll = tf.reduce_sum(ll_mnj)  # scalar
    lp = tf.reduce_sum(logprior_normal(z=z_beta_prop, sigma_z=sigma_z_beta))  # scalar
    return tf.reshape(ll + lp, tf.shape(z_beta_prop))  # (1,)


def _logp_alpha_block(
    z_beta: tf.Tensor,
    z_alpha_prop: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_alpha: tf.Tensor,
) -> tf.Tensor:
    """Log posterior for per-product z_alpha, returned as (J,)."""
    ll_mnj = _ll_mnj_from_z(
        z_beta, z_alpha_prop, z_v, z_fc, z_u_scale, inputs
    )  # (M,N,J)
    ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
    lp_j = logprior_normal(z=z_alpha_prop, sigma_z=sigma_z_alpha)  # (J,)
    return ll_j + lp_j


def _logp_v_block(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v_prop: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_v: tf.Tensor,
) -> tf.Tensor:
    """Log posterior for per-product z_v, returned as (J,)."""
    ll_mnj = _ll_mnj_from_z(
        z_beta, z_alpha, z_v_prop, z_fc, z_u_scale, inputs
    )  # (M,N,J)
    ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
    lp_j = logprior_normal(z=z_v_prop, sigma_z=sigma_z_v)  # (J,)
    return ll_j + lp_j


def _logp_fc_block(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc_prop: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_fc: tf.Tensor,
) -> tf.Tensor:
    """Log posterior for per-product z_fc, returned as (J,)."""
    ll_mnj = _ll_mnj_from_z(
        z_beta, z_alpha, z_v, z_fc_prop, z_u_scale, inputs
    )  # (M,N,J)
    ll_j = tf.reduce_sum(ll_mnj, axis=[0, 1])  # (J,)
    lp_j = logprior_normal(z=z_fc_prop, sigma_z=sigma_z_fc)  # (J,)
    return ll_j + lp_j


def _logp_u_scale_block(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale_prop: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_u_scale: tf.Tensor,
) -> tf.Tensor:
    """Log posterior for per-market z_u_scale, returned as (M,)."""
    ll_mnj = _ll_mnj_from_z(
        z_beta, z_alpha, z_v, z_fc, z_u_scale_prop, inputs
    )  # (M,N,J)
    ll_m = tf.reduce_sum(ll_mnj, axis=[1, 2])  # (M,)
    lp_m = logprior_normal(z=z_u_scale_prop, sigma_z=sigma_z_u_scale)  # (M,)
    return ll_m + lp_m


@tf.function(reduce_retracing=True)
def update_z_beta_scalar(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    inputs: StockpilingInputs,
    sigma_z_beta: tf.Tensor,
    k_beta: tf.Tensor,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for scalar z_beta (shape (1,))."""

    def logp_beta(z_beta_prop: tf.Tensor) -> tf.Tensor:
        return _logp_beta_block(
            z_beta_prop=z_beta_prop,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            inputs=inputs,
            sigma_z_beta=sigma_z_beta,
        )

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
    sigma_z_alpha: tf.Tensor,
    k_alpha: tf.Tensor,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for per-product z_alpha (shape (J,))."""

    def logp_alpha(z_alpha_prop: tf.Tensor) -> tf.Tensor:
        return _logp_alpha_block(
            z_beta=z_beta,
            z_alpha_prop=z_alpha_prop,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            inputs=inputs,
            sigma_z_alpha=sigma_z_alpha,
        )

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
    sigma_z_v: tf.Tensor,
    k_v: tf.Tensor,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for per-product z_v (shape (J,))."""

    def logp_v(z_v_prop: tf.Tensor) -> tf.Tensor:
        return _logp_v_block(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v_prop=z_v_prop,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            inputs=inputs,
            sigma_z_v=sigma_z_v,
        )

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
    sigma_z_fc: tf.Tensor,
    k_fc: tf.Tensor,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for per-product z_fc (shape (J,))."""

    def logp_fc(z_fc_prop: tf.Tensor) -> tf.Tensor:
        return _logp_fc_block(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc_prop=z_fc_prop,
            z_u_scale=z_u_scale,
            inputs=inputs,
            sigma_z_fc=sigma_z_fc,
        )

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
    sigma_z_u_scale: tf.Tensor,
    k_u_scale: tf.Tensor,
    rng: tf.random.Generator,
) -> tuple[tf.Tensor, tf.Tensor]:
    """RW-MH update for per-market z_u_scale (shape (M,))."""

    def logp_u_scale(z_u_scale_prop: tf.Tensor) -> tf.Tensor:
        return _logp_u_scale_block(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale_prop=z_u_scale_prop,
            inputs=inputs,
            sigma_z_u_scale=sigma_z_u_scale,
        )

    z_prop, accepted = rw_mh_step(z_u_scale, logp_u_scale, k_u_scale, rng)
    z_new = tf.where(accepted, z_prop, z_u_scale)
    return z_new, accepted
