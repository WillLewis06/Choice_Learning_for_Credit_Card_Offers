"""
Thin TensorFlow posterior wrapper for the multi-product stockpiling model.

This module contains ONLY:
  - Normal priors on unconstrained sampler variables z_*
  - Likelihood wrapper loglik_mnj(z, inputs) that calls ching.stockpiling_model
  - Block-shaped log-posterior "views" used by elementwise RW-MH updates

All core mechanics (DP, CCPs, latent-inventory forward filter, parameter transforms) live in:
  ching.stockpiling_model

Target Phase-3 shapes:

Observed inputs (in `inputs` dict):
  a_mnjt        (M,N,J,T)
  p_state_mjt   (M,J,T)
  u_mj          (M,J)
  price_vals_mj (M,J,S)
  P_price_mj    (M,J,S,S)
  pi_I0         (I,)
  waste_cost    scalar
  eps           scalar
  tol           scalar
  max_iter      scalar (int32)
  maps          model.InventoryMaps

Unconstrained sampler variables z:
  z_beta, z_alpha, z_v, z_fc : (M,J)
  z_lambda                   : (M,N)
  z_u_scale                  : (M,)

Likelihood contributions:
  loglik_mnj : (M,N,J)

Block-shaped log-posterior views (for elementwise RW-MH):
  - market-product blocks: (M,J)  for z_beta, z_alpha, z_v, z_fc
  - market-consumer block: (M,N)  for z_lambda
  - market block:          (M,)   for z_u_scale
"""

from __future__ import annotations

import tensorflow as tf

from ching import stockpiling_model as model


# =============================================================================
# Priors (up to additive constants)
# =============================================================================


def logprior_normal(z_block: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """
    Elementwise Normal(0, sigma^2) log prior (dropping additive constants).

    Args:
      z_block: arbitrary shape
      sigma: scalar or broadcastable to z_block

    Returns:
      Tensor with same shape as z_block.
    """
    z_block = tf.cast(z_block, tf.float64)
    sigma = tf.cast(sigma, tf.float64)
    return -0.5 * tf.square(z_block / sigma)


# =============================================================================
# Likelihood via core model
# =============================================================================


def _cast_z_to_f64(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Ensure all z_* blocks are float64."""
    return {k: tf.cast(v, tf.float64) for k, v in z.items()}


def loglik_mnj(z: dict[str, tf.Tensor], inputs: dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Per-(market, consumer, product) log-likelihood contributions (M,N,J).

    Computes theta = unconstrained_to_theta(z), solves CCPs via DP under theta,
    then runs the forward filter integrating out latent inventory.

    Returns:
      loglik_mnj: (M,N,J) float64
    """
    z64 = _cast_z_to_f64(z)
    theta = model.unconstrained_to_theta(z64)

    return model.loglik_mnj_from_theta(
        theta=theta,
        a_mnjt=inputs["a_mnjt"],
        p_state_mjt=inputs["p_state_mjt"],
        u_mj=inputs["u_mj"],
        price_vals_mj=inputs["price_vals_mj"],
        P_price_mj=inputs["P_price_mj"],
        pi_I0=inputs["pi_I0"],
        waste_cost=inputs["waste_cost"],
        eps=inputs["eps"],
        tol=inputs["tol"],
        max_iter=inputs["max_iter"],
        maps=inputs["maps"],
    )


# =============================================================================
# Block-shaped log-posterior "views" for RW-MH
# =============================================================================


def _reduce_ll(ll_mnj: tf.Tensor, view: str) -> tf.Tensor:
    """
    Reduce loglik_mnj (M,N,J) to a block view.

    view:
      "mj": sum over N -> (M,J)
      "mn": sum over J -> (M,N)
      "m":  sum over N,J -> (M,)
    """
    if view == "mj":
        return tf.reduce_sum(ll_mnj, axis=1)
    if view == "mn":
        return tf.reduce_sum(ll_mnj, axis=2)
    if view == "m":
        return tf.reduce_sum(ll_mnj, axis=[1, 2])
    raise ValueError(f"Unknown view='{view}' (expected one of: 'mj','mn','m').")


def _logpost_view(
    z: dict[str, tf.Tensor],
    z_key: str,
    view: str,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Generic log-posterior view for elementwise RW-MH.

    Note: This recomputes the full DP+filter likelihood for each call; this is
    intended for elementwise RW-MH kernels that update one z-block at a time.

    Returns:
      Tensor shaped according to `view`.
    """
    ll_mnj = loglik_mnj(z=z, inputs=inputs)
    ll_view = _reduce_ll(ll_mnj, view=view)
    lp_view = logprior_normal(z[z_key], sigma_z[z_key])
    return ll_view + lp_view


def logpost_z_beta_mj(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,J) log posterior view for updating z_beta."""
    return _logpost_view(z=z, z_key="z_beta", view="mj", inputs=inputs, sigma_z=sigma_z)


def logpost_z_alpha_mj(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,J) log posterior view for updating z_alpha."""
    return _logpost_view(
        z=z, z_key="z_alpha", view="mj", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_v_mj(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,J) log posterior view for updating z_v."""
    return _logpost_view(z=z, z_key="z_v", view="mj", inputs=inputs, sigma_z=sigma_z)


def logpost_z_fc_mj(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,J) log posterior view for updating z_fc."""
    return _logpost_view(z=z, z_key="z_fc", view="mj", inputs=inputs, sigma_z=sigma_z)


def logpost_z_lambda_mn(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_lambda."""
    return _logpost_view(
        z=z, z_key="z_lambda", view="mn", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_u_scale_m(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,) log posterior view for updating z_u_scale."""
    return _logpost_view(
        z=z, z_key="z_u_scale", view="m", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_alpha_fc_mj(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    (M,J) joint log posterior view for jointly updating (z_beta, z_alpha, z_fc).

    This view:
      - computes the likelihood ONCE (DP + forward filter) and reduces to (M,J)
      - adds priors for z_beta, z_alpha, and z_fc (and only those moving blocks)

    Other blocks (z_v, z_lambda, z_u_scale) are treated as fixed but must be present
    in `z` so the likelihood can be evaluated.
    """
    ll_mnj = loglik_mnj(z=z, inputs=inputs)
    ll_mj = _reduce_ll(ll_mnj, view="mj")

    lp_beta = logprior_normal(z["z_beta"], sigma_z["z_beta"])
    lp_alpha = logprior_normal(z["z_alpha"], sigma_z["z_alpha"])
    lp_fc = logprior_normal(z["z_fc"], sigma_z["z_fc"])

    return ll_mj + lp_beta + lp_alpha + lp_fc
