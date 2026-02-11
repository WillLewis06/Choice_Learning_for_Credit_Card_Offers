"""
Thin TensorFlow posterior wrapper for the stockpiling model.

This module contains ONLY:
- Normal priors on unconstrained sampler variables z_*
- Likelihood wrapper loglik_mn(z, ...) that calls ching.stockpiling_model
- Block-shaped log-posterior "views" used by elementwise RW-MH updates

All core mechanics (DP, CCPs, inventory filter, parameter transforms) live in:
  ching.stockpiling_model
"""

from __future__ import annotations

import tensorflow as tf

from ching import stockpiling_model as model

InventoryMaps = model.InventoryMaps


# =============================================================================
# Priors (up to additive constants)
# =============================================================================


def logprior_normal_mn(z_block_mn: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """
    Elementwise Normal(0, sigma^2) log prior (dropping additive constants).

    Args:
      z_block_mn: (M,N)
      sigma: scalar or broadcastable to (M,N)

    Returns:
      (M,N)
    """
    return -0.5 * tf.square(z_block_mn / sigma)


def logprior_normal_m(z_block_m: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """
    Elementwise Normal(0, sigma^2) log prior (dropping additive constants).

    Args:
      z_block_m: (M,)
      sigma: scalar or broadcastable to (M,)

    Returns:
      (M,)
    """
    return -0.5 * tf.square(z_block_m / sigma)


# =============================================================================
# Likelihood (per-consumer contributions) via core model
# =============================================================================


def loglik_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
) -> tf.Tensor:
    """
    Per-consumer log-likelihood contributions (M,N).

    Computes theta = unconstrained_to_theta(z), solves CCPs via DP under theta,
    then runs the forward filter integrating out latent inventory.

    Returns:
      (M,N)
    """
    theta = model.unconstrained_to_theta(z)
    return model.loglik_mn_from_theta(
        theta=theta,
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )


# =============================================================================
# Block-shaped log-posterior "views" for RW-MH
# =============================================================================


def _logpost_consumer_block_mn(
    z: dict[str, tf.Tensor],
    z_key: str,
    sigma_key: str,
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Log posterior view for a single per-(m,n) block.

    Returns:
      (M,N) = loglik_mn(z, ...) + logprior_normal_mn(z[z_key], sigmas[sigma_key])
    """
    ll = loglik_mn(
        z=z,
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    lp = logprior_normal_mn(z[z_key], sigmas[sigma_key])
    return ll + lp


def logpost_z_beta_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_beta."""
    return _logpost_consumer_block_mn(
        z=z,
        z_key="z_beta",
        sigma_key="z_beta",
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
        sigmas=sigmas,
    )


def logpost_z_alpha_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_alpha."""
    return _logpost_consumer_block_mn(
        z=z,
        z_key="z_alpha",
        sigma_key="z_alpha",
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
        sigmas=sigmas,
    )


def logpost_z_v_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_v."""
    return _logpost_consumer_block_mn(
        z=z,
        z_key="z_v",
        sigma_key="z_v",
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
        sigmas=sigmas,
    )


def logpost_z_fc_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_fc."""
    return _logpost_consumer_block_mn(
        z=z,
        z_key="z_fc",
        sigma_key="z_fc",
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
        sigmas=sigmas,
    )


def logpost_z_lambda_mn(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,N) log posterior view for updating z_lambda."""
    return _logpost_consumer_block_mn(
        z=z,
        z_key="z_lambda",
        sigma_key="z_lambda",
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
        sigmas=sigmas,
    )


def logpost_u_scale_m(
    z: dict[str, tf.Tensor],
    a_imt: tf.Tensor,
    p_state_mt: tf.Tensor,
    u_m: tf.Tensor,
    price_vals: tf.Tensor,
    P_price: tf.Tensor,
    pi_I0: tf.Tensor,
    waste_cost: tf.Tensor,
    eps: tf.Tensor,
    tol: tf.Tensor,
    max_iter: tf.Tensor,
    maps: InventoryMaps,
    sigmas: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    (M,) log posterior view for updating z_u_scale.

    u_scale is shared within market, so the likelihood contribution is summed over n:
      ll_m = sum_n loglik_mn[m,n]
    """
    ll_mn = loglik_mn(
        z=z,
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        maps=maps,
    )
    ll_m = tf.reduce_sum(ll_mn, axis=1)
    lp_m = logprior_normal_m(z["z_u_scale"], sigmas["z_u_scale"])
    return ll_m + lp_m
