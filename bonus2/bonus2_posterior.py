"""
Thin TensorFlow posterior wrapper for Bonus Q2 (habit + peer + DOW + seasonality) MNL model.

This module contains ONLY:
  - Priors on unconstrained sampler variables z_*
      * Normal priors for most blocks
      * Beta(kappa, 1) prior for decay_rate_j, applied in constrained space with
        change-of-variables (sigmoid Jacobian) when sampling in z-space
  - Likelihood wrapper loglik_mnt(z, inputs) that calls bonus2.bonus2_model
  - Log-posterior "views" used by RW-MH updates

All core mechanics (parameter transforms, habit recursion, peer exposure, utilities, MNL) live in:
  bonus2.bonus2_model

Observed inputs (in `inputs` dict):
  y_mit          (M,N,T)   int32/int64      choices; 0=outside, j+1=inside product j
  delta_mj       (M,J)     float64          Phase-1 baseline utilities (fixed)
  dow_t          (T,)      int32/int64      weekday index in {0..6}
  season_sin_kt  (K,T)     float64          sin((k+1)*season_angle_t[t])
  season_cos_kt  (K,T)     float64          cos((k+1)*season_angle_t[t])
  peer_adj_m     tuple[M]  tf.SparseTensor  (N,N) known within-market adjacency
  L              scalar    int32/int64      peer lookback window length
  decay_rate_eps scalar    float64          optional numeric guard for decay_rate; e.g. 1e-6
  kappa_decay    scalar    float64          Beta prior shape kappa for decay_rate_j ~ Beta(kappa, 1)

Unconstrained sampler variables z (keys expected in `z` dict):
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

Likelihood contributions:
  loglik_mnt : (M,N,T)

Supported log-posterior views:
  - "all": scalar (sum over M,N,T)
  - "m":   (M,)   (sum over N,T)   [valid only when updating a block with leading dim M]
"""

from __future__ import annotations

import tensorflow as tf

from bonus2 import bonus2_model as model


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


def logprior_decay_beta_kappa1_on_z(
    z_decay_rate_j: tf.Tensor,
    kappa_decay: tf.Tensor,
    decay_rate_eps: tf.Tensor,
) -> tf.Tensor:
    """
    Log prior for z_decay_rate_j when decay_rate_j ~ Beta(kappa, 1) in constrained space,
    and decay_rate_j = sigmoid(z_decay_rate_j) in the sampler.

    This returns log p(z) up to additive constants:
      log p(z) = log p(decay) + log |d decay / d z|

    Where:
      p(decay) ∝ decay^(kappa-1)  (since Beta(kappa,1))
      d decay/dz = decay*(1-decay) for sigmoid

    So (dropping constants):
      log p(z) ∝ (kappa-1)*log(decay) + log(decay) + log(1-decay)
              = kappa*log(decay) + log(1-decay)

    Args:
      z_decay_rate_j: (J,)
      kappa_decay: scalar > 0
      decay_rate_eps: scalar >= 0 numeric guard; used to clip decay into [eps, 1-eps]

    Returns:
      (J,) float64 tensor of elementwise log prior contributions.
    """
    z = tf.cast(z_decay_rate_j, tf.float64)
    kappa = tf.cast(kappa_decay, tf.float64)

    eps = tf.cast(decay_rate_eps, tf.float64)
    # Never allow exact 0/1 inside logs.
    eps_safe = tf.maximum(eps, tf.constant(1e-12, tf.float64))

    decay = tf.math.sigmoid(z)
    decay = tf.clip_by_value(decay, eps_safe, 1.0 - eps_safe)

    return kappa * tf.math.log(decay) + tf.math.log1p(-decay)


# =============================================================================
# Likelihood via core model
# =============================================================================


def _cast_z_to_f64(z: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Ensure all z_* blocks are float64."""
    return {k: tf.cast(v, tf.float64) for k, v in z.items()}


def loglik_mnt(z: dict[str, tf.Tensor], inputs: dict[str, tf.Tensor]) -> tf.Tensor:
    """
    Per-(market, consumer, time) log-likelihood contributions (M,N,T).

    Returns:
      loglik_mnt: (M,N,T) float64
    """
    z64 = _cast_z_to_f64(z)
    theta = model.unconstrained_to_theta(z64)

    return model.loglik_mnt_from_theta(
        theta=theta,
        y_mit=inputs["y_mit"],
        delta_mj=inputs["delta_mj"],
        dow_t=inputs["dow_t"],
        season_sin_kt=inputs["season_sin_kt"],
        season_cos_kt=inputs["season_cos_kt"],
        peer_adj_m=inputs["peer_adj_m"],
        L=inputs["L"],
        decay_rate_eps=inputs.get("decay_rate_eps", tf.constant(0.0, tf.float64)),
    )


# =============================================================================
# Log-posterior views for RW-MH
# =============================================================================


def _reduce_ll(ll_mnt: tf.Tensor, view: str) -> tf.Tensor:
    """
    Reduce loglik_mnt (M,N,T) to a view.

    view:
      "all": sum over M,N,T -> scalar
      "m":   sum over N,T   -> (M,)
    """
    if view == "all":
        return tf.reduce_sum(ll_mnt)
    if view == "m":
        return tf.reduce_sum(ll_mnt, axis=[1, 2])
    raise ValueError(f"Unknown view='{view}' (expected one of: 'all','m').")


def _reduce_lp(lp_block: tf.Tensor, view: str) -> tf.Tensor:
    """
    Reduce a prior tensor to match the requested view.

    For "all": return scalar sum.
    For "m":   sum over all axes except axis 0 -> (M,)

    Note: "m" is only valid if lp_block has leading dimension M.
    """
    if view == "all":
        return tf.reduce_sum(lp_block)

    if view == "m":
        # Sum over all axes except the leading axis.
        # If lp_block is rank-1 (M,), this is a no-op.
        rank = tf.rank(lp_block)
        axes = tf.range(1, rank)
        return tf.reduce_sum(lp_block, axis=axes)

    raise ValueError(f"Unknown view='{view}' (expected one of: 'all','m').")


def _logpost_view(
    z: dict[str, tf.Tensor],
    z_key: str,
    view: str,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Generic log-posterior view for RW-MH (Normal prior on z_key).

    This recomputes the full likelihood for each call; this matches the Ching
    posterior design (thin wrapper; computation lives in the model).

    Returns:
      Tensor shaped according to `view`.
    """
    ll_mnt = loglik_mnt(z=z, inputs=inputs)
    ll_view = _reduce_ll(ll_mnt, view=view)

    lp_block = logprior_normal(z[z_key], sigma_z[z_key])
    lp_view = _reduce_lp(lp_block, view=view)

    return ll_view + lp_view


def _logpost_decay_rate_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Scalar log-posterior view for updating z_decay_rate_j, using
    decay_rate_j ~ Beta(kappa_decay, 1) and decay_rate_j = sigmoid(z_decay_rate_j).

    Returns:
      scalar float64
    """
    ll_mnt = loglik_mnt(z=z, inputs=inputs)
    ll = tf.reduce_sum(ll_mnt)

    decay_rate_eps = inputs.get("decay_rate_eps", tf.constant(0.0, tf.float64))
    kappa_decay = inputs["kappa_decay"]

    lp = tf.reduce_sum(
        logprior_decay_beta_kappa1_on_z(
            z_decay_rate_j=z["z_decay_rate_j"],
            kappa_decay=kappa_decay,
            decay_rate_eps=decay_rate_eps,
        )
    )

    return ll + lp


# =============================================================================
# Per-block logpost wrappers (use view="all" for correctness under MNL coupling)
# =============================================================================


def logpost_z_beta_market_mj_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_beta_market_mj."""
    return _logpost_view(
        z=z, z_key="z_beta_market_mj", view="all", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_habit_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_beta_habit_j."""
    return _logpost_view(
        z=z, z_key="z_beta_habit_j", view="all", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_peer_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_beta_peer_j."""
    return _logpost_view(
        z=z, z_key="z_beta_peer_j", view="all", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_decay_rate_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """
    Scalar log posterior view for updating z_decay_rate_j.

    Note: sigma_z is ignored for this block; decay uses Beta(kappa_decay, 1) prior.
    """
    _ = sigma_z
    return _logpost_decay_rate_j_all(z=z, inputs=inputs)


def logpost_z_beta_dow_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_beta_dow_m."""
    return _logpost_view(
        z=z, z_key="z_beta_dow_m", view="all", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_beta_dow_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_beta_dow_j."""
    return _logpost_view(
        z=z, z_key="z_beta_dow_j", view="all", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_a_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_a_m."""
    return _logpost_view(z=z, z_key="z_a_m", view="all", inputs=inputs, sigma_z=sigma_z)


def logpost_z_b_m_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_b_m."""
    return _logpost_view(z=z, z_key="z_b_m", view="all", inputs=inputs, sigma_z=sigma_z)


def logpost_z_a_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_a_j."""
    return _logpost_view(z=z, z_key="z_a_j", view="all", inputs=inputs, sigma_z=sigma_z)


def logpost_z_b_j_all(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """Scalar log posterior view for updating z_b_j."""
    return _logpost_view(z=z, z_key="z_b_j", view="all", inputs=inputs, sigma_z=sigma_z)


# =============================================================================
# Optional: marketwise views (safe only when the updated block factorizes by market)
# =============================================================================


def logpost_z_beta_dow_m_m(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,) log posterior view for updating z_beta_dow_m (marketwise reduction)."""
    return _logpost_view(
        z=z, z_key="z_beta_dow_m", view="m", inputs=inputs, sigma_z=sigma_z
    )


def logpost_z_a_m_m(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,) log posterior view for updating z_a_m (marketwise reduction)."""
    return _logpost_view(z=z, z_key="z_a_m", view="m", inputs=inputs, sigma_z=sigma_z)


def logpost_z_b_m_m(
    z: dict[str, tf.Tensor],
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
) -> tf.Tensor:
    """(M,) log posterior view for updating z_b_m (marketwise reduction)."""
    return _logpost_view(z=z, z_key="z_b_m", view="m", inputs=inputs, sigma_z=sigma_z)
