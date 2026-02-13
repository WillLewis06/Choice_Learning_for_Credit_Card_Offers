"""
bonus2/bonus2_updates.py

Block-wise random-walk Metropolis–Hastings (RW-MH) updates for the Bonus Q2 model.

Pattern (copied from the Ching stockpiling framework):
  - One function per parameter block update.
  - Each update holds the other blocks fixed, builds the full `z` dictionary,
    evaluates the appropriate scalar log-posterior view, and calls the RW-MH kernel.

Conventions and shapes:
  - M = number of markets
  - N = number of consumers per market
  - J = number of products (inside options)
  - K = number of seasonal harmonics

Unconstrained blocks (sampler state) — MUST match bonus2_model.py / bonus2_posterior.py:
  - z_beta_market_mj : (M, J)   market×product intercept shifts
  - z_beta_habit_j   : (J,)     habit sensitivity
  - z_beta_peer_j    : (J,)     peer sensitivity
  - z_decay_rate_j   : (J,)     habit decay latent (mapped to (0,1) in the model transform)
  - z_beta_dow_m     : (M, 7)   market day-of-week shifts
  - z_beta_dow_j     : (J, 7)   product day-of-week shifts
  - z_a_m, z_b_m     : (M, K)   market seasonal Fourier coefficients
  - z_a_j, z_b_j     : (J, K)   product seasonal Fourier coefficients

Notes:
  - Priors are handled inside bonus2_posterior.
    * Most blocks: Normal prior on z-block (via sigma_z[z_key])
    * Decay block: Beta(kappa_decay, 1) prior on decay_rate_j in constrained space
      with sigmoid Jacobian (sigma_z is ignored for this block).
  - All tensors are assumed to be tf.float64 and validated upstream.
  - `k` must be a scalar tf.float64 (proposal scale).
"""

from __future__ import annotations

import tensorflow as tf

from toolbox.mcmc_kernels import rw_mh_step

from bonus2.bonus2_posterior import (
    logpost_z_a_j_all,
    logpost_z_a_m_all,
    logpost_z_b_j_all,
    logpost_z_b_m_all,
    logpost_z_beta_dow_j_all,
    logpost_z_beta_dow_m_all,
    logpost_z_beta_habit_j_all,
    logpost_z_beta_market_mj_all,
    logpost_z_beta_peer_j_all,
    logpost_z_decay_rate_j_all,
)


def _pack_z(
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Build the full `z` dict expected by bonus2_model / bonus2_posterior."""
    return {
        "z_beta_market_mj": z_beta_market_mj,
        "z_beta_habit_j": z_beta_habit_j,
        "z_beta_peer_j": z_beta_peer_j,
        "z_decay_rate_j": z_decay_rate_j,
        "z_beta_dow_m": z_beta_dow_m,
        "z_beta_dow_j": z_beta_dow_j,
        "z_a_m": z_a_m,
        "z_b_m": z_b_m,
        "z_a_j": z_a_j,
        "z_b_j": z_b_j,
    }


@tf.function(reduce_retracing=True)
def update_z_beta_market_mj(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_beta_market_mj (shape (M, J))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_t,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_beta_market_mj_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta_market_mj, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_beta_habit_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_beta_habit_j (shape (J,))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_t,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_beta_habit_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta_habit_j, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_beta_peer_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_beta_peer_j (shape (J,))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_t,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_beta_peer_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta_peer_j, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_decay_rate_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_decay_rate_j (shape (J,))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_t,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        # sigma_z is accepted for API consistency; posterior ignores it for decay.
        return logpost_z_decay_rate_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_decay_rate_j, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_beta_dow_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_beta_dow_m (shape (M, 7))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_t,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_beta_dow_m_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta_dow_m, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_beta_dow_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_beta_dow_j (shape (J, 7))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_t,
            z_a_m,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_beta_dow_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_beta_dow_j, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_a_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_a_m (shape (M, K))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_t,
            z_b_m,
            z_a_j,
            z_b_j,
        )
        return logpost_z_a_m_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_a_m, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_b_m(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_b_m (shape (M, K))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_t,
            z_a_j,
            z_b_j,
        )
        return logpost_z_b_m_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_b_m, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_a_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_a_j (shape (J, K))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_t,
            z_b_j,
        )
        return logpost_z_a_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_a_j, logp, k, rng)


@tf.function(reduce_retracing=True)
def update_z_b_j(
    rng: tf.random.Generator,
    k: tf.Tensor,
    inputs: dict[str, tf.Tensor],
    sigma_z: dict[str, tf.Tensor],
    z_beta_market_mj: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_decay_rate_j: tf.Tensor,
    z_beta_dow_m: tf.Tensor,
    z_beta_dow_j: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    z_a_j: tf.Tensor,
    z_b_j: tf.Tensor,
):
    """RW-MH update for z_b_j (shape (J, K))."""

    def logp(z_t: tf.Tensor) -> tf.Tensor:
        z = _pack_z(
            z_beta_market_mj,
            z_beta_habit_j,
            z_beta_peer_j,
            z_decay_rate_j,
            z_beta_dow_m,
            z_beta_dow_j,
            z_a_m,
            z_b_m,
            z_a_j,
            z_t,
        )
        return logpost_z_b_j_all(z=z, inputs=inputs, sigma_z=sigma_z)

    return rw_mh_step(z_b_j, logp, k, rng)
