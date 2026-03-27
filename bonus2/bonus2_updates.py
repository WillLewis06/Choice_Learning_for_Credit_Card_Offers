"""
bonus2_updates.py

One-step MCMC updates for the Bonus Q2 sampler.

Design
- Each parameter block is updated with a TFP random-walk Metropolis step.
- Proposal mechanics are delegated to tfp.mcmc.RandomWalkMetropolis.
- Each one-step function evaluates an explicit compiled block log-posterior on the
  Bonus2PosteriorTF object.
- Randomness is driven by stateless seeds so the updates are compatible with
  jit_compile=True.

This module performs no input validation.
All tensors are assumed to have already been validated and normalized upstream.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from bonus2.bonus2_posterior import Bonus2PosteriorTF


def _make_rw_kernel(
    target_log_prob_fn,
    scale: tf.Tensor,
) -> tfp.mcmc.RandomWalkMetropolis:
    """Construct a Gaussian random-walk Metropolis kernel."""
    return tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale),
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_intercept_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_beta_intercept: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_beta_intercept_j."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.beta_intercept_block_logpost(
            z_beta_intercept_j=z_block,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )

    kernel = _make_rw_kernel(
        target_log_prob_fn=target_log_prob_fn, scale=k_beta_intercept
    )
    kernel_results = kernel.bootstrap_results(z_beta_intercept_j)
    z_new, kernel_results = kernel.one_step(
        current_state=z_beta_intercept_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_habit_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_beta_habit: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_beta_habit_j."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.beta_habit_block_logpost(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_block,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_beta_habit)
    kernel_results = kernel.bootstrap_results(z_beta_habit_j)
    z_new, kernel_results = kernel.one_step(
        current_state=z_beta_habit_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_peer_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_beta_peer: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_beta_peer_j."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.beta_peer_block_logpost(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_block,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_beta_peer)
    kernel_results = kernel.bootstrap_results(z_beta_peer_j)
    z_new, kernel_results = kernel.one_step(
        current_state=z_beta_peer_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_weekend_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_beta_weekend: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_beta_weekend_jw."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.beta_weekend_block_logpost(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_block,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )

    kernel = _make_rw_kernel(
        target_log_prob_fn=target_log_prob_fn, scale=k_beta_weekend
    )
    kernel_results = kernel.bootstrap_results(z_beta_weekend_jw)
    z_new, kernel_results = kernel.one_step(
        current_state=z_beta_weekend_jw,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def a_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_a: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_a_m."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.a_block_logpost(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_block,
            z_b_m=z_b_m,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_a)
    kernel_results = kernel.bootstrap_results(z_a_m)
    z_new, kernel_results = kernel.one_step(
        current_state=z_a_m,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def b_one_step(
    posterior: Bonus2PosteriorTF,
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    k_b: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for z_b_m."""

    def target_log_prob_fn(z_block: tf.Tensor) -> tf.Tensor:
        return posterior.b_block_logpost(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_block,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_b)
    kernel_results = kernel.bootstrap_results(z_b_m)
    z_new, kernel_results = kernel.one_step(
        current_state=z_b_m,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_new, accepted
