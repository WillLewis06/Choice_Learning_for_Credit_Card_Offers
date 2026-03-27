"""One-step MCMC updates for the Bonus Q2 sampler.

This module provides one random-walk Metropolis step per parameter block using
TensorFlow Probability. Input validation is handled elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf
import tensorflow_probability as tfp

from bonus2.bonus2_posterior import Bonus2PosteriorTF


def _make_rw_kernel(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    scale: tf.Tensor,
) -> tfp.mcmc.RandomWalkMetropolis:
    """Build a Gaussian random-walk Metropolis kernel."""
    return tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale),
    )


def _rw_metropolis_one_step(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    current_state: tf.Tensor,
    scale: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Run one random-walk Metropolis step and return the new state."""
    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=scale)
    kernel_results = kernel.bootstrap_results(current_state)
    seed = tf.random.uniform(
        shape=(2,),
        minval=0,
        maxval=2**31 - 1,
        dtype=tf.int32,
    )
    new_state, kernel_results = kernel.one_step(
        current_state=current_state,
        previous_kernel_results=kernel_results,
        seed=seed,
    )
    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return new_state, accepted


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_beta_intercept_j,
        scale=k_beta_intercept,
    )


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_beta_habit_j,
        scale=k_beta_habit,
    )


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_beta_peer_j,
        scale=k_beta_peer,
    )


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_beta_weekend_jw,
        scale=k_beta_weekend,
    )


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_a_m,
        scale=k_a,
    )


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

    return _rw_metropolis_one_step(
        target_log_prob_fn=target_log_prob_fn,
        current_state=z_b_m,
        scale=k_b,
    )
