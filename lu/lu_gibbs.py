"""Gibbs updates for the Lu shrinkage inclusion indicators."""

from __future__ import annotations

import tensorflow as tf


@tf.function(jit_compile=True, reduce_retracing=True)
def gibbs_gamma(
    njt: tf.Tensor,
    gamma: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Run one Gibbs sweep for the market-product inclusion indicators.

    Update each product column of ``gamma`` conditional on the current values
    of all other columns.
    """

    J_int = tf.shape(gamma)[-1]
    J = tf.cast(J_int, gamma.dtype)

    # Precompute the log variances used in each conditional update.
    log_T0_sq = tf.math.log(T0_sq)
    log_T1_sq = tf.math.log(T1_sq)

    # Track the current number of active indicators in each market.
    s_init = tf.reduce_sum(gamma, axis=-1)

    def cond(j, seed_curr, gamma_curr, s_curr):
        """Continue until all product indices have been updated."""
        return j < J_int

    def body(j, seed_curr, gamma_curr, s_curr):
        """Update ``gamma[:, j]`` conditional on the remaining columns."""

        gamma_j = gamma_curr[:, j]
        njt_j = njt[:, j]

        # Form the leave-one-out count needed for the collapsed prior term.
        s_minus_j = s_curr - gamma_j

        # Compute the conditional log posterior kernels for gamma_j = 0 and 1.
        logp0 = (
            tf.math.log(b_phi + (J - 1.0 - s_minus_j))
            - 0.5 * tf.square(njt_j) / T0_sq
            - 0.5 * log_T0_sq
        )
        logp1 = (
            tf.math.log(a_phi + s_minus_j)
            - 0.5 * tf.square(njt_j) / T1_sq
            - 0.5 * log_T1_sq
        )

        # Convert log odds into the conditional probability of the slab state.
        prob1 = tf.math.sigmoid(logp1 - logp0)

        # Draw the updated inclusion indicator for each market.
        seeds = tf.random.experimental.stateless_split(seed_curr, num=2)
        next_seed = seeds[0]
        draw_seed = seeds[1]
        u = tf.random.stateless_uniform(
            shape=tf.shape(prob1),
            seed=draw_seed,
            dtype=gamma.dtype,
        )
        new_gamma_j = tf.cast(u < prob1, gamma.dtype)

        # Write the updated column back into gamma and refresh the count summary.
        one_hot_j = tf.one_hot(j, depth=J_int, dtype=gamma.dtype)
        gamma_next = (
            gamma_curr * (1.0 - one_hot_j[None, :])
            + new_gamma_j[:, None] * one_hot_j[None, :]
        )
        s_next = s_minus_j + new_gamma_j

        return j + 1, next_seed, gamma_next, s_next

    # Sweep once across all product indices.
    _, _, gamma_out, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, dtype=J_int.dtype),
            seed,
            gamma,
            s_init,
        ),
    )
    return gamma_out
