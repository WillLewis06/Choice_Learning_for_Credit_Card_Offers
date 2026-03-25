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
    J_int = tf.shape(gamma)[-1]
    J = tf.cast(J_int, gamma.dtype)

    log_T0_sq = tf.math.log(T0_sq)
    log_T1_sq = tf.math.log(T1_sq)

    gamma_curr = gamma
    s_curr = tf.reduce_sum(gamma_curr, axis=-1)

    def cond(j, seed_curr, gamma_curr, s_curr):
        del seed_curr, gamma_curr, s_curr
        return j < J_int

    def body(j, seed_curr, gamma_curr, s_curr):
        seeds = tf.random.experimental.stateless_split(seed_curr, num=2)
        next_seed = seeds[0]
        draw_seed = seeds[1]

        gamma_j = gamma_curr[:, j]
        njt_j = njt[:, j]

        s_minus_j = s_curr - gamma_j

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

        m = tf.maximum(logp0, logp1)
        prob1 = tf.exp(logp1 - m) / (tf.exp(logp0 - m) + tf.exp(logp1 - m))

        u = tf.random.stateless_uniform(
            shape=tf.shape(prob1),
            seed=draw_seed,
            dtype=gamma.dtype,
        )
        new_gamma_j = tf.cast(u < prob1, gamma.dtype)

        one_hot_j = tf.one_hot(j, depth=J_int, dtype=gamma.dtype)
        gamma_next = (
            gamma_curr * (1.0 - one_hot_j[None, :])
            + new_gamma_j[:, None] * one_hot_j[None, :]
        )
        s_next = s_minus_j + new_gamma_j

        return j + 1, next_seed, gamma_next, s_next

    _, _, gamma_out, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, dtype=J_int.dtype),
            seed,
            gamma_curr,
            s_curr,
        ),
    )
    return gamma_out
