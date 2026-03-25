from __future__ import annotations

import tensorflow as tf


@tf.function(jit_compile=True, reduce_retracing=True)
def gibbs_gamma(
    njt: tf.Tensor,
    phi: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    eps: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    phi = tf.clip_by_value(phi, eps, tf.cast(1.0, phi.dtype) - eps)
    phi_b = phi[:, None]

    logp0 = -0.5 * tf.square(njt) / T0_sq - 0.5 * tf.math.log(T0_sq)
    logp1 = -0.5 * tf.square(njt) / T1_sq - 0.5 * tf.math.log(T1_sq)

    logit_prob1 = tf.math.log(phi_b) - tf.math.log1p(-phi_b) + (logp1 - logp0)
    prob1 = tf.math.sigmoid(logit_prob1)

    u = tf.random.stateless_uniform(
        shape=tf.shape(prob1),
        seed=seed,
        dtype=njt.dtype,
    )
    return tf.cast(u < prob1, njt.dtype)


@tf.function(jit_compile=True, reduce_retracing=True)
def gibbs_phi(
    gamma: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    eps: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    s = tf.reduce_sum(gamma, axis=-1)
    J = tf.cast(tf.shape(gamma)[-1], gamma.dtype)

    a_post = a_phi + s
    b_post = b_phi + (J - s)

    seeds = tf.random.experimental.stateless_split(seed, num=2)
    seed_x = seeds[0]
    seed_y = seeds[1]

    x = tf.random.stateless_gamma(
        shape=tf.shape(s),
        seed=seed_x,
        alpha=a_post,
        dtype=gamma.dtype,
    )
    y = tf.random.stateless_gamma(
        shape=tf.shape(s),
        seed=seed_y,
        alpha=b_post,
        dtype=gamma.dtype,
    )

    phi = x / (x + y)
    return tf.clip_by_value(phi, eps, tf.cast(1.0, phi.dtype) - eps)
