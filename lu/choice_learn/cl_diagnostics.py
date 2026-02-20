"""
Diagnostics for the choice-learn + Lu shrinkage sampler.

Accumulates running sums used to compute posterior means, and prints a compact
one-line progress summary each iteration.
"""

from __future__ import annotations

import tensorflow as tf


def round4(x: tf.Tensor) -> tf.Tensor:
    """Format a scalar float tensor as a fixed-point string with 4 decimals."""
    x = tf.convert_to_tensor(x)
    return tf.strings.as_string(x, precision=4, scientific=False)


@tf.function(reduce_retracing=True)
def report_iteration_progress(
    it: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    phi: tf.Tensor,
) -> None:
    """Print a compact per-iteration summary (tf.print)."""
    rms_E_bar = tf.sqrt(tf.reduce_mean(tf.square(E_bar)))
    rms_njt = tf.sqrt(tf.reduce_mean(tf.square(njt)))
    max_abs_njt = tf.reduce_max(tf.abs(njt))

    mean_gamma = tf.reduce_mean(gamma)
    mean_phi = tf.reduce_mean(phi)

    tf.print(
        "[ChoiceLearnShrinkage] it=",
        it,
        " | alpha=",
        round4(alpha),
        " | rms(E_bar)=",
        round4(rms_E_bar),
        ", rms(njt)=",
        round4(rms_njt),
        ", max|njt|=",
        round4(max_abs_njt),
        " | mean(gamma)=",
        round4(mean_gamma),
        ", mean(phi)=",
        round4(mean_phi),
    )


class ChoiceLearnShrinkageDiagnostics:
    """Running-sum diagnostics for posterior-mean summaries."""

    def __init__(self, T: int, J: int):
        """Allocate running sums for T markets and J products."""
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.sum_alpha = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.sum_E_bar = tf.Variable(tf.zeros([T], tf.float64), trainable=False)
        self.sum_njt = tf.Variable(tf.zeros([T, J], tf.float64), trainable=False)
        self.sum_phi = tf.Variable(tf.zeros([T], tf.float64), trainable=False)
        self.sum_gamma = tf.Variable(tf.zeros([T, J], tf.float64), trainable=False)

    @tf.function(reduce_retracing=True)
    def step(self, shrink, it: tf.Tensor) -> None:
        """Accumulate running sums from the current sampler state and print progress."""
        alpha = shrink.alpha.read_value()
        E_bar = shrink.E_bar.read_value()
        njt = shrink.njt.read_value()
        phi = shrink.phi.read_value()
        gamma = shrink.gamma.read_value()

        self.saved.assign_add(1)
        self.sum_alpha.assign_add(alpha)
        self.sum_E_bar.assign_add(E_bar)
        self.sum_njt.assign_add(njt)
        self.sum_phi.assign_add(phi)
        self.sum_gamma.assign_add(gamma)

        report_iteration_progress(
            it=it,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
            gamma=gamma,
            phi=phi,
        )

    def get_sums(self):
        """Return (saved, sum_alpha, sum_E_bar, sum_njt, sum_phi, sum_gamma)."""
        return (
            self.saved.read_value(),
            self.sum_alpha.read_value(),
            self.sum_E_bar.read_value(),
            self.sum_njt.read_value(),
            self.sum_phi.read_value(),
            self.sum_gamma.read_value(),
        )
