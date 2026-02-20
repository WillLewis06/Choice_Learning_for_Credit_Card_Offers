"""
Diagnostics and reporting for the Lu shrinkage sampler.

Responsibilities:
  - Maintain running sums used to compute posterior means after sampling.
  - Print a compact, TF-compatible progress line at the end of each iteration.

Every iteration is accumulated by design. Burn-in and thinning are not implemented.
"""

from __future__ import annotations

from typing import Protocol

import tensorflow as tf


class _ShrinkageState(Protocol):
    """Minimal sampler interface required by diagnostics/reporting."""

    beta_p: tf.Tensor
    beta_w: tf.Tensor
    r: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    phi: tf.Tensor
    gamma: tf.Tensor


def format4(x: tf.Tensor) -> tf.Tensor:
    """Format tensor values as strings with 4 decimal places (no scientific notation)."""
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(shrink: _ShrinkageState, it: tf.Tensor) -> None:
    """Print a one-line summary of the current MCMC state via tf.print."""
    sigma = tf.exp(shrink.r)

    # Dimension-stable scale summaries (RMS).
    E_bar_rms = tf.sqrt(tf.reduce_mean(tf.square(shrink.E_bar)))
    njt_rms = tf.sqrt(tf.reduce_mean(tf.square(shrink.njt)))

    gamma_mean = tf.reduce_mean(shrink.gamma)
    phi_mean = tf.reduce_mean(shrink.phi)

    tf.print(
        "[LuShrinkage] it=",
        it,
        " | beta_p=",
        format4(shrink.beta_p),
        ", beta_w=",
        format4(shrink.beta_w),
        ", sigma=",
        format4(sigma),
        " | E_bar_rms=",
        format4(E_bar_rms),
        ", njt_rms=",
        format4(njt_rms),
        " | mean(gamma)=",
        format4(gamma_mean),
        ", mean(phi)=",
        format4(phi_mean),
    )


class LuShrinkageDiagnostics:
    """Running-sum diagnostics for posterior means and iteration reporting."""

    def __init__(self, T: int, J: int):
        """Create zero-initialized running sums for a (T, J) problem."""
        self.T = T
        self.J = J

        # Mutable TF state so accumulation works inside tf.function.
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.sum_beta = tf.Variable(tf.zeros([2], dtype=tf.float64), trainable=False)
        self.sum_sigma = tf.Variable(tf.constant(0.0, tf.float64), trainable=False)

        self.sum_E_bar = tf.Variable(tf.zeros([self.T], tf.float64), trainable=False)
        self.sum_njt = tf.Variable(
            tf.zeros([self.T, self.J], tf.float64), trainable=False
        )

        self.sum_phi = tf.Variable(tf.zeros([self.T], tf.float64), trainable=False)
        self.sum_gamma = tf.Variable(
            tf.zeros([self.T, self.J], tf.float64), trainable=False
        )

    def _accumulate_draw(self, shrink: _ShrinkageState) -> None:
        """Add the current sampler state to the running sums."""
        self.saved.assign_add(1)
        self.sum_beta.assign_add(tf.stack([shrink.beta_p, shrink.beta_w], axis=0))
        self.sum_sigma.assign_add(tf.exp(shrink.r))

        self.sum_E_bar.assign_add(shrink.E_bar)
        self.sum_njt.assign_add(shrink.njt)

        self.sum_phi.assign_add(shrink.phi)
        self.sum_gamma.assign_add(shrink.gamma)

    @tf.function(reduce_retracing=True)
    def step(self, shrink: _ShrinkageState, it: tf.Tensor) -> None:
        """Accumulate one draw and print a progress line."""
        self._accumulate_draw(shrink)
        report_iteration_progress(shrink, it)

    def get_sums(
        self,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        """Return raw running sums as tensors for downstream aggregation."""
        return (
            self.saved.read_value(),
            self.sum_beta.read_value(),
            self.sum_sigma.read_value(),
            self.sum_E_bar.read_value(),
            self.sum_njt.read_value(),
            self.sum_phi.read_value(),
            self.sum_gamma.read_value(),
        )
