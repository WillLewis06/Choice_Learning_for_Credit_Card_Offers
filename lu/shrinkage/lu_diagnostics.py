"""
Diagnostics and reporting for the Lu shrinkage sampler.

This module has two responsibilities:
  - Maintain running sums needed to compute posterior means after sampling.
  - Print a compact, TF-compatible progress line at the end of each iteration.

There is no burn-in or thinning logic here: every iteration is accumulated.
That choice is controlled by the caller (e.g. by choosing n_iter or by adding
burn-in/thinning outside this class).
"""

from __future__ import annotations

import tensorflow as tf


def round4(x: tf.Tensor) -> tf.Tensor:
    """Format tensor values as strings with 4 decimal places (no scientific notation)."""
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(shrink: "LuShrinkageEstimator", it) -> None:
    """Print a one-line summary of the current MCMC state.

    This is designed to be cheap and informative:
      - global scalars: beta_p, beta_w, sigma
      - simple scale checks: norms of E_bar and njt
      - sparsity summaries: mean(gamma), mean(phi)

    Implementation notes:
      - Uses tf.print so it can run inside tf.function.
      - Avoids .numpy() and Python-side formatting.
    """
    beta_p = shrink.beta_p
    beta_w = shrink.beta_w
    sigma = tf.exp(shrink.r)

    E_bar_norm = tf.norm(shrink.E_bar)
    njt_norm = tf.norm(shrink.njt)

    gamma_mean = tf.reduce_mean(shrink.gamma)
    phi_mean = tf.reduce_mean(shrink.phi)

    tf.print(
        "[LuShrinkage] it=",
        it,
        " | beta_p=",
        round4(beta_p),
        ", beta_w=",
        round4(beta_w),
        ", sigma=",
        round4(sigma),
        " | E_bar_norm=",
        round4(E_bar_norm),
        ", njt_norm=",
        round4(njt_norm),
        " | mean(gamma)=",
        round4(gamma_mean),
        ", mean(phi)=",
        round4(phi_mean),
    )


class LuShrinkageDiagnostics:
    """Running-sum diagnostics for posterior means and iteration reporting.

    Usage:
      diag = LuShrinkageDiagnostics(T, J)
      for it in range(n_iter):
          ... update sampler state ...
          diag.step(shrink, it)

      saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = diag.get_sums()

    Stored sums:
      - saved: number of retained draws
      - sum_beta: sum of [beta_p, beta_w] across draws
      - sum_sigma: sum of sigma = exp(r) across draws
      - sum_E_bar: sum of E_bar across draws
      - sum_njt: sum of njt across draws
      - sum_phi: sum of phi across draws
      - sum_gamma: sum of gamma across draws
    """

    def __init__(self, T: int, J: int):
        """Create zero-initialized running sums for a (T, J) problem."""
        self.T = int(T)
        self.J = int(J)

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

    def _accumulate_draw(self, shrink: "LuShrinkageEstimator") -> None:
        """Add the current sampler state to the running sums."""
        self.saved.assign_add(1)
        self.sum_beta.assign_add(tf.stack([shrink.beta_p, shrink.beta_w], axis=0))
        self.sum_sigma.assign_add(tf.exp(shrink.r))

        self.sum_E_bar.assign_add(shrink.E_bar)
        self.sum_njt.assign_add(shrink.njt)

        self.sum_phi.assign_add(shrink.phi)
        self.sum_gamma.assign_add(shrink.gamma)

    @tf.function(reduce_retracing=True)
    def step(self, shrink: "LuShrinkageEstimator", it) -> None:
        """Record one iteration: accumulate sums and print a progress line."""
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
