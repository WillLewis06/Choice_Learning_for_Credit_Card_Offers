"""
Diagnostics and reporting for the Ching stockpiling sampler.

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


def report_iteration_progress(shrink: "StockpilingEstimator", it) -> None:
    """Print a one-line summary of the current MCMC state.

    This is designed to be cheap and informative:
      - simple global summaries: means of beta, alpha, v, fc, lambda_c
      - scale check: mean and norm of u_scale

    Implementation notes:
      - Uses tf.print so it can run inside tf.function.
      - Avoids .numpy() and Python-side formatting.
    """
    z = shrink.z

    beta = tf.math.sigmoid(z["z_beta"])
    alpha = tf.exp(z["z_alpha"])
    v = tf.exp(z["z_v"])
    fc = tf.exp(z["z_fc"])
    lambda_c = tf.math.sigmoid(z["z_lambda"])
    u_scale = tf.exp(z["z_u_scale"])

    beta_mean = tf.reduce_mean(beta)
    alpha_mean = tf.reduce_mean(alpha)
    v_mean = tf.reduce_mean(v)
    fc_mean = tf.reduce_mean(fc)
    lambda_c_mean = tf.reduce_mean(lambda_c)

    u_scale_mean = tf.reduce_mean(u_scale)

    tf.print(
        "[Stockpiling] it=",
        it,
        " | mean(beta)=",
        round4(beta_mean),
        ", mean(alpha)=",
        round4(alpha_mean),
        ", mean(v)=",
        round4(v_mean),
        ", mean(fc)=",
        round4(fc_mean),
        ", mean(lambda_c)=",
        round4(lambda_c_mean),
        " | u_scale_mean=",
        round4(u_scale_mean),
    )


class StockpilingDiagnostics:
    """Running-sum diagnostics for posterior means and iteration reporting.

    Usage:
      diag = StockpilingDiagnostics(M, N)
      for it in range(n_iter):
          ... update sampler state ...
          diag.step(estimator, it)

      saved, sum_beta, sum_alpha, sum_v, sum_fc, sum_lambda_c, sum_u_scale = diag.get_sums()

    Stored sums:
      - saved: number of retained draws
      - sum_beta: sum of beta across draws (shape [M, N])
      - sum_alpha: sum of alpha across draws (shape [M, N])
      - sum_v: sum of v across draws (shape [M, N])
      - sum_fc: sum of fc across draws (shape [M, N])
      - sum_lambda_c: sum of lambda_c across draws (shape [M, N])
      - sum_u_scale: sum of u_scale across draws (shape [M])
    """

    def __init__(self, M: int, N: int):
        """Create zero-initialized running sums for an (M, N) problem."""
        self.M = int(M)
        self.N = int(N)

        # Mutable TF state so accumulation works inside tf.function.
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.sum_beta = tf.Variable(
            tf.zeros([self.M, self.N], tf.float64), trainable=False
        )
        self.sum_alpha = tf.Variable(
            tf.zeros([self.M, self.N], tf.float64), trainable=False
        )
        self.sum_v = tf.Variable(
            tf.zeros([self.M, self.N], tf.float64), trainable=False
        )
        self.sum_fc = tf.Variable(
            tf.zeros([self.M, self.N], tf.float64), trainable=False
        )
        self.sum_lambda_c = tf.Variable(
            tf.zeros([self.M, self.N], tf.float64), trainable=False
        )

        self.sum_u_scale = tf.Variable(tf.zeros([self.M], tf.float64), trainable=False)

    def _accumulate_draw(self, shrink: "StockpilingEstimator") -> None:
        """Add the current sampler state to the running sums."""
        self.saved.assign_add(1)

        z = shrink.z
        beta = tf.math.sigmoid(z["z_beta"])
        alpha = tf.exp(z["z_alpha"])
        v = tf.exp(z["z_v"])
        fc = tf.exp(z["z_fc"])
        lambda_c = tf.math.sigmoid(z["z_lambda"])
        u_scale = tf.exp(z["z_u_scale"])

        self.sum_beta.assign_add(beta)
        self.sum_alpha.assign_add(alpha)
        self.sum_v.assign_add(v)
        self.sum_fc.assign_add(fc)
        self.sum_lambda_c.assign_add(lambda_c)
        self.sum_u_scale.assign_add(u_scale)

    @tf.function(reduce_retracing=True)
    def step(self, shrink: "StockpilingEstimator", it) -> None:
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
            self.sum_alpha.read_value(),
            self.sum_v.read_value(),
            self.sum_fc.read_value(),
            self.sum_lambda_c.read_value(),
            self.sum_u_scale.read_value(),
        )
