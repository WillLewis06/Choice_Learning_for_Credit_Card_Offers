from __future__ import annotations

import tensorflow as tf


def round4(x: tf.Tensor) -> tf.Tensor:
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(shrink: "LuShrinkageEstimator", it) -> None:
    """
    Print current state values (scalars + cheap aggregates) at end of iteration.

    TF-compatible: uses tf.print (no .numpy(), no Python floats).
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
    """
    Owns:
      - running-sum accumulation for posterior-mean summaries
      - per-iteration progress printing

    Intended call pattern from LuShrinkageEstimator:
      diag = LuShrinkageDiagnostics(T, J)
      for it in range(n_iter):
          ... update blocks ...
          diag.step(self, it)
      saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = diag.get_sums()
    """

    def __init__(self, T: int, J: int):
        self.T = int(T)
        self.J = int(J)

        # TF-friendly mutable state.
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.sum_beta = tf.Variable(tf.zeros([2], dtype=tf.float64), trainable=False)
        self.sum_sigma = tf.Variable(
            tf.constant(0.0, dtype=tf.float64), trainable=False
        )
        self.sum_E_bar = tf.Variable(
            tf.zeros([self.T], dtype=tf.float64), trainable=False
        )
        self.sum_njt = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )
        self.sum_phi = tf.Variable(
            tf.zeros([self.T], dtype=tf.float64), trainable=False
        )
        self.sum_gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )

    def _accumulate_draw(self, shrink: "LuShrinkageEstimator") -> None:
        """
        Accumulate the current state into running sums.
        No burn-in/thinning logic; every iteration is retained.
        """
        self.saved.assign_add(1)
        self.sum_beta.assign_add(tf.stack([shrink.beta_p, shrink.beta_w], axis=0))
        self.sum_sigma.assign_add(tf.exp(shrink.r))
        self.sum_E_bar.assign_add(shrink.E_bar)
        self.sum_njt.assign_add(shrink.njt)
        self.sum_phi.assign_add(shrink.phi)
        self.sum_gamma.assign_add(shrink.gamma)

    @tf.function
    def step(self, shrink: "LuShrinkageEstimator", it) -> None:
        """
        Called once per iteration:
          - accumulate current draw into running sums
          - print progress line
        """
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
        """
        Return the raw running sums for the estimator to finalize results.
        TF-compatible: returns tensors (no Python int conversion).
        """
        return (
            self.saved.read_value(),
            self.sum_beta.read_value(),
            self.sum_sigma.read_value(),
            self.sum_E_bar.read_value(),
            self.sum_njt.read_value(),
            self.sum_phi.read_value(),
            self.sum_gamma.read_value(),
        )
