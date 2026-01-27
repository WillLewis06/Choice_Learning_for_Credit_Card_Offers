from __future__ import annotations

import tensorflow as tf


def init_progress_state(shrink: "LuShrinkageEstimator") -> dict:
    """
    Initialize a lightweight snapshot of the current state.
    Scalars + cheap aggregates only.

    TF-compatible: returns tensors (no .numpy(), no Python floats).
    """
    return {
        "beta_p": tf.identity(shrink.beta_p),
        "beta_w": tf.identity(shrink.beta_w),
        "r": tf.identity(shrink.r),
        "E_bar_norm": tf.norm(shrink.E_bar),
        "njt_norm": tf.norm(shrink.njt),
        "gamma_mean": tf.reduce_mean(tf.cast(shrink.gamma, tf.float64)),
        "phi_mean": tf.reduce_mean(shrink.phi),
    }


def report_iteration_progress(shrink: "LuShrinkageEstimator", it) -> dict:
    """
    Print current state values (scalars + cheap aggregates) at end of iteration.
    Returns updated snapshot for next iteration.

    TF-compatible: uses tf.print and returns tensors (no .numpy(), no Python floats).
    """
    it_t = tf.convert_to_tensor(it)

    beta_p = tf.identity(shrink.beta_p)
    beta_w = tf.identity(shrink.beta_w)
    r_val = tf.identity(shrink.r)
    sigma = tf.exp(shrink.r)

    E_bar_norm = tf.norm(shrink.E_bar)
    njt_norm = tf.norm(shrink.njt)

    gamma_mean = tf.reduce_mean(tf.cast(shrink.gamma, tf.float64))
    phi_mean = tf.reduce_mean(shrink.phi)

    tf.print(
        "[LuShrinkage] it=",
        it_t,
        " | beta_p=",
        beta_p,
        ", beta_w=",
        beta_w,
        ", sigma=",
        sigma,
        " | E_bar_norm=",
        E_bar_norm,
        ", njt_norm=",
        njt_norm,
        " | mean(gamma)=",
        gamma_mean,
        ", mean(phi)=",
        phi_mean,
    )

    return {
        "beta_p": beta_p,
        "beta_w": beta_w,
        "r": r_val,
        "E_bar_norm": E_bar_norm,
        "njt_norm": njt_norm,
        "gamma_mean": gamma_mean,
        "phi_mean": phi_mean,
    }


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
        self.sum_gamma.assign_add(tf.cast(shrink.gamma, tf.float64))

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
