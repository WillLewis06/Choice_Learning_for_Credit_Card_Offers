from __future__ import annotations

import tensorflow as tf


def init_progress_state(shrink: LuShrinkageEstimator) -> dict:
    """
    Initialize a lightweight snapshot of the current state.
    Scalars + cheap aggregates only.
    """
    return {
        "beta_p": float(shrink.beta_p.numpy()),
        "beta_w": float(shrink.beta_w.numpy()),
        "r": float(shrink.r.numpy()),
        "E_bar_norm": float(tf.norm(shrink.E_bar).numpy()),
        "njt_norm": float(tf.norm(shrink.njt).numpy()),
        "gamma_mean": float(tf.reduce_mean(tf.cast(shrink.gamma, tf.float64)).numpy()),
        "phi_mean": float(tf.reduce_mean(shrink.phi).numpy()),
    }


def report_iteration_progress(shrink: LuShrinkageEstimator, it: int) -> dict:
    """
    Print current state values (scalars + cheap aggregates) at end of iteration.
    Returns updated snapshot for next iteration.
    """
    beta_p = float(shrink.beta_p.numpy())
    beta_w = float(shrink.beta_w.numpy())
    r_val = float(shrink.r.numpy())
    sigma = float(tf.exp(shrink.r).numpy())

    E_bar_norm = float(tf.norm(shrink.E_bar).numpy())
    njt_norm = float(tf.norm(shrink.njt).numpy())

    gamma_mean = float(tf.reduce_mean(tf.cast(shrink.gamma, tf.float64)).numpy())
    phi_mean = float(tf.reduce_mean(shrink.phi).numpy())

    print(
        f"[LuShrinkage] it={it} | "
        f"beta_p={beta_p:.4f}, beta_w={beta_w:.4f}, sigma={sigma:.4f} | "
        f"E_bar_norm={E_bar_norm:.4e}, njt_norm={njt_norm:.4e} | "
        f"mean(gamma)={gamma_mean:.4f}, mean(phi)={phi_mean:.4f}"
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
      - progress printing state (prev snapshot)
      - running-sum accumulation for posterior-mean summaries

    Intended call pattern from LuShrinkageEstimator:
      diag = LuShrinkageDiagnostics(T, J)
      diag.start(self)
      for it in range(n_iter):
          ... update blocks ...
          diag.step(self, it)
      saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = diag.get_sums()
    """

    def __init__(self, T: int, J: int):
        self.T = int(T)
        self.J = int(J)

        self.saved: int = 0
        self.sum_beta = tf.zeros([2], dtype=tf.float64)
        self.sum_sigma = tf.constant(0.0, dtype=tf.float64)
        self.sum_E_bar = tf.zeros([self.T], dtype=tf.float64)
        self.sum_njt = tf.zeros([self.T, self.J], dtype=tf.float64)
        self.sum_phi = tf.zeros([self.T], dtype=tf.float64)
        self.sum_gamma = tf.zeros([self.T, self.J], dtype=tf.float64)

    def _accumulate_draw(self, shrink: LuShrinkageEstimator) -> None:
        """
        Accumulate the current state into running sums.
        No burn-in/thinning logic; every iteration is retained.
        """
        self.saved += 1
        self.sum_beta += tf.stack([shrink.beta_p, shrink.beta_w], axis=0)
        self.sum_sigma += tf.exp(shrink.r)
        self.sum_E_bar += shrink.E_bar
        self.sum_njt += shrink.njt
        self.sum_phi += shrink.phi
        self.sum_gamma += tf.cast(shrink.gamma, tf.float64)

    def step(self, shrink: LuShrinkageEstimator, it: int) -> None:
        """
        Called once per iteration:
          - accumulate current draw into running sums
          - print progress line
        """
        self._accumulate_draw(shrink)
        report_iteration_progress(shrink, it)

    def get_sums(
        self,
    ) -> tuple[int, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Return the raw running sums for the estimator to finalize results.
        """
        return (
            int(self.saved),
            self.sum_beta,
            self.sum_sigma,
            self.sum_E_bar,
            self.sum_njt,
            self.sum_phi,
            self.sum_gamma,
        )
