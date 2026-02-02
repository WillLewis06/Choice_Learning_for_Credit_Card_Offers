# market_shock_estimators/lu_shrinkage.py
#
# Clean Lu (2025) shrinkage sampler (two-normal spike-and-slab, paper-aligned):
#   - Estimator-style API: fit() + get_results().
#   - Log densities delegated to LuPosteriorTF.
#   - MH mechanics via tmh_step (Laplace independence MH) and rw_mh_step.
#
# Blocking (minimal, Lu-aligned on point 1):
#   - Global: (beta_p, beta_w) via TMH; r via RW-MH.
#   - Market t: E_bar_t via RW-MH;
#               njt_t (full J vector) via TMH;
#               gamma_t via conditional Bernoulli (Gibbs);
#               phi_t via Gibbs.
#
from __future__ import annotations

# tmp debugging
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from market_shock_estimators.lu_posterior import LuPosteriorTF
from market_shock_estimators.lu_shrinkage_diagnostics import LuShrinkageDiagnostics
from market_shock_estimators.lu_shrinkage_updates import (
    update_beta,
    update_r,
    update_E_bar,
    update_njt,
    update_gamma,
    update_phi,
)
from market_shock_estimators.lu_shrinkage_tuning import tune_shrinkage


class LuShrinkageEstimator:
    """
    Lu (2025) shrinkage estimator MCMC sampler (Section 4 simulation target).

    Public API:
      - fit(...)
      - get_results()

    State variables (paper-aligned):
      Global:
        beta_p, beta_w, r
      Market-level:
        E_bar_t, njt[t,j]
      Sparsity/hyper:
        gamma[t,j] in {0,1}, phi[t] in (0,1)

    Data:
      pjt, wjt, qjt, q0t
    """

    def __init__(
        self,
        pjt: np.ndarray,
        wjt: np.ndarray,
        qjt: np.ndarray,
        q0t: np.ndarray,
        n_draws: int,
        seed: int,
    ):
        self.success: bool = False
        self._results: dict | None = None

        # -----------------------------
        # Data
        # -----------------------------
        self.pjt = tf.convert_to_tensor(pjt, dtype=tf.float64)  # (T,J)
        self.wjt = tf.convert_to_tensor(wjt, dtype=tf.float64)  # (T,J)
        self.qjt = tf.convert_to_tensor(qjt, dtype=tf.float64)  # (T,J)
        self.q0t = tf.convert_to_tensor(q0t, dtype=tf.float64)  # (T,)

        if self.pjt.shape.rank != 2:
            raise ValueError("pjt must be rank-2 with shape (T,J).")
        if self.wjt.shape != self.pjt.shape:
            raise ValueError("wjt must have same shape as pjt.")
        if self.qjt.shape != self.pjt.shape:
            raise ValueError("qjt must have same shape as pjt.")
        if self.q0t.shape.rank != 1 or int(self.q0t.shape[0]) != int(self.pjt.shape[0]):
            raise ValueError("q0t must be shape (T,) matching pjt first dimension.")

        self.T = int(self.pjt.shape[0])
        self.J = int(self.pjt.shape[1])

        # -----------------------------
        # Posterior object
        # -----------------------------
        self.posterior = LuPosteriorTF(n_draws=int(n_draws), seed=int(seed))

        # -----------------------------
        # RNG
        # -----------------------------
        self.rng = tf.random.Generator.from_seed(int(seed))

        # -----------------------------
        # Initialize state
        # -----------------------------
        self.beta_p = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.beta_w = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.r = tf.Variable(0.0, dtype=tf.float64, trainable=False)  # log(sigma)

        self.E_bar = tf.Variable(
            tf.fill([self.T], self.posterior.E_bar_mean),
            trainable=False,
        )

        self.njt = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )
        self.gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )

        phi0 = self.posterior.a_phi / (self.posterior.a_phi + self.posterior.b_phi)
        self.phi = tf.Variable(tf.fill([self.T], phi0), trainable=False)

        # set in _run_mcmc_loop (python-owned), used inside compiled iteration step
        self._diag: LuShrinkageDiagnostics | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _debug_save_k(self, k_r, k_E_bar, k_beta, k_njt):
        path = Path("./_debug_cache/lu_shrinkage_k.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                {
                    "k_r": float(k_r.numpy()),
                    "k_E_bar": float(k_E_bar.numpy()),
                    "k_beta": float(k_beta.numpy()),
                    "k_njt": float(k_njt.numpy()),
                }
            )
        )

    def _debug_load_k(self):
        path = Path("./_debug_cache/lu_shrinkage_k.json")
        d = json.loads(path.read_text())

        return (
            tf.constant(d["k_r"], dtype=tf.float64),
            tf.constant(d["k_E_bar"], dtype=tf.float64),
            tf.constant(d["k_beta"], dtype=tf.float64),
            tf.constant(d["k_njt"], dtype=tf.float64),
        )

    def fit(
        self,
        n_iter: int,
        pilot_length: int,
        ridge: float,
        target_low: float,
        target_high: float,
        max_rounds: int,
        factor_rw: float,
        factor_tmh: float,
    ) -> None:
        """
        Run MCMC and store posterior-mean summaries internally.
        """
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")

        if pilot_length <= 0:
            raise ValueError("pilot_length must be positive.")

        # Tuning
        self.pilot_length = pilot_length
        self.ridge = ridge

        # Tuning hyperparameters (owned by orchestration)
        self.target_low = float(target_low)
        self.target_high = float(target_high)
        self.max_rounds = int(max_rounds)
        self.factor_rw = float(factor_rw)
        self.factor_tmh = float(factor_tmh)

        k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned = tune_shrinkage(self)
        # tmp for faster debugging
        self._debug_save_k(k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned)
        # k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned = self._debug_load_k()

        diag = LuShrinkageDiagnostics(T=self.T, J=self.J)

        self._run_mcmc_loop(
            n_iter=n_iter,
            k_beta=k_beta_tuned,
            k_njt=k_njt_tuned,
            k_r=k_r_tuned,
            k_E_bar=k_E_bar_tuned,
            ridge=ridge,
            diag=diag,
        )

        self.success = True

    def get_results(self) -> dict:
        """
        Return the minimal posterior-mean quantities needed by assess_estimator.py.

        Notes
        -----
        - This computes posterior means on-demand from the diagnostics running sums.
        - It assumes `fit()` has been run and `self._diag` is populated.
        """
        saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = (
            self._diag.get_sums()
        )
        saved_f = tf.cast(saved, tf.float64)

        beta_mean = (sum_beta / saved_f).numpy()
        sigma_mean = float((sum_sigma / saved_f).numpy())
        E_bar_mean = (sum_E_bar / saved_f).numpy()
        njt_mean = (sum_njt / saved_f).numpy()
        E_mean = E_bar_mean[:, None] + njt_mean
        phi_mean = (sum_phi / saved_f).numpy()
        gamma_mean = (sum_gamma / saved_f).numpy()

        return {
            "success": True,
            "beta_p_hat": float(beta_mean[0]),
            "beta_w_hat": float(beta_mean[1]),
            "sigma_hat": sigma_mean,
            "E_hat": E_mean,
            "E_bar_hat": E_bar_mean,
            "njt_hat": njt_mean,
            "phi_hat": phi_mean,
            "gamma_hat": gamma_mean,
            "n_saved": int(saved),
        }

    # ------------------------------------------------------------------
    # MCMC orchestration
    # ------------------------------------------------------------------

    def _run_mcmc_loop(
        self,
        n_iter: int,
        k_beta: tf.Tensor,
        k_njt: tf.Tensor,
        k_r: tf.Tensor,
        k_E_bar: tf.Tensor,
        ridge: float,
        diag: LuShrinkageDiagnostics,
    ) -> None:
        """
        Owns the full MCMC loop, mutating sampler state (tf.Variables) and
        accumulating posterior draw sums.
        """
        self._diag = diag  # python-owned handle, used inside compiled step

        ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

        for it in range(n_iter):
            it_t = tf.convert_to_tensor(it, dtype=tf.int32)
            self._mcmc_iteration_step(
                it=it_t,
                k_beta=k_beta,
                k_njt=k_njt,
                k_r=k_r,
                k_E_bar=k_E_bar,
                ridge=ridge_t,
            )

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it, k_beta, k_njt, k_r, k_E_bar, ridge):
        # (beta_p, beta_w)
        beta_p_new, beta_w_new, _ = update_beta(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            pjt=self.pjt,
            wjt=self.wjt,
            beta_p=self.beta_p,
            beta_w=self.beta_w,
            r=self.r,
            E_bar=self.E_bar,
            njt=self.njt,
            k_beta=k_beta,
            ridge=ridge,
        )
        self.beta_p.assign(beta_p_new)
        self.beta_w.assign(beta_w_new)

        # r
        r_new, _ = update_r(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            pjt=self.pjt,
            wjt=self.wjt,
            beta_p=self.beta_p,
            beta_w=self.beta_w,
            r=self.r,
            E_bar=self.E_bar,
            njt=self.njt,
            k_r=k_r,
        )
        self.r.assign(r_new)

        # E_bar (vector)
        E_bar_new, _ = update_E_bar(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            pjt=self.pjt,
            wjt=self.wjt,
            beta_p=self.beta_p,
            beta_w=self.beta_w,
            r=self.r,
            E_bar=self.E_bar,
            njt=self.njt,
            gamma=self.gamma,
            phi=self.phi,
            k_E_bar=k_E_bar,
        )
        self.E_bar.assign(E_bar_new)

        # njt (market sweep)
        njt_new, _ = update_njt(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            pjt=self.pjt,
            wjt=self.wjt,
            beta_p=self.beta_p,
            beta_w=self.beta_w,
            r=self.r,
            E_bar=self.E_bar,
            njt=self.njt,
            gamma=self.gamma,
            phi=self.phi,
            k_njt=k_njt,
            ridge=ridge,
        )
        self.njt.assign(njt_new)

        # gamma
        gamma_new = update_gamma(
            posterior=self.posterior,
            rng=self.rng,
            njt=self.njt,
            phi=self.phi,
        )
        self.gamma.assign(gamma_new)

        # phi
        phi_new = update_phi(
            posterior=self.posterior,
            rng=self.rng,
            gamma=self.gamma,
        )
        self.phi.assign(phi_new)

        # diagnostics
        self._diag.step(self, it)
