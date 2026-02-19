"""
Lu shrinkage estimator (MCMC sampler).

This module defines `LuShrinkageEstimator`, a self-contained sampler that:
  - holds the observed market data (pjt, wjt, qjt, q0t),
  - owns the current MCMC state (tf.Variables),
  - tunes proposal scales once using a pilot run,
  - runs an MCMC loop and accumulates posterior means via diagnostics.

The heavy-lifting for each parameter-block update lives in `lu_updates.py`.
This file focuses on orchestration and a minimal public API:
  - fit(...)
  - get_results()
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_diagnostics import LuShrinkageDiagnostics
from lu.shrinkage.lu_posterior import LuPosteriorTF
from lu.shrinkage.lu_tuning import tune_shrinkage
from lu.shrinkage.lu_updates import (
    update_E_bar,
    update_beta,
    update_gamma,
    update_njt,
    update_phi,
    update_r,
)
from lu.shrinkage.lu_validate_input import (
    fit_validate_input,
    init_validate_input,
)


class LuShrinkageEstimator:
    """MCMC sampler for the Lu shrinkage model (simulation target).

    Observed data (fixed):
      - pjt, wjt, qjt, q0t

    Latent structure (sampled):
      - Global: beta_p, beta_w, r
      - Market-level: E_bar[t]
      - Market-product: njt[t, j]
      - Sparsity: gamma[t, j] in {0,1}, phi[t] in (0,1)

    The sampler updates these blocks in a fixed order each iteration and uses
    `LuShrinkageDiagnostics` to accumulate running sums for posterior means.
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
        """Construct the estimator and initialize state.

        This converts input arrays to TF tensors, validates shapes/dtypes, creates
        the `LuPosteriorTF` helper, and initializes all latent state variables.
        """
        # -----------------------------
        # Observed market data (fixed)
        # -----------------------------
        self.pjt = tf.convert_to_tensor(pjt, dtype=tf.float64)  # (T,J)
        self.wjt = tf.convert_to_tensor(wjt, dtype=tf.float64)  # (T,J)
        self.qjt = tf.convert_to_tensor(qjt, dtype=tf.float64)  # (T,J)
        self.q0t = tf.convert_to_tensor(q0t, dtype=tf.float64)  # (T,)

        init_validate_input(
            pjt=self.pjt,
            wjt=self.wjt,
            qjt=self.qjt,
            q0t=self.q0t,
            n_draws=int(n_draws),
            seed=int(seed),
        )

        self.T = int(self.pjt.shape[0])
        self.J = int(self.pjt.shape[1])

        # -----------------------------
        # Posterior helper (likelihood + priors)
        # -----------------------------
        self.posterior = LuPosteriorTF(n_draws=int(n_draws), seed=int(seed))

        # -----------------------------
        # RNG owned by the sampler
        # -----------------------------
        self.rng = tf.random.Generator.from_seed(int(seed))

        # -----------------------------
        # Latent state (tf.Variables mutated in-place)
        # -----------------------------
        self.beta_p = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.beta_w = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.r = tf.Variable(0.0, dtype=tf.float64, trainable=False)  # log(sigma)

        # Initialize E_bar at its prior mean (broadcast across markets).
        self.E_bar = tf.Variable(
            tf.fill([self.T], self.posterior.E_bar_mean),
            trainable=False,
        )

        # Initialize spikes/slab shocks and indicators at zero.
        self.njt = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64),
            trainable=False,
        )
        self.gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64),
            trainable=False,
        )

        # Initialize phi to the Beta prior mean a/(a+b), broadcast across markets.
        phi0 = self.posterior.a_phi / (self.posterior.a_phi + self.posterior.b_phi)
        self.phi = tf.Variable(tf.fill([self.T], phi0), trainable=False)

        # Diagnostics handle is created at fit-time and is used inside the compiled step.
        self._diag: LuShrinkageDiagnostics | None = None

    # ------------------------------------------------------------------
    # Optional local cache for tuned proposal scales
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """Tune proposal scales, run MCMC, and accumulate posterior means."""
        fit_validate_input(
            n_iter=n_iter,
            pilot_length=pilot_length,
            ridge=ridge,
            target_low=target_low,
            target_high=target_high,
            max_rounds=max_rounds,
            factor_rw=factor_rw,
            factor_tmh=factor_tmh,
        )

        # Store tuning configuration on the instance so `tune_shrinkage(self)` can
        # read a consistent set of settings.
        self.pilot_length = pilot_length
        self.ridge = ridge
        self.target_low = target_low
        self.target_high = target_high
        self.max_rounds = max_rounds
        self.factor_rw = factor_rw
        self.factor_tmh = factor_tmh

        # Tune each proposal scale (k_*) using a short pilot run.
        k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned = tune_shrinkage(self)

        # Diagnostics accumulates running sums used to compute posterior means.
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

    def get_results(self) -> dict:
        """Return posterior-mean summaries from the accumulated running sums.

        The diagnostics object stores running sums for each parameter block.
        This method converts those into posterior means and returns the minimal
        result dictionary used by downstream evaluation code.
        """
        if self._diag is None:
            raise ValueError("get_results() called before fit().")

        saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = (
            self._diag.get_sums()
        )
        saved_f = tf.cast(saved, tf.float64)

        # Global parameters
        beta_mean = (sum_beta / saved_f).numpy()
        sigma_mean = float((sum_sigma / saved_f).numpy())

        # Latent shocks
        E_bar_mean = (sum_E_bar / saved_f).numpy()  # (T,)
        njt_mean = (sum_njt / saved_f).numpy()  # (T,J)
        E_mean = E_bar_mean[:, None] + njt_mean  # (T,J)

        # Sparsity indicators and inclusion rates
        phi_mean = (sum_phi / saved_f).numpy()
        gamma_mean = (sum_gamma / saved_f).numpy()

        return {
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
        """Run the Python-owned iteration loop and mutate sampler state.

        The outer loop is kept in Python for simplicity. Each iteration calls a
        compiled step function that updates the TF variables in-place.
        """
        self._diag = diag

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
        """Run one full MCMC iteration and record diagnostics.

        Update order:
          1) (beta_p, beta_w)   TMH
          2) r                 RW-MH
          3) E_bar             elementwise RW-MH across markets
          4) njt               TMH sweep over markets (J-dimensional per market)
          5) gamma             Gibbs
          6) phi               Gibbs

        The final line calls diagnostics to accumulate this draw and report
        progress.
        """
        # -----------------------------
        # Global coefficients: (beta_p, beta_w)
        # -----------------------------
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

        # -----------------------------
        # log(sigma): r
        # -----------------------------
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

        # -----------------------------
        # Market shock: E_bar (vector over markets)
        # -----------------------------
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

        # -----------------------------
        # Market-product shocks: njt (market-by-market TMH sweep)
        # -----------------------------
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

        # -----------------------------
        # Sparsity indicators: gamma
        # -----------------------------
        gamma_new = update_gamma(
            posterior=self.posterior,
            rng=self.rng,
            njt=self.njt,
            phi=self.phi,
        )
        self.gamma.assign(gamma_new)

        # -----------------------------
        # Inclusion rates: phi
        # -----------------------------
        phi_new = update_phi(
            posterior=self.posterior,
            rng=self.rng,
            gamma=self.gamma,
        )
        self.phi.assign(phi_new)

        # -----------------------------
        # Accumulate posterior means and print progress
        # -----------------------------
        self._diag.step(self, it)
