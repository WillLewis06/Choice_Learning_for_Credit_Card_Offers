"""
Lu shrinkage estimator (MCMC sampler).

This module defines `LuShrinkageEstimator`, which:
- stores observed market data (pjt, wjt, qjt, q0t),
- owns the current MCMC state (tf.Variables),
- tunes proposal scales once using a pilot run,
- runs an MCMC loop and accumulates posterior means via diagnostics.

All computation is performed in tf.float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_diagnostics import LuShrinkageDiagnostics
from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF
from lu.shrinkage.lu_tuning import tune_shrinkage
from lu.shrinkage.lu_updates import (
    update_E_bar,
    update_beta,
    update_gamma,
    update_njt,
    update_phi,
    update_r,
)
from lu.shrinkage.lu_validate_input import fit_validate_input, init_validate_input


@dataclass(frozen=True)
class LuShrinkageFitConfig:
    """Controls for proposal tuning and sampling.

    There is no burn-in or thinning: every iteration is accumulated.
    """

    n_iter: int
    pilot_length: int
    ridge: float
    target_low: float
    target_high: float
    max_rounds: int
    factor_rw: float
    factor_tmh: float


class LuShrinkageEstimator:
    """MCMC sampler for the Lu shrinkage model.

    Observed data (fixed):
      - pjt, wjt, qjt, q0t

    Latent state (sampled):
      - Global: beta_p, beta_w, r
      - Market-level: E_bar[t]
      - Market-product: njt[t, j]
      - Sparsity: gamma[t, j] in {0,1}, phi[t] in (0,1)
    """

    def __init__(
        self,
        pjt: np.ndarray,
        wjt: np.ndarray,
        qjt: np.ndarray,
        q0t: np.ndarray,
        posterior_config: LuPosteriorConfig,
    ):
        """Construct the estimator and initialize sampler state.

        Args:
            pjt, wjt: Observed product attributes, shape (T, J).
            qjt: Observed inside-good counts, shape (T, J).
            q0t: Observed outside-good counts, shape (T,).
            posterior_config: Fully-specified posterior configuration (no defaults).
        """
        if posterior_config.dtype != tf.float64:
            raise ValueError("posterior_config.dtype must be tf.float64.")

        # -----------------------------
        # Observed market data (fixed)
        # -----------------------------
        self.pjt = tf.convert_to_tensor(pjt, dtype=tf.float64)  # (T, J)
        self.wjt = tf.convert_to_tensor(wjt, dtype=tf.float64)  # (T, J)
        self.qjt = tf.convert_to_tensor(qjt, dtype=tf.float64)  # (T, J)
        self.q0t = tf.convert_to_tensor(q0t, dtype=tf.float64)  # (T,)

        init_validate_input(
            pjt=self.pjt,
            wjt=self.wjt,
            qjt=self.qjt,
            q0t=self.q0t,
            n_draws=posterior_config.n_draws,
            seed=posterior_config.seed,
        )

        self.T = int(self.pjt.shape[0])
        self.J = int(self.pjt.shape[1])

        # -----------------------------
        # Posterior helper (likelihood + priors)
        # -----------------------------
        self.posterior = LuPosteriorTF(config=posterior_config)

        # -----------------------------
        # RNG owned by the sampler
        # -----------------------------
        self.rng = tf.random.Generator.from_seed(posterior_config.seed)

        # -----------------------------
        # Latent state (tf.Variables mutated in-place)
        # -----------------------------
        self.beta_p = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.beta_w = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.r = tf.Variable(0.0, dtype=tf.float64, trainable=False)  # log(sigma)

        # E_bar initialized at its prior mean.
        self.E_bar = tf.Variable(
            tf.fill([self.T], self.posterior.E_bar_mean),
            dtype=tf.float64,
            trainable=False,
        )

        # njt and gamma initialized at zero.
        self.njt = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64),
            trainable=False,
        )
        self.gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64),
            trainable=False,
        )

        # phi initialized at the Beta prior mean a/(a+b).
        phi0 = self.posterior.a_phi / (self.posterior.a_phi + self.posterior.b_phi)
        self.phi = tf.Variable(
            tf.fill([self.T], phi0), dtype=tf.float64, trainable=False
        )

        # Diagnostics handle is created at fit-time.
        self._diag: Optional[LuShrinkageDiagnostics] = None

        # Fit controls are stored as a single config object (used by tuning).
        self._fit_config: Optional[LuShrinkageFitConfig] = None

    # ------------------------------------------------------------------
    # Tuning controls (read by tune_shrinkage)
    # ------------------------------------------------------------------

    def _require_fit_config(self) -> LuShrinkageFitConfig:
        if self._fit_config is None:
            raise ValueError(
                "fit() must be called before proposal tuning is available."
            )
        return self._fit_config

    @property
    def pilot_length(self) -> int:
        return self._require_fit_config().pilot_length

    @property
    def ridge(self) -> float:
        return self._require_fit_config().ridge

    @property
    def target_low(self) -> float:
        return self._require_fit_config().target_low

    @property
    def target_high(self) -> float:
        return self._require_fit_config().target_high

    @property
    def max_rounds(self) -> int:
        return self._require_fit_config().max_rounds

    @property
    def factor_rw(self) -> float:
        return self._require_fit_config().factor_rw

    @property
    def factor_tmh(self) -> float:
        return self._require_fit_config().factor_tmh

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, config: LuShrinkageFitConfig) -> None:
        """Tune proposal scales, run MCMC, and accumulate posterior means."""
        fit_validate_input(
            n_iter=config.n_iter,
            pilot_length=config.pilot_length,
            ridge=config.ridge,
            target_low=config.target_low,
            target_high=config.target_high,
            max_rounds=config.max_rounds,
            factor_rw=config.factor_rw,
            factor_tmh=config.factor_tmh,
        )
        self._fit_config = config

        # Tune proposal scales (k_*) using a short pilot run.
        k_r, k_E_bar, k_beta, k_njt = tune_shrinkage(self)

        # Accumulate posterior means over all iterations.
        diag = LuShrinkageDiagnostics(T=self.T, J=self.J)
        self._run_mcmc_loop(
            n_iter=config.n_iter,
            k_beta=k_beta,
            k_njt=k_njt,
            k_r=k_r,
            k_E_bar=k_E_bar,
            ridge=config.ridge,
            diag=diag,
        )

    def get_results(self) -> dict:
        """Return posterior-mean summaries from accumulated running sums."""
        if self._diag is None:
            raise ValueError("get_results() called before fit().")

        saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = (
            self._diag.get_sums()
        )
        saved_f = tf.cast(saved, tf.float64)

        beta_mean = (sum_beta / saved_f).numpy()  # (2,)
        sigma_mean = float((sum_sigma / saved_f).numpy())

        E_bar_mean = (sum_E_bar / saved_f).numpy()  # (T,)
        njt_mean = (sum_njt / saved_f).numpy()  # (T, J)
        E_mean = E_bar_mean[:, None] + njt_mean  # (T, J)

        phi_mean = (sum_phi / saved_f).numpy()  # (T,)
        gamma_mean = (sum_gamma / saved_f).numpy()  # (T, J)

        return {
            "beta_p_hat": float(beta_mean[0]),
            "beta_w_hat": float(beta_mean[1]),
            "sigma_hat": sigma_mean,
            "E_hat": E_mean,
            "E_bar_hat": E_bar_mean,
            "njt_hat": njt_mean,
            "phi_hat": phi_mean,
            "gamma_hat": gamma_mean,
            "n_saved": int(saved.numpy()),
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
        """Run the iteration loop and mutate sampler state."""
        self._diag = diag
        ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

        for it in range(n_iter):
            self._mcmc_iteration_step(
                it=tf.convert_to_tensor(it, dtype=tf.int32),
                k_beta=k_beta,
                k_njt=k_njt,
                k_r=k_r,
                k_E_bar=k_E_bar,
                ridge=ridge_t,
            )

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it, k_beta, k_njt, k_r, k_E_bar, ridge):
        """Run one MCMC iteration and record diagnostics."""
        # (beta_p, beta_w): TMH.
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

        # r: RW-MH.
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

        # E_bar: elementwise RW-MH.
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

        # njt: TMH sweep across markets.
        njt_new, _accepted_count = update_njt(
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

        # gamma: Gibbs.
        gamma_new = update_gamma(
            posterior=self.posterior,
            rng=self.rng,
            njt=self.njt,
            phi=self.phi,
        )
        self.gamma.assign(gamma_new)

        # phi: Gibbs.
        phi_new = update_phi(
            posterior=self.posterior,
            rng=self.rng,
            gamma=self.gamma,
        )
        self.phi.assign(phi_new)

        # Running sums + progress line.
        self._diag.step(self, it)
