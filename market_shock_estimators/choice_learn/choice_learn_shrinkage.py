"""
Choice-learn shrinkage estimator (MCMC sampler).

This module defines `ChoiceLearnShrinkageEstimator`, a self-contained sampler that:
  - holds observed market data (delta_cl, qjt, q0t),
  - owns the current MCMC state (tf.Variables),
  - tunes proposal scales once using a pilot run,
  - runs an MCMC loop and accumulates posterior means via diagnostics.

Model (systematic utility):
  delta_{t,j} = alpha * delta_cl_{t,j} + E_bar[t] + njt[t,j]

Heavy lifting for parameter-block updates lives in `choice_learn_updates.py`.

Design note (aligned with the original Lu implementation):
  - The compiled step `_mcmc_iteration_step` only updates state.
  - Diagnostics accumulation/printing is called outside the compiled step
    from the Python-owned loop in `_run_mcmc_loop`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from market_shock_estimators.choice_learn.choice_learn_diagnostics import (
    ChoiceLearnShrinkageDiagnostics,
)
from market_shock_estimators.choice_learn.choice_learn_posterior import LuPosteriorTF
from market_shock_estimators.choice_learn.choice_learn_tuning import tune_shrinkage
from market_shock_estimators.choice_learn.choice_learn_updates import (
    update_E_bar,
    update_alpha,
    update_gamma,
    update_njt,
    update_phi,
)
from market_shock_estimators.choice_learn.choice_learn_validate_input import (
    fit_validate_input,
    init_validate_input,
)


class ChoiceLearnShrinkageEstimator:
    """MCMC sampler for the choice-learn + Lu sparse-shock model.

    Observed data (fixed):
      - delta_cl, qjt, q0t

    Latent structure (sampled):
      - Global: alpha
      - Market-level: E_bar[t]
      - Market-product: njt[t, j]
      - Sparsity: gamma[t, j] in {0,1}, phi[t] in (0,1)
    """

    def __init__(
        self,
        delta_cl: np.ndarray,
        qjt: np.ndarray,
        q0t: np.ndarray,
        seed: int,
    ):
        """Construct the estimator and initialize state."""
        # -----------------------------
        # Observed market data (fixed)
        # -----------------------------
        self.delta_cl = tf.convert_to_tensor(delta_cl, dtype=tf.float64)  # (T,J)
        self.qjt = tf.convert_to_tensor(qjt, dtype=tf.float64)  # (T,J)
        self.q0t = tf.convert_to_tensor(q0t, dtype=tf.float64)  # (T,)

        init_validate_input(delta_cl=self.delta_cl, qjt=self.qjt, q0t=self.q0t)

        self.T = int(self.delta_cl.shape[0])
        self.J = int(self.delta_cl.shape[1])

        # -----------------------------
        # Posterior helper (likelihood + priors)
        # -----------------------------
        self.posterior = LuPosteriorTF()

        # -----------------------------
        # RNG owned by the sampler
        # -----------------------------
        self.rng = tf.random.Generator.from_seed(int(seed))

        # -----------------------------
        # Latent state (tf.Variables mutated in-place)
        # -----------------------------
        self.alpha = tf.Variable(1.0, dtype=tf.float64, trainable=False)

        # Initialize E_bar at its prior mean (broadcast across markets).
        self.E_bar = tf.Variable(
            tf.fill([self.T], self.posterior.E_bar_mean),
            trainable=False,
        )

        # Initialize shocks and indicators at zero.
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

        # Diagnostics handle is created at fit-time for accumulation and results.
        self._diag: ChoiceLearnShrinkageDiagnostics | None = None

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

        # Tune proposal scales (k_*).
        k_alpha_tuned, k_E_bar_tuned, k_njt_tuned = tune_shrinkage(self)

        # Diagnostics accumulates running sums used to compute posterior means.
        diag = ChoiceLearnShrinkageDiagnostics(T=self.T, J=self.J)

        self._run_mcmc_loop(
            n_iter=n_iter,
            k_alpha=k_alpha_tuned,
            k_E_bar=k_E_bar_tuned,
            k_njt=k_njt_tuned,
            ridge=ridge,
            diag=diag,
        )

    def get_results(self) -> dict:
        """Return posterior-mean summaries from the accumulated running sums."""
        if self._diag is None:
            raise ValueError("get_results() called before fit().")

        saved, sum_alpha, sum_E_bar, sum_njt, sum_phi, sum_gamma = self._diag.get_sums()
        saved_f = tf.cast(saved, tf.float64)

        alpha_mean = float((sum_alpha / saved_f).numpy())

        E_bar_mean = (sum_E_bar / saved_f).numpy()  # (T,)
        njt_mean = (sum_njt / saved_f).numpy()  # (T,J)
        E_mean = E_bar_mean[:, None] + njt_mean  # (T,J)

        phi_mean = (sum_phi / saved_f).numpy()  # (T,)
        gamma_mean = (sum_gamma / saved_f).numpy()  # (T,J)

        return {
            "alpha_hat": alpha_mean,
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
        k_alpha: tf.Tensor,
        k_E_bar: tf.Tensor,
        k_njt: tf.Tensor,
        ridge: float,
        diag: ChoiceLearnShrinkageDiagnostics,
    ) -> None:
        """Run the Python-owned iteration loop, mutate sampler state, and record diagnostics."""
        self._diag = diag
        ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

        for it in range(n_iter):
            it_t = tf.convert_to_tensor(it, dtype=tf.int32)

            # State update (compiled)
            self._mcmc_iteration_step(
                it=it_t,
                k_alpha=k_alpha,
                k_E_bar=k_E_bar,
                k_njt=k_njt,
                ridge=ridge_t,
            )

            # Diagnostics (outside compiled state update; aligned with original Lu pattern)
            diag.step(self, it_t)

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it, k_alpha, k_E_bar, k_njt, ridge):
        """Run one full MCMC iteration (state updates only).

        Update order:
          1) alpha             RW-MH
          2) E_bar             elementwise RW-MH across markets
          3) njt               TMH sweep over markets (J-dimensional per market)
          4) gamma             Gibbs
          5) phi               Gibbs
        """
        # -----------------------------
        # Global scaling: alpha
        # -----------------------------
        alpha_new, _ = update_alpha(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            delta_cl=self.delta_cl,
            alpha=self.alpha,
            E_bar=self.E_bar,
            njt=self.njt,
            k_alpha=k_alpha,
        )
        self.alpha.assign(alpha_new)

        # -----------------------------
        # Market shock: E_bar (vector over markets)
        # -----------------------------
        E_bar_new, _ = update_E_bar(
            posterior=self.posterior,
            rng=self.rng,
            qjt=self.qjt,
            q0t=self.q0t,
            delta_cl=self.delta_cl,
            alpha=self.alpha,
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
            delta_cl=self.delta_cl,
            alpha=self.alpha,
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
