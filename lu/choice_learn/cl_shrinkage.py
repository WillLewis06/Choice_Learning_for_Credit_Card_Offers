"""
Choice-learn + Lu shrinkage estimator (MCMC sampler).

Samples sparse market-product shocks under a spike-and-slab prior, conditional on
fixed choice-learn mean utilities.

Model (systematic utility):
  delta[t, j] = alpha * delta_cl[t, j] + E_bar[t] + njt[t, j]
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import tensorflow as tf

from lu.choice_learn.cl_diagnostics import ChoiceLearnShrinkageDiagnostics
from lu.choice_learn.cl_posterior import LuPosteriorTF
from lu.choice_learn.cl_tuning import tune_shrinkage
from lu.choice_learn.cl_updates import (
    update_E_bar,
    update_alpha,
    update_gamma,
    update_njt,
    update_phi,
)
from lu.choice_learn.cl_validate_input import (
    validate_data_inputs,
    validate_fit_config,
    validate_init_config,
)


class ChoiceLearnShrinkageEstimator:
    """MCMC sampler for the choice-learn + Lu sparse-shock model.

    Fixed data:
      - delta_cl: (T, J) float64 baseline mean utilities from choice-learn
      - qjt:      (T, J) float64 inside counts/shares
      - q0t:      (T,)   float64 outside counts/shares

    Sampled state:
      - alpha: scalar
      - E_bar: (T,)
      - njt:   (T, J)
      - gamma: (T, J) in {0,1} (stored as float64)
      - phi:   (T,) in (0,1)
    """

    def __init__(
        self,
        delta_cl: tf.Tensor,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        config: Mapping[str, object],
    ):
        """Validate inputs, build posterior helper, and initialize sampler state.

        Required init config keys:
          - seed: int
          - posterior: mapping of LuPosteriorTF hyperparameters
          - init_state: mapping with keys {alpha, E_bar, njt, gamma, phi}
        """
        # Boundary validation for data tensors and init config.
        T, J = validate_data_inputs(delta_cl=delta_cl, qjt=qjt, q0t=q0t)
        validate_init_config(config=config, T=T, J=J)

        self.T = T
        self.J = J

        self.delta_cl = delta_cl
        self.qjt = qjt
        self.q0t = q0t

        seed = config["seed"]
        posterior_cfg = config["posterior"]
        init_state = config["init_state"]

        # Posterior helper (likelihood + priors), fully specified by config.
        self.posterior = LuPosteriorTF(posterior_cfg)

        # RNG owned by the sampler (used by the main chain only).
        self.rng = tf.random.Generator.from_seed(seed)

        # Latent state (explicitly initialized from config; no fallbacks).
        self.alpha = tf.Variable(init_state["alpha"], dtype=tf.float64, trainable=False)
        self.E_bar = tf.Variable(init_state["E_bar"], dtype=tf.float64, trainable=False)
        self.njt = tf.Variable(init_state["njt"], dtype=tf.float64, trainable=False)
        self.gamma = tf.Variable(init_state["gamma"], dtype=tf.float64, trainable=False)
        self.phi = tf.Variable(init_state["phi"], dtype=tf.float64, trainable=False)

        self._diag: ChoiceLearnShrinkageDiagnostics | None = None

    def fit(self, fit_config: Mapping[str, object]) -> None:
        """Tune proposal scales, run MCMC, and accumulate posterior means."""
        validate_fit_config(fit_config)

        n_iter = fit_config["n_iter"]

        pilot_length = fit_config["pilot_length"]
        ridge = fit_config["ridge"]
        target_low = fit_config["target_low"]
        target_high = fit_config["target_high"]
        max_rounds = fit_config["max_rounds"]
        factor_rw = fit_config["factor_rw"]
        factor_tmh = fit_config["factor_tmh"]

        k_alpha0 = tf.constant(fit_config["k_alpha0"], tf.float64)
        k_E_bar0 = tf.constant(fit_config["k_E_bar0"], tf.float64)
        k_njt0 = tf.constant(fit_config["k_njt0"], tf.float64)
        tune_seed = fit_config["tune_seed"]

        ridge_t = tf.constant(ridge, tf.float64)

        # Build the minimal view object required by tune_shrinkage().
        tune_view = SimpleNamespace(
            pilot_length=pilot_length,
            ridge=ridge_t,
            target_low=target_low,
            target_high=target_high,
            max_rounds=max_rounds,
            factor_rw=factor_rw,
            factor_tmh=factor_tmh,
            k_alpha0=k_alpha0,
            k_E_bar0=k_E_bar0,
            k_njt0=k_njt0,
            tune_seed=tune_seed,
            T=self.T,
            J=self.J,
            qjt=self.qjt,
            q0t=self.q0t,
            delta_cl=self.delta_cl,
            alpha=self.alpha,
            E_bar=self.E_bar,
            njt=self.njt,
            gamma=self.gamma,
            phi=self.phi,
            posterior=self.posterior,
        )

        k_alpha, k_E_bar, k_njt = tune_shrinkage(tune_view)

        diag = ChoiceLearnShrinkageDiagnostics(T=self.T, J=self.J)
        self._run_mcmc_loop(
            n_iter=n_iter,
            k_alpha=k_alpha,
            k_E_bar=k_E_bar,
            k_njt=k_njt,
            ridge=ridge_t,
            diag=diag,
        )

    def get_results(self) -> dict:
        """Return posterior-mean summaries from accumulated running sums."""
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
            "n_saved": int(saved.numpy()),
        }

    def _run_mcmc_loop(
        self,
        n_iter: int,
        k_alpha: tf.Tensor,
        k_E_bar: tf.Tensor,
        k_njt: tf.Tensor,
        ridge: tf.Tensor,
        diag: ChoiceLearnShrinkageDiagnostics,
    ) -> None:
        """Run the iteration loop, mutate sampler state, and record diagnostics."""
        self._diag = diag

        for it in range(n_iter):
            it_t = tf.convert_to_tensor(it, dtype=tf.int32)

            # State update (compiled).
            self._mcmc_iteration_step(
                k_alpha=k_alpha,
                k_E_bar=k_E_bar,
                k_njt=k_njt,
                ridge=ridge,
            )

            # Diagnostics: accumulate and print a progress line (compiled).
            diag.step(self, it_t)

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, k_alpha, k_E_bar, k_njt, ridge):
        """Run one full MCMC iteration (state updates only)."""
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

        gamma_new = update_gamma(
            posterior=self.posterior,
            rng=self.rng,
            njt=self.njt,
            phi=self.phi,
        )
        self.gamma.assign(gamma_new)

        phi_new = update_phi(
            posterior=self.posterior,
            rng=self.rng,
            gamma=self.gamma,
        )
        self.phi.assign(phi_new)
