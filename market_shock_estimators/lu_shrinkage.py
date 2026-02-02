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
from market_shock_estimators.lu_shrinkage_kernels import (
    tmh_step,
    rw_mh_step,
    gibbs_phi,
    sample_gamma_given_n_phi_market,
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
        self._update_beta(k_beta, ridge)
        self._update_r(k_r)
        self._update_E_bar(k_E_bar)
        self._update_njt(k_njt, ridge)
        self._update_gamma()
        self._update_phi()
        self._diag.step(self, it)

    # ------------------------------------------------------------------
    # Variable updates (All markets batched)
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _update_beta(self, k_beta: tf.Tensor, ridge: tf.Tensor) -> None:
        beta0 = tf.stack([self.beta_p, self.beta_w], axis=0)

        def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
            beta_p = theta_vec[0]
            beta_w = theta_vec[1]
            ll_t = self.posterior.loglik_vec(
                qjt=self.qjt,
                q0t=self.q0t,
                pjt=self.pjt,
                wjt=self.wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=self.r,
                E_bar=self.E_bar,
                njt=self.njt,
            )
            ll = tf.reduce_sum(ll_t)

            # Global prior includes r as well; r is fixed in this update so that term is constant.
            lp = self.posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=self.r)
            return ll + lp

        beta_new, _ = tmh_step(
            theta0=beta0,
            logp_fn=logp_beta,
            ridge=ridge,
            rng=self.rng,
            k=k_beta,
        )
        self.beta_p.assign(beta_new[0])
        self.beta_w.assign(beta_new[1])

    @tf.function(reduce_retracing=True)
    def _update_r(self, k_r: tf.Tensor) -> None:
        def logp_r(r_val: tf.Tensor) -> tf.Tensor:
            ll_t = self.posterior.loglik_vec(
                qjt=self.qjt,
                q0t=self.q0t,
                pjt=self.pjt,
                wjt=self.wjt,
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=r_val,
                E_bar=self.E_bar,
                njt=self.njt,
            )
            ll = tf.reduce_sum(ll_t)

            # Global prior includes beta terms too; beta is fixed here so those terms are constant.
            lp = self.posterior.logprior_global(
                beta_p=self.beta_p, beta_w=self.beta_w, r=r_val
            )
            return ll + lp

        r_new, _, _ = rw_mh_step(
            theta0=self.r,
            logp_fn=logp_r,
            k=k_r,
            rng=self.rng,
        )
        self.r.assign(r_new)

    @tf.function(reduce_retracing=True)
    def _update_E_bar(self, k_E_bar: tf.Tensor) -> None:
        """
        Update E_bar (all markets) via RW-MH, batched across markets.

        Uses rw_mh_step with a per-market log posterior vector, so each market's
        accept/reject is independent (conditional on the global state).
        """

        def logp_E_bar_vec(E_bar_val: tf.Tensor) -> tf.Tensor:
            # Returns (T,) where entry t is market t log posterior contribution.

            return self.posterior.logpost_vec(
                qjt=self.qjt,
                q0t=self.q0t,
                pjt=self.pjt,
                wjt=self.wjt,
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=self.r,
                E_bar=E_bar_val,
                njt=self.njt,
                gamma=self.gamma,
                phi=self.phi,
            )

        E_bar_new, _, _ = rw_mh_step(
            theta0=self.E_bar,
            logp_fn=logp_E_bar_vec,
            k=k_E_bar,
            rng=self.rng,
        )

        self.E_bar.assign(E_bar_new)

    @tf.function(reduce_retracing=True)
    def _update_njt(self, k_njt: tf.Tensor, ridge: tf.Tensor) -> None:
        """
        Update njt for all markets.

        Implementation choice: keep a market loop inside this function (sequential),
        because TMH is per-market and uses a stateful RNG.
        """

        # Snapshot current njt as tensors for loop-carried updates
        njt0 = self.njt.read_value()  # (T,J)
        E_bar0 = self.E_bar.read_value()  # (T,)
        gamma0 = self.gamma.read_value()  # (T,J)
        phi0 = self.phi.read_value()  # (T,)

        T_t = tf.shape(self.pjt)[0]

        ta_n = tf.TensorArray(tf.float64, size=T_t).unstack(njt0)

        def cond(t, ta_n_in):
            return t < T_t

        def body(t, ta_n_in):
            # data slices (market t)
            qjt_t = self.qjt[t]
            q0t_t = self.q0t[t]
            pjt_t = self.pjt[t]
            wjt_t = self.wjt[t]

            # state slices (market t)
            E_bar_t = E_bar0[t]
            njt_t = ta_n_in.read(t)
            gamma_t = gamma0[t]
            phi_t = phi0[t]

            def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
                ll = self.posterior.market_loglik(
                    qjt_t=qjt_t,
                    q0t_t=q0t_t,
                    pjt_t=pjt_t,
                    wjt_t=wjt_t,
                    beta_p=self.beta_p,
                    beta_w=self.beta_w,
                    r=self.r,
                    E_bar_t=E_bar_t,
                    njt_t=njt_t_val,
                )
                lp_1 = self.posterior.logprior_market_vec(
                    E_bar=tf.reshape(E_bar_t, (1,)),
                    njt=tf.expand_dims(njt_t_val, axis=0),
                    gamma=tf.expand_dims(gamma_t, axis=0),
                    phi=tf.reshape(phi_t, (1,)),
                )
                return ll + lp_1[0]

            njt_new, _ = tmh_step(
                theta0=njt_t,
                logp_fn=logp_njt_t,
                ridge=ridge,
                rng=self.rng,
                k=k_njt,
            )

            ta_n_out = ta_n_in.write(t, njt_new)
            return t + 1, ta_n_out

        t0 = tf.constant(0, dtype=tf.int32)
        _, ta_n = tf.while_loop(
            cond,
            body,
            loop_vars=(t0, ta_n),
            parallel_iterations=1,
        )

        # Commit once
        self.njt.assign(ta_n.stack())

    @tf.function(reduce_retracing=True)
    def _update_gamma(self) -> None:
        """
        Update gamma for all markets given current njt and phi.

        Vectorized Gibbs step across (T,J).
        """
        gamma_new = sample_gamma_given_n_phi_market(
            njt_t=self.njt,  # (T,J)
            phi_t=self.phi[:, None],  # broadcast to (T,1) -> (T,J)
            T0_sq=self.posterior.T0_sq,
            T1_sq=self.posterior.T1_sq,
            log_T0_sq=self.posterior.log_T0_sq,
            log_T1_sq=self.posterior.log_T1_sq,
            rng=self.rng,
        )

        self.gamma.assign(gamma_new)

    @tf.function(reduce_retracing=True)
    def _update_phi(self) -> None:
        """
        Update phi for all markets given current gamma via batched Gibbs.

        phi_t | gamma_t ~ Beta(a_phi + sum_j gamma_tj, b_phi + J - sum_j gamma_tj)
        """
        phi_new = gibbs_phi(
            gamma=self.gamma,  # (T,J)
            a_phi=self.posterior.a_phi,  # scalar
            b_phi=self.posterior.b_phi,  # scalar
            rng=self.rng,
        )
        self.phi.assign(phi_new)
