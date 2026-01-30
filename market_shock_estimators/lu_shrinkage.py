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
    gibbs_phi_market,
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
            tf.fill([self.T], tf.cast(self.posterior.E_bar_mean, tf.float64)),
            trainable=False,
        )

        self.njt = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )
        self.gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.float64), trainable=False
        )

        phi0 = tf.cast(self.posterior.a_phi, tf.float64) / tf.cast(
            (self.posterior.a_phi + self.posterior.b_phi), tf.float64
        )
        self.phi = tf.Variable(tf.fill([self.T], phi0), trainable=False)

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
        max_lbfgs_iters: int,
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

        ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

        # Tuning
        self.pilot_length = pilot_length
        self.ridge = ridge
        self.max_lbfgs_iters = max_lbfgs_iters

        # Tuning hyperparameters (owned by orchestration)
        self.target_low = float(target_low)
        self.target_high = float(target_high)
        self.max_rounds = int(max_rounds)
        self.factor_rw = float(factor_rw)
        self.factor_tmh = float(factor_tmh)

        # k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned = tune_shrinkage(self)
        # tmp for faster debugging
        # self._debug_save_k(k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned)
        k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned = self._debug_load_k()

        # One-time warm start for TMH njt block (prevents njt from sticking at 0)
        self._initialize_njt_modes(ridge=ridge, max_lbfgs_iters=max_lbfgs_iters)

        diag = LuShrinkageDiagnostics(T=self.T, J=self.J)

        self._run_mcmc_loop(
            n_iter=n_iter,
            k_beta=k_beta_tuned,
            k_njt=k_njt_tuned,
            k_r=k_r_tuned,
            k_E_bar=k_E_bar_tuned,
            ridge=ridge,
            max_lbfgs_iters=max_lbfgs_iters,
            diag=diag,
        )

        saved, sum_beta, sum_sigma, sum_E_bar, sum_njt, sum_phi, sum_gamma = (
            diag.get_sums()
        )

        self._finalize_results(
            saved=saved,
            sum_beta=sum_beta,
            sum_sigma=sum_sigma,
            sum_E_bar=sum_E_bar,
            sum_njt=sum_njt,
            sum_phi=sum_phi,
            sum_gamma=sum_gamma,
        )
        self.success = True

    def get_results(self) -> dict:
        if self._results is None:
            return {"success": False}
        return dict(self._results)

    def _initialize_njt_modes(self, ridge: float, max_lbfgs_iters: int) -> None:
        """
        One-time initialization of njt by setting each market block njt[t, :]
        to the conditional mode (argmax) of the market log-posterior given
        current values of (beta_p, beta_w, r, E_bar[t], gamma[t], phi[t]).

        This is to avoid TMH (independence) getting stuck when initialized at 0.
        """
        dtype = tf.float64

        njt_new_full = tf.identity(self.njt.read_value())

        for t in range(self.T):
            qjt_t = self.qjt[t]
            q0t_t = self.q0t[t]
            pjt_t = self.pjt[t]
            wjt_t = self.wjt[t]

            E_bar_t = self.E_bar[t]
            gamma_t = self.gamma[t]
            phi_t = self.phi[t]

            theta0 = tf.cast(self.njt[t], dtype)

            def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
                return self.posterior.market_logpost(
                    qjt_t=qjt_t,
                    q0t_t=q0t_t,
                    pjt_t=pjt_t,
                    wjt_t=wjt_t,
                    beta_p=self.beta_p,
                    beta_w=self.beta_w,
                    r=self.r,
                    E_bar_t=E_bar_t,
                    njt_t=njt_t_val,
                    gamma_t=gamma_t,
                    phi_t=phi_t,
                )

            # L-BFGS maximize logp (equivalently minimize -logp)
            def val_and_grad(x: tf.Tensor):
                x = tf.convert_to_tensor(x, dtype=dtype)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    val = -logp_njt_t(x)
                grad = tape.gradient(val, x)
                return val, grad

            res = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=val_and_grad,
                initial_position=theta0,
                max_iterations=int(max_lbfgs_iters),
            )

            mu = res.position

            # Robust fallback: if not converged or non-finite, keep theta0
            ok = tf.logical_and(
                tf.cast(res.converged, tf.bool),
                tf.reduce_all(tf.math.is_finite(mu)),
            )
            mu_safe = tf.where(ok, mu, theta0)

            njt_new_full = tf.tensor_scatter_nd_update(
                njt_new_full, indices=[[t]], updates=tf.expand_dims(mu_safe, axis=0)
            )

        self.njt.assign(njt_new_full)

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
        max_lbfgs_iters: int,
        diag: LuShrinkageDiagnostics,
    ) -> None:
        """
        Owns the full MCMC loop, mutating sampler state (tf.Variables) and
        accumulating posterior draw sums.
        """
        # Make scalar hyperparams tensors (stable dtypes; reduces retracing risk when
        # passed into tf.function-compiled market updates)
        ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

        for it in range(n_iter):
            self._update_beta_block(
                k_beta=k_beta, ridge=ridge_t, max_lbfgs_iters=max_lbfgs_iters
            )
            self._update_r_block(k_r=k_r)

            for t in range(self.T):
                # Slice market-specific data/state in Python using t (int).
                qjt_t = self.qjt[t]
                q0t_t = self.q0t[t]
                pjt_t = self.pjt[t]
                wjt_t = self.wjt[t]

                E_bar_t = self.E_bar[t]
                njt_t = self.njt[t]
                gamma_t = self.gamma[t]
                phi_t = self.phi[t]

                E_bar_new, njt_new, gamma_new, phi_new = self._update_market_block(
                    qjt_t=qjt_t,
                    q0t_t=q0t_t,
                    pjt_t=pjt_t,
                    wjt_t=wjt_t,
                    E_bar_t=E_bar_t,
                    njt_t=njt_t,
                    gamma_t=gamma_t,
                    phi_t=phi_t,
                    k_E_bar=k_E_bar,
                    k_njt=k_njt,
                    ridge=ridge_t,
                    max_lbfgs_iters=max_lbfgs_iters,
                )

                # Write back into full state (scatter update) using t (int).
                self.E_bar.assign(
                    tf.tensor_scatter_nd_update(self.E_bar, [[t]], [E_bar_new])
                )
                self.njt.assign(
                    tf.tensor_scatter_nd_update(
                        self.njt, [[t]], tf.expand_dims(njt_new, axis=0)
                    )
                )
                self.gamma.assign(
                    tf.tensor_scatter_nd_update(
                        self.gamma, [[t]], tf.expand_dims(gamma_new, axis=0)
                    )
                )
                self.phi.assign(tf.tensor_scatter_nd_update(self.phi, [[t]], [phi_new]))

            diag.step(self, it)

    def _finalize_results(
        self,
        saved: int,
        sum_beta: tf.Tensor,
        sum_sigma: tf.Tensor,
        sum_E_bar: tf.Tensor,
        sum_njt: tf.Tensor,
        sum_phi: tf.Tensor,
        sum_gamma: tf.Tensor,
    ) -> None:
        saved_f = tf.cast(saved, tf.float64)

        beta_mean = (sum_beta / saved_f).numpy()
        sigma_mean = float((sum_sigma / saved_f).numpy())
        E_bar_mean = (sum_E_bar / saved_f).numpy()
        njt_mean = (sum_njt / saved_f).numpy()
        E_mean = E_bar_mean[:, None] + njt_mean
        phi_mean = (sum_phi / saved_f).numpy()
        gamma_mean = (sum_gamma / saved_f).numpy()

        self._results = {
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
    # Block updates (mutate state)
    # ------------------------------------------------------------------

    def _update_beta_block(
        self, k_beta: tf.Tensor, ridge: tf.Tensor, max_lbfgs_iters: int
    ) -> None:
        beta0 = tf.stack([self.beta_p, self.beta_w], axis=0)

        def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
            beta_p = theta_vec[0]
            beta_w = theta_vec[1]
            ll = tf.constant(0.0, dtype=tf.float64)
            for t in range(self.T):
                ll += self.posterior.market_loglik(
                    qjt_t=self.qjt[t],
                    q0t_t=self.q0t[t],
                    pjt_t=self.pjt[t],
                    wjt_t=self.wjt[t],
                    beta_p=beta_p,
                    beta_w=beta_w,
                    r=self.r,
                    E_bar_t=self.E_bar[t],
                    njt_t=self.njt[t],
                )
            return ll + self.posterior.logprior_beta(beta_p=beta_p, beta_w=beta_w)

        beta_new, _ = tmh_step(
            theta0=beta0,
            logp_fn=logp_beta,
            ridge=ridge,
            rng=self.rng,
            k=k_beta,
        )
        self.beta_p.assign(beta_new[0])
        self.beta_w.assign(beta_new[1])

    def _update_r_block(self, k_r: tf.Tensor) -> None:
        def logp_r(r_val: tf.Tensor) -> tf.Tensor:
            ll = tf.constant(0.0, dtype=tf.float64)
            for t in range(self.T):
                ll += self.posterior.market_loglik(
                    qjt_t=self.qjt[t],
                    q0t_t=self.q0t[t],
                    pjt_t=self.pjt[t],
                    wjt_t=self.wjt[t],
                    beta_p=self.beta_p,
                    beta_w=self.beta_w,
                    r=r_val,
                    E_bar_t=self.E_bar[t],
                    njt_t=self.njt[t],
                )
            return ll + self.posterior.logprior_r(r=r_val)

        r_new, _, _ = rw_mh_step(
            theta0=self.r,
            logp_fn=logp_r,
            k=k_r,
            rng=self.rng,
        )
        self.r.assign(r_new)

    # ------------------------------------------------------------------
    # Market block (functional per-market update; compiled)
    # ------------------------------------------------------------------

    @tf.function
    def _update_market_block(
        self,
        *,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
        phi_t: tf.Tensor,
        k_E_bar: tf.Tensor,
        k_njt: tf.Tensor,
        ridge: tf.Tensor,
        max_lbfgs_iters: int,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        E_bar_t_new = self._update_E_bar_t(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
            gamma_t=gamma_t,
            phi_t=phi_t,
            k_E_bar=k_E_bar,
        )

        njt_t_new = self._update_njt_tmh_full(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            E_bar_t=E_bar_t_new,
            njt_t=njt_t,
            gamma_t=gamma_t,
            phi_t=phi_t,
            k_njt=k_njt,
            ridge=ridge,
            max_lbfgs_iters=max_lbfgs_iters,
        )

        gamma_t_new = self._update_gamma_t_given_n_phi(
            njt_t=njt_t_new,
            phi_t=phi_t,
        )

        phi_t_new = self._update_phi_t(
            gamma_t=gamma_t_new,
        )

        return E_bar_t_new, njt_t_new, gamma_t_new, phi_t_new

    def _update_E_bar_t(
        self,
        *,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
        phi_t: tf.Tensor,
        k_E_bar: tf.Tensor,
    ) -> tf.Tensor:
        def logp_E_bar_t(E_bar_t_val: tf.Tensor) -> tf.Tensor:
            return self.posterior.market_logpost(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=self.r,
                E_bar_t=E_bar_t_val,
                njt_t=njt_t,
                gamma_t=gamma_t,
                phi_t=phi_t,
            )

        E_bar_new, _, _ = rw_mh_step(
            theta0=E_bar_t,
            logp_fn=logp_E_bar_t,
            k=k_E_bar,
            rng=self.rng,
        )

        return E_bar_new

    def _update_njt_tmh_full(
        self,
        *,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
        phi_t: tf.Tensor,
        k_njt: tf.Tensor,
        ridge: tf.Tensor,
        max_lbfgs_iters: int,
    ) -> tf.Tensor:
        def logp_njt_full(njt_t_val: tf.Tensor) -> tf.Tensor:
            return self.posterior.market_logpost(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=self.r,
                E_bar_t=E_bar_t,
                njt_t=njt_t_val,
                gamma_t=gamma_t,
                phi_t=phi_t,
            )

        njt_new, _ = tmh_step(
            theta0=njt_t,
            logp_fn=logp_njt_full,
            ridge=tf.cast(ridge, tf.float64),
            rng=self.rng,
            k=k_njt,
        )
        return njt_new

    def _update_gamma_t_given_n_phi(
        self,
        *,
        njt_t: tf.Tensor,
        phi_t: tf.Tensor,
    ) -> tf.Tensor:
        return sample_gamma_given_n_phi_market(
            njt_t=njt_t,
            phi_t=phi_t,
            T0_sq=self.posterior.T0_sq,
            T1_sq=self.posterior.T1_sq,
            log_T0_sq=self.posterior.log_T0_sq,
            log_T1_sq=self.posterior.log_T1_sq,
            rng=self.rng,
        )

    def _update_phi_t(
        self,
        *,
        gamma_t: tf.Tensor,
    ) -> tf.Tensor:
        return gibbs_phi_market(
            gamma_t=gamma_t,
            a_phi=self.posterior.a_phi,
            b_phi=self.posterior.b_phi,
            rng=self.rng,
        )
