# market_shock_estimators/lu_shrinkage.py
#
# Clean Lu (2025) shrinkage sampler (Option A: point-mass spike-and-slab):
#   - Estimator-style API: fit() + get_results() (no step()).
#   - Log densities delegated to LuPosteriorTF.
#   - MH mechanics via tmh_step (Laplace independence MH) and rw_mh_step.
#
# Blocking (minimal, Lu-aligned):
#   - Global: (beta_p, beta_w) via TMH; r via RW-MH.
#   - Market t: E_bar_t via RW-MH;
#               (gamma_t, njt_t) via MH toggles + TMH on active njt;
#               phi_t via Gibbs.
#
from __future__ import annotations

import numpy as np
import tensorflow as tf

from market_shock_estimators.lu_posterior import LuPosteriorTF
from market_shock_estimators.rw_mh import rw_mh_step
from market_shock_estimators.tmh import tmh_step
from market_shock_estimators.lu_shrinkage_kernels import (
    mh_toggle_gamma_market,
    gibbs_phi_market,
)
from market_shock_estimators.lu_shrinkage_diagnostics import (
    init_progress_state,
    report_iteration_progress,
)


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
        self.posterior = LuPosteriorTF(
            n_draws=int(n_draws),
            seed=int(seed),
        )

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
            tf.zeros([self.T, self.J], dtype=tf.int32), trainable=False
        )

        phi0 = tf.cast(self.posterior.a_phi, tf.float64) / (
            tf.cast(self.posterior.a_phi + self.posterior.b_phi, tf.float64)
        )
        self.phi = tf.Variable(tf.fill([self.T], phi0), trainable=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        n_iter: int,
        burn_in: int,
        thin: int,
        r_step: float,
        E_bar_step: float,
        ridge: float,
        max_lbfgs_iters: int,
    ) -> None:
        """
        Run MCMC and store posterior-mean summaries internally.
        """
        if n_iter <= 0:
            raise ValueError("n_iter must be positive.")
        if burn_in < 0 or burn_in >= n_iter:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_iter.")
        if thin <= 0:
            raise ValueError("thin must be positive.")

        saved = 0
        sum_beta = tf.zeros([2], dtype=tf.float64)
        sum_sigma = tf.constant(0.0, dtype=tf.float64)
        sum_E_bar = tf.zeros([self.T], dtype=tf.float64)
        sum_njt = tf.zeros([self.T, self.J], dtype=tf.float64)
        sum_phi = tf.zeros([self.T], dtype=tf.float64)
        sum_gamma = tf.zeros([self.T, self.J], dtype=tf.float64)

        prev_state = init_progress_state(self)

        # -----------------------------
        # MCMC loop
        # -----------------------------
        for it in range(n_iter):
            # 1) Global beta
            beta0 = tf.stack([self.beta_p, self.beta_w], axis=0)

            def logp_beta(theta_vec):
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
                        njt_t=self.njt[t] * tf.cast(self.gamma[t], tf.float64),
                    )
                return ll + self.posterior.logprior_beta(beta_p=beta_p, beta_w=beta_w)

            beta_new, _ = tmh_step(
                theta0=beta0,
                logp_fn=logp_beta,
                ridge=tf.cast(ridge, tf.float64),
                max_lbfgs_iters=max_lbfgs_iters,
                rng=self.rng,
            )
            self.beta_p.assign(beta_new[0])
            self.beta_w.assign(beta_new[1])

            # 2) Global r
            def logp_r(r_val):
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
                        njt_t=self.njt[t] * tf.cast(self.gamma[t], tf.float64),
                    )
                return ll + self.posterior.logprior_r(r=r_val)

            r_new, _ = rw_mh_step(
                theta0=self.r,
                logp_fn=logp_r,
                step_size=r_step,
                rng=self.rng,
            )
            self.r.assign(r_new)

            # 3) Market blocks
            for t in range(self.T):

                def logp_E_bar_t(E_bar_t_val):
                    return self.posterior.market_logpost(
                        qjt_t=self.qjt[t],
                        q0t_t=self.q0t[t],
                        pjt_t=self.pjt[t],
                        wjt_t=self.wjt[t],
                        beta_p=self.beta_p,
                        beta_w=self.beta_w,
                        r=self.r,
                        E_bar_t=E_bar_t_val,
                        njt_t=self.njt[t],
                        gamma_t=self.gamma[t],
                        phi_t=self.phi[t],
                    )

                E_bar_new, _ = rw_mh_step(
                    theta0=self.E_bar[t],
                    logp_fn=logp_E_bar_t,
                    step_size=E_bar_step,
                    rng=self.rng,
                )
                self.E_bar.assign(
                    tf.tensor_scatter_nd_update(self.E_bar, [[t]], [E_bar_new])
                )

                gamma_t_new, njt_t_new = mh_toggle_gamma_market(self, t)
                self.gamma.assign(
                    tf.tensor_scatter_nd_update(self.gamma, [[t]], [gamma_t_new])
                )
                self.njt.assign(
                    tf.tensor_scatter_nd_update(self.njt, [[t]], [njt_t_new])
                )

                gamma_t = tf.cast(self.gamma[t], tf.int32)
                active_idx = tf.where(tf.equal(gamma_t, 1))[:, 0]
                K = tf.shape(active_idx)[0]

                if int(K.numpy()) > 0:
                    njt_t_current = self.njt[t]
                    njt_active0 = tf.gather(njt_t_current, active_idx)

                    def logp_njt_active(njt_active_val):
                        njt_full = tf.zeros([self.J], dtype=tf.float64)
                        njt_full = tf.tensor_scatter_nd_update(
                            njt_full,
                            active_idx[:, None],
                            njt_active_val,
                        )
                        return self.posterior.market_logpost(
                            qjt_t=self.qjt[t],
                            q0t_t=self.q0t[t],
                            pjt_t=self.pjt[t],
                            wjt_t=self.wjt[t],
                            beta_p=self.beta_p,
                            beta_w=self.beta_w,
                            r=self.r,
                            E_bar_t=self.E_bar[t],
                            njt_t=njt_full,
                            gamma_t=self.gamma[t],
                            phi_t=self.phi[t],
                        )

                    njt_active_new, _ = tmh_step(
                        theta0=njt_active0,
                        logp_fn=logp_njt_active,
                        ridge=tf.cast(ridge, tf.float64),
                        max_lbfgs_iters=max_lbfgs_iters,
                        rng=self.rng,
                    )

                    njt_full_new = tf.zeros([self.J], dtype=tf.float64)
                    njt_full_new = tf.tensor_scatter_nd_update(
                        njt_full_new,
                        active_idx[:, None],
                        tf.cast(njt_active_new, tf.float64),
                    )
                    self.njt.assign(
                        tf.tensor_scatter_nd_update(self.njt, [[t]], [njt_full_new])
                    )
                else:
                    self.njt.assign(
                        tf.tensor_scatter_nd_update(
                            self.njt, [[t]], [tf.zeros([self.J], dtype=tf.float64)]
                        )
                    )

                phi_t_new = gibbs_phi_market(
                    gamma_t=self.gamma[t],
                    a_phi=self.posterior.a_phi,
                    b_phi=self.posterior.b_phi,
                    J=self.J,
                    rng=self.rng,
                )

                self.phi.assign(
                    tf.tensor_scatter_nd_update(
                        self.phi,
                        [[t]],
                        [tf.cast(phi_t_new, tf.float64)],
                    )
                )

            # 4) Save draw
            if it >= burn_in and ((it - burn_in) % thin == 0):
                saved += 1
                sum_beta += tf.stack([self.beta_p, self.beta_w], axis=0)
                sum_sigma += tf.exp(self.r)
                sum_E_bar += tf.identity(self.E_bar)
                sum_njt += tf.identity(self.njt)
                sum_phi += tf.identity(self.phi)
                sum_gamma += tf.cast(self.gamma, tf.float64)

            # ---- Progress report
            prev_state = report_iteration_progress(self, it, prev_state)

        if saved == 0:
            raise RuntimeError("No posterior draws were saved (check burn_in/thin).")

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
        self.success = True

    def get_results(self) -> dict:
        if self._results is None:
            return {"success": False}
        return dict(self._results)
