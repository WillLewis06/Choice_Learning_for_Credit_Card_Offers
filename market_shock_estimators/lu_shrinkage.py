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
#               (gamma_t, eta_t) via MH toggles + TMH on active eta;
#               phi_t via Gibbs.

#
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from market_shock_estimators.lu_posterior import LuPosteriorTF
from market_shock_estimators.rw_mh import rw_mh_step
from market_shock_estimators.tmh import tmh_step


@dataclass(frozen=True)
class ShrinkageMCMCConfig:
    n_iter: int = 1500
    burn_in: int = 500
    thin: int = 5

    # RW-MH step sizes (scalars)
    r_step: float = 0.05
    E_bar_step: float = 0.05

    # TMH numeric parameters
    ridge: float = 1e-6
    max_lbfgs_iters: int = 100


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
        E_bar_t, eta[t,j]
      Sparsity/hyper:
        gamma[t,j] in {0,1}, phi[t] in (0,1)

    Data:
      pjt, wjt, qjt, q0t
    """

    def __init__(
        self,
        *,
        pjt: np.ndarray,
        wjt: np.ndarray,
        qjt: np.ndarray,
        q0t: np.ndarray,
        n_draws: int,
        seed: int,
        # Posterior hyperparameters (defaults match LuPosteriorTF defaults unless overridden)
        posterior_kwargs: dict | None = None,
        dtype=tf.float64,
    ):
        self.dtype = dtype
        self.success: bool = False
        self._results: dict | None = None

        # -----------------------------
        # Data
        # -----------------------------
        self.pjt = tf.convert_to_tensor(pjt, dtype=dtype)  # (T,J)
        self.wjt = tf.convert_to_tensor(wjt, dtype=dtype)  # (T,J)
        self.qjt = tf.convert_to_tensor(qjt, dtype=dtype)  # (T,J)
        self.q0t = tf.convert_to_tensor(q0t, dtype=dtype)  # (T,)

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
        # Posterior object (owns fixed v_draws for RC integration)
        # -----------------------------
        posterior_kwargs = {} if posterior_kwargs is None else dict(posterior_kwargs)
        self.posterior = LuPosteriorTF(
            n_draws=int(n_draws),
            seed=int(seed),
            dtype=dtype,
            **posterior_kwargs,
        )

        # -----------------------------
        # RNG (for proposals + Gibbs)
        # -----------------------------
        self.rng = tf.random.Generator.from_seed(int(seed))

        # -----------------------------
        # Initialize state (minimal, deterministic)
        # -----------------------------
        self.beta_p = tf.Variable(0.0, dtype=dtype, trainable=False)
        self.beta_w = tf.Variable(0.0, dtype=dtype, trainable=False)
        self.r = tf.Variable(0.0, dtype=dtype, trainable=False)  # r = log(sigma)

        self.E_bar = tf.Variable(
            tf.fill([self.T], tf.cast(self.posterior.E_bar_mean, dtype)),
            trainable=False,
        )  # (T,)

        self.eta = tf.Variable(
            tf.zeros([self.T, self.J], dtype=dtype), trainable=False
        )  # (T,J)

        # Start sparse: gamma=0, phi=Beta(a_phi,b_phi) prior mean, eta=0
        self.gamma = tf.Variable(
            tf.zeros([self.T, self.J], dtype=tf.int32), trainable=False
        )
        phi0 = tf.cast(self.posterior.a_phi, dtype) / (
            tf.cast(self.posterior.a_phi + self.posterior.b_phi, dtype)
        )
        self.phi = tf.Variable(tf.fill([self.T], phi0), trainable=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        *,
        n_iter: int = 1500,
        burn_in: int = 500,
        thin: int = 5,
        r_step: float = 0.05,
        E_bar_step: float = 0.05,
        ridge: float = 1e-6,
        max_lbfgs_iters: int = 100,
        verbose: bool = False,
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

        cfg = ShrinkageMCMCConfig(
            n_iter=int(n_iter),
            burn_in=int(burn_in),
            thin=int(thin),
            r_step=float(r_step),
            E_bar_step=float(E_bar_step),
            ridge=float(ridge),
            max_lbfgs_iters=int(max_lbfgs_iters),
        )

        # -----------------------------
        # Accumulators for posterior means
        # -----------------------------
        saved = 0
        sum_beta = tf.zeros([2], dtype=self.dtype)
        sum_sigma = tf.constant(0.0, dtype=self.dtype)
        sum_E_bar = tf.zeros([self.T], dtype=self.dtype)
        sum_eta = tf.zeros([self.T, self.J], dtype=self.dtype)
        sum_phi = tf.zeros([self.T], dtype=self.dtype)
        sum_gamma = tf.zeros([self.T, self.J], dtype=self.dtype)

        # -----------------------------
        # MCMC loop
        # -----------------------------
        for it in range(cfg.n_iter):
            # 1) Global: beta = (beta_p, beta_w) via TMH (2-vector)
            beta0 = tf.stack([self.beta_p, self.beta_w], axis=0)

            def logp_beta(theta_vec):
                beta_p = theta_vec[0]
                beta_w = theta_vec[1]
                ll = tf.constant(0.0, dtype=self.dtype)
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
                        eta_t=self.eta[t] * tf.cast(self.gamma[t], self.dtype),
                    )
                return ll + self.posterior.logprior_beta(beta_p=beta_p, beta_w=beta_w)

            beta_new, _ = tmh_step(
                theta0=beta0,
                logp_fn=logp_beta,
                ridge=tf.cast(cfg.ridge, self.dtype),
                max_lbfgs_iters=cfg.max_lbfgs_iters,
                rng=self.rng,
            )
            self.beta_p.assign(beta_new[0])
            self.beta_w.assign(beta_new[1])

            # 2) Global: r via RW-MH (scalar)
            def logp_r(r_val):
                ll = tf.constant(0.0, dtype=self.dtype)
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
                        eta_t=self.eta[t] * tf.cast(self.gamma[t], self.dtype),
                    )
                return ll + self.posterior.logprior_r(r=r_val)

            r_new, _ = rw_mh_step(
                theta0=self.r,
                logp_fn=logp_r,
                step_size=tf.cast(cfg.r_step, self.dtype),
                rng=self.rng,
            )
            self.r.assign(r_new)

            # 3) Market-by-market blocks (Option A)
            for t in range(self.T):
                # 3a) E_bar_t via RW-MH (scalar)
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
                        eta_t=self.eta[t],
                        gamma_t=self.gamma[t],
                        phi_t=self.phi[t],
                    )

                E_bar_new, _ = rw_mh_step(
                    theta0=self.E_bar[t],
                    logp_fn=logp_E_bar_t,
                    step_size=tf.cast(cfg.E_bar_step, self.dtype),
                    rng=self.rng,
                )
                self.E_bar.assign(
                    tf.tensor_scatter_nd_update(
                        self.E_bar, [[t]], [tf.cast(E_bar_new, self.dtype)]
                    )
                )

                # 3b) gamma_t via MH toggles (birth/death), updates eta_t accordingly
                self._mh_toggle_gamma_market(t)

                # 3c) eta_t via TMH on active coordinates only
                gamma_t = tf.cast(self.gamma[t], tf.int32)  # (J,)
                active_idx = tf.where(tf.equal(gamma_t, 1))[:, 0]  # (K,)
                K = tf.shape(active_idx)[0]

                if int(K.numpy()) > 0:
                    eta_t_current = tf.cast(self.eta[t], self.dtype)
                    eta_active0 = tf.gather(eta_t_current, active_idx)  # (K,)

                    def logp_eta_active(eta_active_val):
                        eta_full = tf.zeros([self.J], dtype=self.dtype)
                        eta_full = tf.tensor_scatter_nd_update(
                            eta_full,
                            active_idx[:, None],
                            tf.cast(eta_active_val, self.dtype),
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
                            eta_t=eta_full,
                            gamma_t=self.gamma[t],
                            phi_t=self.phi[t],
                        )

                    eta_active_new, _ = tmh_step(
                        theta0=eta_active0,
                        logp_fn=logp_eta_active,
                        ridge=tf.cast(cfg.ridge, self.dtype),
                        max_lbfgs_iters=cfg.max_lbfgs_iters,
                        rng=self.rng,
                    )

                    eta_full_new = tf.zeros([self.J], dtype=self.dtype)
                    eta_full_new = tf.tensor_scatter_nd_update(
                        eta_full_new,
                        active_idx[:, None],
                        tf.cast(eta_active_new, self.dtype),
                    )
                    self.eta.assign(
                        tf.tensor_scatter_nd_update(self.eta, [[t]], [eta_full_new])
                    )
                else:
                    self.eta.assign(
                        tf.tensor_scatter_nd_update(
                            self.eta, [[t]], [tf.zeros([self.J], dtype=self.dtype)]
                        )
                    )

                # 3d) phi_t via Gibbs (conjugate)
                self._gibbs_phi_market(t)

                # 3e) Optional invariant check (reuse verbose flag)
                if verbose:
                    inactive = 1.0 - tf.cast(self.gamma[t], self.dtype)
                    viol = tf.reduce_max(tf.abs(self.eta[t]) * inactive)
                    if bool((viol > tf.cast(1e-10, self.dtype)).numpy()):
                        raise ValueError(
                            f"Option A invariant violated at market t={t}: "
                            f"max |eta_j| for gamma_j=0 is {float(viol.numpy())}"
                        )

            # 4) Save draw
            if it >= cfg.burn_in and ((it - cfg.burn_in) % cfg.thin == 0):
                saved += 1
                sum_beta += tf.stack([self.beta_p, self.beta_w], axis=0)
                sum_sigma += tf.exp(self.r)

                sum_E_bar += tf.identity(self.E_bar)
                sum_eta += tf.identity(self.eta)

                sum_phi += tf.identity(self.phi)
                sum_gamma += tf.cast(self.gamma, self.dtype)

            if verbose and (it % 100 == 0):
                # Lightweight diagnostics only
                sigma_now = float(tf.exp(self.r).numpy())
                inc_rate = float(
                    tf.reduce_mean(tf.cast(self.gamma, self.dtype)).numpy()
                )
                print(
                    f"[LuShrinkage] it={it} sigma={sigma_now:.3f} mean(gamma)={inc_rate:.3f}"
                )

        if saved == 0:
            raise RuntimeError("No posterior draws were saved (check burn_in/thin).")

        # -----------------------------
        # Posterior means + result bundle
        # -----------------------------
        saved_f = tf.cast(saved, self.dtype)

        beta_mean = (sum_beta / saved_f).numpy()
        sigma_mean = float((sum_sigma / saved_f).numpy())

        E_bar_mean = (sum_E_bar / saved_f).numpy()
        eta_mean = (sum_eta / saved_f).numpy()
        E_mean = E_bar_mean[:, None] + eta_mean

        phi_mean = (sum_phi / saved_f).numpy()
        gamma_mean = (sum_gamma / saved_f).numpy()

        self._results = {
            "success": True,
            # paper-aligned names
            "beta_p_hat": float(beta_mean[0]),
            "beta_w_hat": float(beta_mean[1]),
            "sigma_hat": sigma_mean,
            # primary target for assessment
            "E_hat": E_mean,
            # optional diagnostics
            "E_bar_hat": E_bar_mean,
            "eta_hat": eta_mean,
            "phi_hat": phi_mean,
            "gamma_hat": gamma_mean,
            "n_saved": int(saved),
            "mcmc": {
                "n_iter": cfg.n_iter,
                "burn_in": cfg.burn_in,
                "thin": cfg.thin,
            },
        }
        self.success = True

    def get_results(self) -> dict:
        """
        Return estimator outputs in a single bundle.

        Keys (minimum consistent with assess_estimator_results usage elsewhere):
          - "success":   bool
          - "sigma_hat": scalar float
          - "E_hat":     (T, J) array
          - "beta_p_hat", "beta_w_hat": scalars

        Also returns posterior means of latent components and sparsity stats.
        """
        if self._results is None:
            return {"success": False}
        return dict(self._results)

    # ------------------------------------------------------------------
    # Gibbs updates (market t)
    # ------------------------------------------------------------------

    def _mh_toggle_gamma_market(self, t: int) -> None:
        """
        Option A (point-mass spike): MH toggles for (gamma_{jt}, eta_{jt}).

        For each j:
          - birth:  gamma 0->1 and draw eta_j ~ N(0, sigma_eta_sq)
          - death:  gamma 1->0 and set eta_j = 0

        Accept/reject using market_logpost difference + proposal correction.
        """
        gamma_t = tf.cast(self.gamma[t], tf.int32)  # (J,)
        eta_t = tf.cast(self.eta[t], self.dtype)  # (J,)
        phi_t = tf.cast(self.phi[t], self.dtype)  # scalar

        for j in range(self.J):
            g_old = int(gamma_t[j].numpy())

            gamma_old = gamma_t
            eta_old = eta_t * tf.cast(gamma_old, self.dtype)  # enforce invariant

            if g_old == 0:
                # birth: propose gamma=1 and eta_j ~ slab prior
                var = tf.cast(self.posterior.sigma_eta_sq, self.dtype)
                sd = tf.sqrt(var)
                eta_j_prop = sd * self.rng.normal([], dtype=self.dtype)

                gamma_new = tf.tensor_scatter_nd_update(gamma_old, [[j]], [1])
                eta_new = tf.tensor_scatter_nd_update(eta_old, [[j]], [eta_j_prop])
            else:
                # death: propose gamma=0 and eta_j = 0
                gamma_new = tf.tensor_scatter_nd_update(gamma_old, [[j]], [0])
                eta_new = tf.tensor_scatter_nd_update(
                    eta_old, [[j]], [tf.cast(0.0, self.dtype)]
                )

            eta_new = eta_new * tf.cast(gamma_new, self.dtype)  # enforce invariant

            lp_old = self.posterior.market_logpost(
                qjt_t=self.qjt[t],
                q0t_t=self.q0t[t],
                pjt_t=self.pjt[t],
                wjt_t=self.wjt[t],
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=self.r,
                E_bar_t=self.E_bar[t],
                eta_t=eta_old,
                gamma_t=gamma_old,
                phi_t=phi_t,
            )
            lp_new = self.posterior.market_logpost(
                qjt_t=self.qjt[t],
                q0t_t=self.q0t[t],
                pjt_t=self.pjt[t],
                wjt_t=self.wjt[t],
                beta_p=self.beta_p,
                beta_w=self.beta_w,
                r=self.r,
                E_bar_t=self.E_bar[t],
                eta_t=eta_new,
                gamma_t=gamma_new,
                phi_t=phi_t,
            )

            # Proposal correction for reversible birth/death:
            var = tf.cast(self.posterior.sigma_eta_sq, self.dtype)
            if g_old == 0:
                # forward draws eta_j_prop; reverse is deterministic
                log_q_forward = (
                    -0.5 * tf.math.log(self.posterior.two_pi * var)
                    - 0.5 * tf.square(eta_j_prop) / var
                )
                log_q_reverse = tf.cast(0.0, self.dtype)
            else:
                # forward deterministic; reverse would draw eta_j from slab at old value
                eta_j_old = tf.cast(eta_old[j], self.dtype)
                log_q_forward = tf.cast(0.0, self.dtype)
                log_q_reverse = (
                    -0.5 * tf.math.log(self.posterior.two_pi * var)
                    - 0.5 * tf.square(eta_j_old) / var
                )

            log_alpha = (lp_new - lp_old) + (log_q_reverse - log_q_forward)
            u = self.rng.uniform([], dtype=self.dtype)
            if bool((tf.math.log(u) < log_alpha).numpy()):
                gamma_t = tf.cast(gamma_new, tf.int32)
                eta_t = eta_new

        # persist (already masked)
        self.gamma.assign(tf.tensor_scatter_nd_update(self.gamma, [[t]], [gamma_t]))
        self.eta.assign(tf.tensor_scatter_nd_update(self.eta, [[t]], [eta_t]))

    def _gibbs_phi_market(self, t: int) -> None:
        """
        Gibbs update: phi_t | gamma_t ~ Beta(a + sum gamma, b + J - sum gamma).
        """
        gamma_t = tf.cast(self.gamma[t], self.dtype)
        a_post = tf.cast(self.posterior.a_phi, self.dtype) + tf.reduce_sum(gamma_t)
        b_post = (
            tf.cast(self.posterior.b_phi, self.dtype)
            + tf.cast(self.J, self.dtype)
            - tf.reduce_sum(gamma_t)
        )

        phi_new = _sample_beta_tf(self.rng, a_post, b_post, dtype=self.dtype)
        self.phi.assign(
            tf.tensor_scatter_nd_update(self.phi, [[t]], [tf.cast(phi_new, self.dtype)])
        )


def _sample_beta_tf(rng: tf.random.Generator, a: tf.Tensor, b: tf.Tensor, dtype):
    """
    Sample Beta(a,b) using Gamma draws:
        X ~ Gamma(a, 1), Y ~ Gamma(b, 1), return X / (X + Y).
    Uses stateless Gamma draws seeded from the provided tf.random.Generator.
    """
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)

    seeds = rng.make_seeds(2)  # (2,2) int32
    seed_x = seeds[0]
    seed_y = seeds[1]

    x = tf.random.stateless_gamma(
        shape=[],
        seed=seed_x,
        alpha=a,
        beta=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    y = tf.random.stateless_gamma(
        shape=[],
        seed=seed_y,
        alpha=b,
        beta=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    return x / (x + y)
