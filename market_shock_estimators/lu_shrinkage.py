# market_shock_estimators/lu_shrinkage.py

import numpy as np

from market_shock_estimators.tmh import TMHState, tmh_step
from market_shock_estimators.rw_mh import rw_mh_step


class LuShrinkageEstimator:
    """
    Shrinkage estimator sampler for Lu (2025).

    Responsibilities:
      - hold sampler state
      - slice market data
      - orchestrate TMH / RW-MH updates
      - cache posterior evaluations per market

    All economics and TensorFlow live in lu_posterior.py.
    """

    def __init__(
        self,
        *,
        posterior,
        x,
        Z,
        q,
        q0,
        beta_init,
        r_init,
        Ebar_init,
        eta_init,
        gamma_init,
        phi_init,
        beta_step=0.05,
        r_step=0.05,
        tmh_kappa=0.003,
        rng=None,
    ):
        """
        All arrays are NumPy.

        Dimensions:
          T markets
          J products
          K regressors
          d RC dimension
        """
        self.posterior = posterior

        self.x = x  # (T,J,K)
        self.Z = Z  # (T,J,d)
        self.q = q  # (T,J)
        self.q0 = q0  # (T,)

        self.T, self.J, _ = x.shape

        # State
        self.beta = beta_init.copy()  # (K,)
        self.r = r_init.copy()  # (d,)
        self.Ebar = Ebar_init.copy()  # (T,)
        self.eta = eta_init.copy()  # (T,J)
        self.gamma = gamma_init.copy()  # (T,J)
        self.phi = phi_init.copy()  # (T,)

        # Proposal parameters
        self.beta_step = beta_step
        self.r_step = r_step
        self.tmh_kappa = tmh_kappa

        self.rng = rng if rng is not None else np.random.default_rng()

        # Acceptance counters
        self.acc_beta = 0
        self.acc_r = 0
        self.acc_eta = np.zeros(self.T, dtype=int)

    # ------------------------------------------------------------------
    # Main sampler step
    # ------------------------------------------------------------------

    def step(self):
        """One full MCMC iteration."""

        # print("[lu_shrinkage.step] start")

        beta_old = self.beta.copy()
        r_old = self.r.copy()
        eta_old = self.eta.copy()

        # ------------------------------------------------------------
        # 1) Update beta (RW-MH)
        # ------------------------------------------------------------
        def beta_logp(b):
            lp = 0.0
            for t in range(self.T):
                theta = np.concatenate(([self.Ebar[t]], self.eta[t]))
                lp += self.posterior.market_logp(
                    theta=theta,
                    q=self.q[t],
                    q0=self.q0[t],
                    x=self.x[t],
                    Z=self.Z[t],
                    beta=b,
                    r=self.r,
                    gamma=self.gamma[t],
                )
            # prior
            lp += -0.5 * np.sum(b * b) / self.posterior.beta_var.numpy()
            return lp

        self.beta, accepted, _ = rw_mh_step(
            self.beta, beta_logp, self.beta_step**2, self.rng
        )
        self.acc_beta += int(accepted)
        # print("[lu_shrinkage.step] beta update done")

        # ------------------------------------------------------------
        # 2) Update r (RW-MH)
        # ------------------------------------------------------------
        def r_logp(r):
            lp = 0.0
            for t in range(self.T):
                theta = np.concatenate(([self.Ebar[t]], self.eta[t]))
                lp += self.posterior.market_logp(
                    theta=theta,
                    q=self.q[t],
                    q0=self.q0[t],
                    x=self.x[t],
                    Z=self.Z[t],
                    beta=self.beta,
                    r=r,
                    gamma=self.gamma[t],
                )
            lp += -0.5 * np.sum(r * r) / self.posterior.r_var.numpy()
            return lp

        self.r, accepted, _ = rw_mh_step(self.r, r_logp, self.r_step**2, self.rng)
        self.acc_r += int(accepted)
        # print("[lu_shrinkage.step] r update done")

        # ------------------------------------------------------------
        # 3) Update (Ebar_t, eta_t) via TMH, market by market
        # ------------------------------------------------------------
        n_eta_accepted = 0
        for t in range(self.T):

            # print(f"[lu_shrinkage.step] market {t}")
            # Slice market data once
            q_t = self.q[t]
            q0_t = self.q0[t]
            x_t = self.x[t]
            Z_t = self.Z[t]
            gamma_t = self.gamma[t]

            # Initial theta
            theta_curr = np.concatenate(([self.Ebar[t]], self.eta[t]))

            # --------------------------------------------------------
            # Local cache (two slots only)
            # --------------------------------------------------------

            cache = {}

            def evaluate(theta):
                theta = np.asarray(theta)
                key = (
                    theta.tobytes()
                )  # exact key; if theta differs numerically, we recompute safely
                if key not in cache:
                    lp, grad, hess = self.posterior.market_grad_hess(
                        theta=theta,
                        q=q_t,
                        q0=q0_t,
                        x=x_t,
                        Z=Z_t,
                        beta=self.beta,
                        r=self.r,
                        gamma=gamma_t,
                    )
                    # print(f"[lu_shrinkage.step] market {t} grad/hess computed")
                    # keep cache tiny: TMH typically needs only current + proposal
                    if len(cache) >= 2:
                        cache.clear()
                    cache[key] = (lp, grad, hess)
                return cache[key]

            def logp(theta):
                return evaluate(theta)[0]

            def grad(theta):
                return evaluate(theta)[1]

            def hess(theta):
                return evaluate(theta)[2]

            state = TMHState(theta=theta_curr, logp=float(logp(theta_curr)))

            state_new, accepted = tmh_step(
                state,
                logp,
                grad,
                hess,
                rng=self.rng,
                kappa=self.tmh_kappa,
            )

            # print(f"[lu_shrinkage.step] market {t} TMH done accepted={accepted}")
            theta_new = state_new.theta

            if accepted:
                self.Ebar[t] = theta_new[0]
                self.eta[t] = theta_new[1:]
                self.acc_eta[t] += 1
                n_eta_accepted += 1

        print(
            f"[LuShrinkage] eta_accept_frac={n_eta_accepted / self.T:.3f} ({n_eta_accepted}/{self.T})"
        )
        # ------------------------------------------------------------
        # 4) Update gamma and phi (Gibbs)
        # ------------------------------------------------------------
        for t in range(self.T):
            eta_t = self.eta[t]
            phi_t = self.phi[t]

            # gamma_jt | eta_jt, phi_t
            T1_sq = float(self.posterior.T1_sq.numpy())
            T0_sq = float(self.posterior.T0_sq.numpy())

            # Include Normal(0, tau^2) normalizing constants: 1/sqrt(tau^2)
            p1 = phi_t * (1.0 / np.sqrt(T1_sq)) * np.exp(-0.5 * eta_t * eta_t / T1_sq)
            p0 = (
                (1.0 - phi_t)
                * (1.0 / np.sqrt(T0_sq))
                * np.exp(-0.5 * eta_t * eta_t / T0_sq)
            )

            prob = p1 / (p1 + p0)
            self.gamma[t] = (self.rng.random(self.J) < prob).astype(int)

            # phi_t | gamma_t
            a = self.posterior.a_phi.numpy() + np.sum(self.gamma[t])
            b = self.posterior.b_phi.numpy() + self.J - np.sum(self.gamma[t])
            self.phi[t] = self.rng.beta(a, b)

        _print_iteration_diagnostics(
            beta_old=beta_old,
            r_old=r_old,
            eta_old=eta_old,
            beta=self.beta,
            r=self.r,
            eta=self.eta,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def state(self):
        return {
            "beta": self.beta.copy(),
            "r": self.r.copy(),
            "Ebar": self.Ebar.copy(),
            "eta": self.eta.copy(),
            "gamma": self.gamma.copy(),
            "phi": self.phi.copy(),
        }


def _print_iteration_diagnostics(beta_old, r_old, eta_old, beta, r, eta):
    """
    Minimal per-iteration diagnostics.
    Uses only old snapshots and current state.
    """

    # Parameter movement
    d_beta = float(np.linalg.norm(beta - beta_old))
    d_r = float(np.linalg.norm(r - r_old))
    d_eta = float(np.mean(np.linalg.norm(eta - eta_old, axis=1)))

    # Shrinkage activity
    eta_norm = float(np.mean(np.linalg.norm(eta, axis=1)))

    print(
        "[LuShrinkage diagnostics] "
        f"||Δβ||={d_beta:.3e} "
        f"|Δr|={d_r:.3e} "
        f"mean||Δη||={d_eta:.3e} "
        f"mean||η||={eta_norm:.3e}"
    )
