# market_shock_estimators/lu_shrinkage.py

import numpy as np

from market_shock_estimators.tmh import tmh_update_block
from market_shock_estimators.lu_posterior import LuPosteriorTF


class LuShrinkageEstimator:
    """
    Lu (2025) shrinkage estimator — Section 4 posterior with Lu-style TMH updates.

    Design (canonical posterior in TF):
      - All log posterior evaluation uses LuPosteriorTF (single source of truth).
      - TF provides market-block derivatives (grad/Hess) via LuPosteriorTF.
      - This class only orchestrates sampling (TMH + Gibbs) and stores state.

    Random coefficients:
      - General d-dimensional RC on selected columns of x_jt via rc_indices.
      - Default is d=1 with rc_indices=[price_index] when draws is 1D.

    Spike-and-slab:
      - njt | gamma_jt uses T0_sq (spike) / T1_sq (slab).
      - gamma_jt | phi_t Bernoulli; phi_t Beta(a_phi,b_phi).
    """

    def __init__(
        self,
        x_jt,
        q_jt,
        q0_t,
        draws,
        *,
        price_index=0,
        rc_indices=None,
        T0_sq=1e-3,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
        # Priors (Lu defaults used in paper-style implementations)
        beta_var=10.0,
        Ebar_var=10.0,
        r_var=0.5,
        # TMH controls
        tmh_ridge=1e-6,
        tmh_kappa=None,  # None => Lu default 2.38/sqrt(d) inside tmh.py
        tmh_newton_max_iter=25,
        tmh_newton_tol=1e-8,
        # MCMC controls
        max_iter=3000,
        burn_in=1000,
        thin=5,
        seed=0,
    ):
        self.x_jt = np.asarray(x_jt, dtype=float)
        self.q_jt = np.asarray(q_jt, dtype=float)
        self.q0_t = np.asarray(q0_t, dtype=float)
        draws = np.asarray(draws, dtype=float)

        if self.x_jt.ndim != 3:
            raise ValueError("x_jt must have shape (T, J, K).")
        self.T, self.J, self.K = self.x_jt.shape

        if self.q_jt.shape != (self.T, self.J):
            raise ValueError(f"q_jt must have shape (T, J)=({self.T},{self.J}).")
        if self.q0_t.shape != (self.T,):
            raise ValueError(f"q0_t must have shape (T,)=({self.T},).")
        if np.any(self.q_jt < 0) or np.any(self.q0_t < 0):
            raise ValueError("q_jt and q0_t must be nonnegative.")

        # Normalize draws to (R,d)
        if draws.ndim == 1:
            draws = draws[:, None]
        if draws.ndim != 2 or draws.shape[0] == 0:
            raise ValueError("draws must be 1D (R,) or 2D (R,d) with R>0.")
        self.draws = draws
        self.R, self.d = self.draws.shape

        self.price_index = int(price_index)
        if not (0 <= self.price_index < self.K):
            raise ValueError(f"price_index must be in [0, {self.K - 1}].")

        # rc_indices determines which x columns have random coefficients (length d)
        if rc_indices is None:
            if self.d == 1:
                rc_indices = [self.price_index]
            else:
                raise ValueError(
                    "rc_indices must be provided when draws has d>1. "
                    "Default rc_indices=[price_index] only applies when d=1."
                )
        rc_indices = np.asarray(rc_indices, dtype=int).ravel()
        if rc_indices.size != self.d:
            raise ValueError(
                f"len(rc_indices) must equal d={self.d}. Got {rc_indices.size}."
            )
        if np.any(rc_indices < 0) or np.any(rc_indices >= self.K):
            raise ValueError(f"rc_indices entries must be in [0, {self.K - 1}].")
        self.rc_indices = rc_indices.tolist()

        self.T0_sq = float(T0_sq)
        self.T1_sq = float(T1_sq)
        if self.T0_sq <= 0.0 or self.T1_sq <= 0.0:
            raise ValueError("T0_sq and T1_sq must be positive.")
        self._log_T0_sq = float(np.log(self.T0_sq))
        self._log_T1_sq = float(np.log(self.T1_sq))

        self.a_phi = float(a_phi)
        self.b_phi = float(b_phi)

        self.beta_var = float(beta_var)
        self.Ebar_var = float(Ebar_var)
        self.r_var = float(r_var)

        self.tmh_ridge = float(tmh_ridge)
        self.tmh_kappa = None if tmh_kappa is None else float(tmh_kappa)
        self.tmh_newton_max_iter = int(tmh_newton_max_iter)
        self.tmh_newton_tol = float(tmh_newton_tol)

        self.max_iter = int(max_iter)
        self.burn_in = int(burn_in)
        self.thin = int(thin)

        self.rng = np.random.default_rng(seed)

        # Canonical posterior object (TF)
        self.posterior = LuPosteriorTF(
            x_jt=self.x_jt,
            q_jt=self.q_jt,
            q0_t=self.q0_t,
            draws=self.draws,
            rc_indices=self.rc_indices,
            beta_var=self.beta_var,
            Ebar_var=self.Ebar_var,
            r_var=self.r_var,
            T0_sq=self.T0_sq,
            T1_sq=self.T1_sq,
            a_phi=self.a_phi,
            b_phi=self.b_phi,
        )

        self.converged = False

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def initialize(self):
        self.beta = np.zeros(self.K)
        self.r = np.zeros(self.d)
        self.E_bar_t = np.zeros(self.T)
        self.njt = np.zeros((self.T, self.J))

        self.gamma_jt = np.ones((self.T, self.J), dtype=int)
        self.phi_t = np.full(self.T, 0.5)

        self._beta_mean = np.zeros_like(self.beta)
        self._r_mean = np.zeros_like(self.r)
        self._Ebar_mean = np.zeros_like(self.E_bar_t)
        self._njt_mean = np.zeros_like(self.njt)

    # ------------------------------------------------------------------
    # Discrete updates (Gibbs/conjugate)
    # ------------------------------------------------------------------

    def _update_discrete_gamma_phi(self):
        # gamma_jt | njt, phi_t  (spike-and-slab posterior inclusion)
        log_phi = np.log(self.phi_t[:, None] + 1e-15)
        log_1mphi = np.log(1.0 - self.phi_t[:, None] + 1e-15)

        log_slab = log_phi - 0.5 * (self.njt**2) / self.T1_sq - 0.5 * self._log_T1_sq
        log_spike = log_1mphi - 0.5 * (self.njt**2) / self.T0_sq - 0.5 * self._log_T0_sq

        p = 1.0 / (1.0 + np.exp(log_spike - log_slab))
        self.gamma_jt = (self.rng.uniform(size=p.shape) < p).astype(int)

        # phi_t | gamma_t  (Beta conjugate)
        gsum = self.gamma_jt.sum(axis=1)
        self.phi_t = self.rng.beta(self.a_phi + gsum, self.b_phi + self.J - gsum)

    # ------------------------------------------------------------------
    # One MCMC sweep
    # ------------------------------------------------------------------

    def update(self, lp):
        """
        One sweep:
          1) TMH global block: (beta, r)
          2) TMH market-wise blocks: (E_bar_t[t], njt[t,:]) with TF grad/Hess
          3) Gibbs/conjugate discrete updates: (gamma_jt, phi_t)
        """

        # Full posterior callable required by tmh_update_block (API requires zero-arg)
        def full_logp():
            return float(
                self.posterior.full_logp(
                    self.beta, self.r, self.E_bar_t, self.njt, self.gamma_jt, self.phi_t
                ).numpy()
            )

        # ---------------------------
        # TMH: global block (beta, r)
        # ---------------------------
        def get_block_beta_r():
            theta = np.concatenate([self.beta, self.r])
            return theta, lp

        def set_block_beta_r(theta_new):
            theta_new = np.asarray(theta_new, dtype=float)
            self.beta = theta_new[: self.K].copy()
            self.r = theta_new[self.K : self.K + self.d].copy()

        _ = tmh_update_block(
            get_block_beta_r,
            set_block_beta_r,
            full_logp,
            self.rng,
            kappa=self.tmh_kappa,
            ridge=self.tmh_ridge,
            newton_max_iter=self.tmh_newton_max_iter,
            newton_tol=self.tmh_newton_tol,
        )
        lp = full_logp()

        # -------------------------------------------------------
        # TMH: market-wise blocks (E_bar_t[t], njt[t,:]) with TF derivatives
        # -------------------------------------------------------
        beta_curr = self.beta
        r_curr = self.r

        for t in range(self.T):

            def get_block_t(t=t):
                theta = np.concatenate([np.array([self.E_bar_t[t]]), self.njt[t, :]])
                return theta, lp

            def set_block_t(theta_new, t=t):
                theta_new = np.asarray(theta_new, dtype=float)
                self.E_bar_t[t] = float(theta_new[0])
                self.njt[t, :] = theta_new[1:].copy()

            gamma_t = self.gamma_jt[t, :]

            def block_grad(theta_block_new, t=t, gamma_t=gamma_t):
                g, _H = self.posterior.market_block_grad_hess(
                    t, theta_block_new, beta_curr, r_curr, gamma_t
                )
                return np.asarray(g.numpy(), dtype=float)

            def block_hess(theta_block_new, t=t, gamma_t=gamma_t):
                _g, H = self.posterior.market_block_grad_hess(
                    t, theta_block_new, beta_curr, r_curr, gamma_t
                )
                return np.asarray(H.numpy(), dtype=float)

            _ = tmh_update_block(
                get_block_t,
                set_block_t,
                full_logp,
                self.rng,
                block_grad_log_posterior=block_grad,
                block_hess_log_posterior=block_hess,
                kappa=self.tmh_kappa,
                ridge=self.tmh_ridge,
                newton_max_iter=self.tmh_newton_max_iter,
                newton_tol=self.tmh_newton_tol,
            )
            lp = full_logp()

        # ---------------------------
        # Discrete updates
        # ---------------------------
        self._update_discrete_gamma_phi()
        lp = full_logp()
        return lp

    # ------------------------------------------------------------------
    # Fit / results
    # ------------------------------------------------------------------

    def fit(self):
        self.initialize()

        lp = float(
            self.posterior.full_logp(
                self.beta, self.r, self.E_bar_t, self.njt, self.gamma_jt, self.phi_t
            ).numpy()
        )

        kept = 0
        for it in range(self.max_iter):
            lp = self.update(lp)

            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                kept += 1
                w = 1.0 / kept
                self._beta_mean = (1 - w) * self._beta_mean + w * self.beta
                self._r_mean = (1 - w) * self._r_mean + w * self.r
                self._Ebar_mean = (1 - w) * self._Ebar_mean + w * self.E_bar_t
                self._njt_mean = (1 - w) * self._njt_mean + w * self.njt

        self.converged = kept > 0
        return self

    def get_results(self):
        return {
            "success": self.converged,
            "beta_hat": self._beta_mean.copy(),
            "r_hat": self._r_mean.copy(),
            "sigma_hat": np.exp(self._r_mean).copy(),
            "E_hat": self._Ebar_mean[:, None] + self._njt_mean,
        }
