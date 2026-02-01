import numpy as np
from scipy.optimize import minimize

from market_shock_estimators.inversion import invert_all_markets


def build_strong_IVs(wjt, ujt):
    """
    Strong (cost) instruments, paper-style.

    Constructs Zjt with entries per (t,j):
        Z_strong_jt = [1, wjt, wjt^2, ujt, ujt^2]
    """
    wjt = np.asarray(wjt, dtype=float)
    ujt = np.asarray(ujt, dtype=float)
    ones = np.ones_like(wjt)
    return np.stack([ones, wjt, wjt**2, ujt, ujt**2], axis=2)


def build_weak_IVs(wjt):
    """
    Weak instruments (polynomials in w), paper-style.

    Constructs Zjt with entries per (t,j):
        Z_weak_jt = [1, wjt, wjt^2, wjt^3, wjt^4]
    """
    wjt = np.asarray(wjt, dtype=float)
    ones = np.ones_like(wjt)
    return np.stack([ones, wjt, wjt**2, wjt**3, wjt**4], axis=2)


class BLPEstimator:
    """
    Minimal BLP-style estimator for Lu(25) simulations.

    Demand side (Lu-aligned regressors):
        delta_jt = beta_p * pjt + beta_w * wjt + xi_jt
    i.e., Xjt is constructed internally as [pjt, wjt] with no constant.

    Pipeline:
      - For each candidate sigma, invert demand to recover delta (Berry contraction).
      - Estimate beta via IV: delta = X beta + xi.
      - Form moments g = mean(Z * xi) and minimize g' W g.

    Weighting:
      - Two-step GMM:
          Step 1: W1 = (Z'Z)^{+}
          Step 2: W2 = Omega_hat^{+}

    Robustness additions (numerical, not econometric):
      - Bounds sigma to (0, sigma_max].
      - Converts inversion failures into a large penalty objective value.
      - Warm-starts inversion from last successful delta and applies damping.
    """

    def __init__(
        self,
        sjt,
        s0t,
        pjt,
        wjt,
        Zjt,
        n_draws,
        seed,
        *,
        tol=1e-8,
        share_tol=1e-10,
        max_iter=10_000,
        sigma_max=5.0,
        fail_penalty=1e12,
        damping=1.0,
    ):
        self.sjt = np.asarray(sjt, dtype=float)
        self.s0t = np.asarray(s0t, dtype=float)
        self.pjt = np.asarray(pjt, dtype=float)
        self.wjt = np.asarray(wjt, dtype=float)
        self.Zjt = np.asarray(Zjt, dtype=float)

        self.n_draws = int(n_draws)
        self.seed = int(seed)

        # Lu-aligned demand regressors: X = [p, w] (no constant)
        self.Xjt = np.stack([self.pjt, self.wjt], axis=2)

        # RC integration draws belong to the estimator (not orchestration)
        rng = np.random.default_rng(self.seed)
        self.v_draws = rng.standard_normal(self.n_draws).astype(float)

        self.tol = float(tol)
        self.share_tol = float(share_tol)
        self.max_iter = int(max_iter)

        self.sigma_max = float(sigma_max)
        self.fail_penalty = float(fail_penalty)

        self.damping = float(damping)
        self._delta_warm_start = None  # shape (T,J) after first successful inversion

        self.sigma_hat = None
        self.beta_hat = None
        self.beta_p_hat = None
        self.beta_w_hat = None
        self.E_hat = None
        self.success = False

        self._check_inputs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_inputs(self):
        # -------------------------
        # Dimensionality checks
        # -------------------------
        if self.sjt.ndim != 2:
            raise ValueError(f"sjt must be 2D (T,J). Got ndim={self.sjt.ndim}.")
        if self.pjt.ndim != 2:
            raise ValueError(f"pjt must be 2D (T,J). Got ndim={self.pjt.ndim}.")
        if self.wjt.ndim != 2:
            raise ValueError(f"wjt must be 2D (T,J). Got ndim={self.wjt.ndim}.")
        if self.s0t.ndim != 1:
            raise ValueError(f"s0t must be 1D (T,). Got ndim={self.s0t.ndim}.")
        if self.Xjt.ndim != 3:
            raise ValueError(f"Xjt must be 3D (T,J,Kx). Got ndim={self.Xjt.ndim}.")
        if self.Zjt.ndim != 3:
            raise ValueError(f"Zjt must be 3D (T,J,Kz). Got ndim={self.Zjt.ndim}.")
        if self.v_draws.ndim != 1:
            raise ValueError(f"v_draws must be 1D (R,). Got ndim={self.v_draws.ndim}.")

        T, J = self.sjt.shape

        # -------------------------
        # Shape consistency checks
        # -------------------------
        if self.pjt.shape != (T, J):
            raise ValueError(f"pjt must have shape {(T, J)}. Got {self.pjt.shape}.")
        if self.wjt.shape != (T, J):
            raise ValueError(f"wjt must have shape {(T, J)}. Got {self.wjt.shape}.")
        if self.s0t.shape != (T,):
            raise ValueError(f"s0t must have shape {(T,)}. Got {self.s0t.shape}.")
        if self.Xjt.shape != (T, J, 2):
            raise ValueError(
                f"Xjt must have shape (T,J,2) for [p,w]. Got {self.Xjt.shape}."
            )
        if self.Zjt.shape[0] != T or self.Zjt.shape[1] != J:
            raise ValueError(
                f"Zjt must have shape (T,J,Kz) with T={T}, J={J}. Got {self.Zjt.shape}."
            )
        if self.v_draws.size < 1:
            raise ValueError("n_draws must be >= 1 (v_draws non-empty).")

        # -------------------------
        # Finite-value checks
        # -------------------------
        for name, arr in [
            ("sjt", self.sjt),
            ("s0t", self.s0t),
            ("pjt", self.pjt),
            ("wjt", self.wjt),
            ("Xjt", self.Xjt),
            ("Zjt", self.Zjt),
            ("v_draws", self.v_draws),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf.")

        # -------------------------
        # Share validity checks
        # -------------------------
        if np.any(self.sjt <= 0.0):
            raise ValueError("sjt must be strictly positive.")
        if np.any(self.s0t <= 0.0):
            raise ValueError("s0t must be strictly positive.")
        if np.any(self.sjt >= 1.0):
            raise ValueError("sjt must be strictly less than 1.")
        if np.any(self.s0t >= 1.0):
            raise ValueError("s0t must be strictly less than 1.")

        share_id_tol = 1e-8
        share_err = np.max(np.abs(self.s0t + self.sjt.sum(axis=1) - 1.0))
        if not np.isfinite(share_err) or share_err > share_id_tol:
            raise ValueError(
                f"Share identity violated: max|s0t+sum(sjt)-1|={share_err:.3e} > {share_id_tol:.1e}"
            )

        # -------------------------
        # Parameter sanity checks
        # -------------------------
        if not isinstance(self.n_draws, (int, np.integer)) or int(self.n_draws) < 1:
            raise ValueError("n_draws must be a positive integer.")
        if not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer.")
        if not np.isfinite(self.sigma_max) or self.sigma_max <= 0.0:
            raise ValueError("sigma_max must be a finite positive scalar.")
        if not np.isfinite(self.fail_penalty) or self.fail_penalty <= 0.0:
            raise ValueError("fail_penalty must be a finite positive scalar.")
        if not (0.0 < float(self.damping) <= 1.0):
            raise ValueError("damping must be in (0,1].")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be a finite positive scalar.")
        if not np.isfinite(self.share_tol) or self.share_tol <= 0.0:
            raise ValueError("share_tol must be a finite positive scalar.")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer.")

    def _invert_demand(self, sigma):
        """
        Berry inversion:
            (sjt, s0t, pjt, sigma) -> delta_jt

        Uses warm start and damping if available.
        """
        delta = invert_all_markets(
            sjt=self.sjt,
            s0t=self.s0t,
            pjt=self.pjt,
            sigma=float(sigma),
            v_draws=self.v_draws,
            delta_init=self._delta_warm_start,
            damping=self.damping,
            tol=self.tol,
            share_tol=self.share_tol,
            max_iter=self.max_iter,
        )
        self._delta_warm_start = delta
        return delta

    def _estimate_beta(self, delta):
        """
        IV regression:
            delta_jt = X_jt beta + xi_jt
        where X_jt = [pjt, wjt] (constructed internally).
        """
        delta_vec = delta.reshape(-1, 1)  # (n_obs, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)  # (n_obs, 2)
        Z = self.Zjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kz)

        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        Pz = Z @ ZTZ_inv @ Z.T

        XPZX = X.T @ Pz @ X
        XPZy = X.T @ Pz @ delta_vec

        beta_hat = np.linalg.pinv(XPZX) @ XPZy  # (2,1)
        return beta_hat

    def _compute_E_hat(self, delta, beta_hat):
        """
        Recovered demand shocks:
            xi_hat_jt = delta_jt - X_jt beta_hat
        """
        delta_vec = delta.reshape(-1, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)

        E_vec = delta_vec - X @ beta_hat
        return E_vec.reshape(self.sjt.shape)

    def _moments_and_omega(self, E_hat):
        """
        Compute:
          - per-observation moments m_i = z_i * xi_i
          - sample mean g_bar = mean_i m_i
          - moment covariance Omega_hat = (m'm)/n
        """
        E_vec = E_hat.reshape(-1, 1)  # (n, 1)
        Z = self.Zjt.reshape(E_vec.shape[0], -1)  # (n, Kz)

        m = Z * E_vec  # (n, Kz)
        g_bar = m.mean(axis=0)  # (Kz,)

        m_centered = m - g_bar  # broadcast (n, Kz) - (Kz,)
        Omega_hat = (m_centered.T @ m_centered) / float(m.shape[0])  # (Kz, Kz)

        return g_bar, Omega_hat

    def _gmm_objective(self, sigma, W):
        """
        Q(sigma) = g_bar(sigma)' W g_bar(sigma),
        where g_bar(sigma) = mean_i [ z_i * xi_i(sigma) ].
        """
        delta = self._invert_demand(sigma)
        beta_hat = self._estimate_beta(delta)
        # Split coefficients: X = [p, w]
        self.beta_p_hat = float(beta_hat[0, 0])
        self.beta_w_hat = float(beta_hat[1, 0])

        E_hat = self._compute_E_hat(delta, beta_hat)

        g_bar, _ = self._moments_and_omega(E_hat)
        return float(g_bar @ W @ g_bar)

    def _safe_gmm_objective(self, sigma, W):
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0 or sigma > self.sigma_max:
            return self.fail_penalty

        try:
            q = self._gmm_objective(sigma, W)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            return self.fail_penalty

        if (q is None) or (not np.isfinite(q)):
            return self.fail_penalty

        return float(q)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, sigma_init, sigma_min=1e-3, sigma_max=1.0, grid_step=25):
        """
        Two-step GMM:

        Step 1:
          - W1 = (Z'Z)^+ and sigma via grid + Nelder–Mead.

        Step 2:
          - Omega_hat at sigma_hat_1, W2 = Omega_hat^+, re-optimize sigma via NM.

        Stores:
          - self.sigma_hat, self.beta_hat, self.E_hat
        """
        sigma_init = float(sigma_init)
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        grid_step = int(grid_step)

        sigma_max_grid = min(sigma_max, self.sigma_max)

        Z = self.Zjt.reshape(-1, self.Zjt.shape[2])
        W1 = np.linalg.pinv(Z.T @ Z)

        # Grid search (step 1)
        sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max_grid), grid_step)
        best = {"Q": np.inf, "sigma": None}
        for s in sigmas:
            q = self._safe_gmm_objective(float(s), W1)
            if np.isfinite(q) and (q < best["Q"]) and (q < self.fail_penalty):
                best["Q"] = float(q)
                best["sigma"] = float(s)

        sigma_start = best["sigma"]
        if sigma_start is None:
            sigma_start = min(max(sigma_init, sigma_min), sigma_max_grid)

        # Step 1: Nelder–Mead on log-sigma
        best1 = {"Q": np.inf, "sigma": None}

        def obj1(theta_vec):
            sigma = float(np.exp(theta_vec[0]))
            q = self._safe_gmm_objective(sigma, W1)
            if np.isfinite(q) and (q < best1["Q"]) and (q < self.fail_penalty):
                best1["Q"] = float(q)
                best1["sigma"] = float(sigma)
            return float(q)

        res1 = minimize(
            fun=obj1, x0=np.array([np.log(sigma_start)]), method="Nelder-Mead"
        )

        sigma_hat_1 = best1["sigma"]
        if sigma_hat_1 is None:
            sigma_hat_1 = float(np.exp(res1.x[0]))
        sigma_hat_1 = float(min(max(sigma_hat_1, sigma_min), self.sigma_max))

        # Build W2
        delta_1 = self._invert_demand(sigma_hat_1)
        beta_1 = self._estimate_beta(delta_1)
        E_1 = self._compute_E_hat(delta_1, beta_1)

        _, Omega_hat = self._moments_and_omega(E_1)
        W2 = np.linalg.pinv(Omega_hat)

        # Step 2: Nelder–Mead on log-sigma
        best2 = {"Q": np.inf, "sigma": None}

        def obj2(theta_vec):
            sigma = float(np.exp(theta_vec[0]))
            q = self._safe_gmm_objective(sigma, W2)
            if np.isfinite(q) and (q < best2["Q"]) and (q < self.fail_penalty):
                best2["Q"] = float(q)
                best2["sigma"] = float(sigma)
            return float(q)

        res2 = minimize(
            fun=obj2, x0=np.array([np.log(sigma_hat_1)]), method="Nelder-Mead"
        )

        sigma_hat_2 = best2["sigma"]
        if sigma_hat_2 is None:
            sigma_hat_2 = float(np.exp(res2.x[0]))
        sigma_hat_2 = float(min(max(sigma_hat_2, sigma_min), self.sigma_max))

        # Final recomputation at sigma_hat_2
        self.sigma_hat = sigma_hat_2

        delta_hat = self._invert_demand(self.sigma_hat)
        self.beta_hat = self._estimate_beta(delta_hat)
        self.E_hat = self._compute_E_hat(delta_hat, self.beta_hat)

        print("[BLP] Fit complete")
        self.success = True

    def get_results(self):
        """
        Keys:
          - "success":   bool
          - "sigma_hat": scalar float or None
          - "beta_hat":  (2, 1) array or None   (beta_p, beta_w)
          - "E_hat":     (T, J) array or None
        """
        return {
            "success": self.success,
            "sigma_hat": self.sigma_hat,
            "beta_p_hat": self.beta_p_hat,  # scalar
            "beta_w_hat": self.beta_w_hat,  # scalar
            "E_hat": self.E_hat,
        }
