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

    Pipeline:
      - For each candidate sigma, invert demand to recover delta (Berry contraction).
      - Estimate beta via IV: delta = X beta + xi.
      - Form moments g = mean(Z * xi) and minimize g' W g.

    Weighting:
      - One-step GMM with first-step 2SLS-style weighting W = (Z'Z)^{+}
        (Moore–Penrose pseudoinverse for robustness).

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
        Xjt,
        Zjt,
        v_draws,
        *,
        tol=1e-8,
        share_tol=1e-10,
        max_iter=10_000,
        sigma_max=5.0,
        fail_penalty=1e12,
        damping=1,
    ):

        self.sjt = np.asarray(sjt, dtype=float)
        self.s0t = np.asarray(s0t, dtype=float)
        self.pjt = np.asarray(pjt, dtype=float)
        self.Xjt = np.asarray(Xjt, dtype=float)
        self.Zjt = np.asarray(Zjt, dtype=float)

        self.v_draws = np.asarray(v_draws, dtype=float)

        self.tol = tol
        self.share_tol = share_tol
        self.max_iter = max_iter

        self.sigma_max = float(sigma_max)
        self.fail_penalty = float(fail_penalty)

        self.damping = float(damping)
        self._delta_warm_start = None  # shape (T,J) after first successful inversion

        self.sigma_hat = None
        self.beta_hat = None
        self.E_hat = None

        # fit completed bool
        self.success = False

        self._check_inputs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_inputs(self):
        """
        Strict panel-level input validation.

        Ensures:
          - Correct array dimensions and mutually consistent shapes
          - Finite values (no NaN/inf)
          - Shares strictly in (0,1)
          - Per-market share identity: s0t + sum_j sjt == 1 (within tolerance)
          - v_draws is 1D and non-empty
          - basic numeric parameter validity (damping, sigma_max, etc.)
        """
        # -------------------------
        # Dimensionality checks
        # -------------------------
        if self.sjt.ndim != 2:
            raise ValueError(f"sjt must be 2D (T,J). Got ndim={self.sjt.ndim}.")
        if self.pjt.ndim != 2:
            raise ValueError(f"pjt must be 2D (T,J). Got ndim={self.pjt.ndim}.")
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
        if self.s0t.shape != (T,):
            raise ValueError(f"s0t must have shape {(T,)}. Got {self.s0t.shape}.")
        if self.Xjt.shape[0] != T or self.Xjt.shape[1] != J:
            raise ValueError(
                f"Xjt must have shape (T,J,Kx) with T={T}, J={J}. Got {self.Xjt.shape}."
            )
        if self.Zjt.shape[0] != T or self.Zjt.shape[1] != J:
            raise ValueError(
                f"Zjt must have shape (T,J,Kz) with T={T}, J={J}. Got {self.Zjt.shape}."
            )
        if self.v_draws.size < 1:
            raise ValueError("v_draws must be non-empty.")

        # -------------------------
        # Finite-value checks
        # -------------------------
        for name, arr in [
            ("sjt", self.sjt),
            ("s0t", self.s0t),
            ("pjt", self.pjt),
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

        # Share identity check: s0t + sum_j sjt == 1
        share_id_tol = 1e-8
        share_err = np.max(np.abs(self.s0t + self.sjt.sum(axis=1) - 1.0))
        if not np.isfinite(share_err) or share_err > share_id_tol:
            raise ValueError(
                f"Share identity violated: max|s0t+sum(sjt)-1|={share_err:.3e} > {share_id_tol:.1e}"
            )

        # -------------------------
        # Parameter sanity checks
        # -------------------------
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

        # Update warm start only on success
        self._delta_warm_start = delta

        return delta

    def _estimate_beta(self, delta):
        """
        IV regression:
            delta_jt = X_jt beta + E_jt
        """

        delta_vec = delta.reshape(-1, 1)  # (n_obs, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kx)
        Z = self.Zjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kz)

        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        Pz = Z @ ZTZ_inv @ Z.T

        XPZX = X.T @ Pz @ X
        XPZy = X.T @ Pz @ delta_vec

        beta_hat = np.linalg.pinv(XPZX) @ XPZy  # (Kx,1)

        return beta_hat

    def _compute_E_hat(self, delta, beta_hat):
        """
        Compute recovered demand shocks:
            E_hat_jt = delta_jt - X_jt beta_hat
        """

        delta_vec = delta.reshape(-1, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)

        E_vec = delta_vec - X @ beta_hat
        E_hat = E_vec.reshape(self.sjt.shape)

        return E_hat

    def _moments_and_omega(self, E_hat):
        """
        Compute:
          - per-observation moments m_i = z_i * xi_i
          - sample mean g_bar = mean_i m_i
          - moment covariance Omega_hat = (m'm)/n

        Here xi_i is the recovered demand shock at (t,j).
        """
        E_vec = E_hat.reshape(-1, 1)  # (n, 1)
        Z = self.Zjt.reshape(E_vec.shape[0], -1)  # (n, K)

        m = Z * E_vec  # (n, K)
        g_bar = m.mean(axis=0)  # (K,)
        Omega_hat = (m.T @ m) / float(m.shape[0])  # (K, K)

        return g_bar, Omega_hat

    def _gmm_objective(self, sigma, W):
        """
        GMM objective:
            Q(sigma) = g_bar(sigma)' W g_bar(sigma)

        where g_bar(sigma) = mean_i [ z_i * xi_i(sigma) ].
        """
        delta = self._invert_demand(sigma)
        beta_hat = self._estimate_beta(delta)
        E_hat = self._compute_E_hat(delta, beta_hat)

        g_bar, _ = self._moments_and_omega(E_hat)

        Q = float(g_bar @ W @ g_bar)
        return Q

    def _safe_gmm_objective(self, sigma, W):
        """
        Safe wrapper around _gmm_objective(sigma, W).
        """
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
        Two-step GMM with a single grid search (first-step only).

        Step 1:
          - Use W1 = (Z'Z)^+ (your current default) and find sigma via grid+NM.

        Step 2:
          - Compute Omega_hat at sigma_hat_step1, set W2 = Omega_hat^+,
            re-optimize sigma via NM (no second grid).

        Stores:
          - self.sigma_hat, self.beta_hat, self.E_hat
        """
        sigma_init = float(sigma_init)
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        grid_step = int(grid_step)

        # Cap the grid range by the estimator hard bound
        sigma_max_grid = min(sigma_max, self.sigma_max)

        # Precompute first-step weighting
        Z = self.Zjt.reshape(-1, self.Zjt.shape[2])
        W1 = np.linalg.pinv(Z.T @ Z)

        # -----------------------------
        # Grid search (first-step only)
        # -----------------------------
        sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max_grid), grid_step)

        best = {"Q": np.inf, "sigma": None}

        for s in sigmas:
            q = self._safe_gmm_objective(float(s), W1)
            if np.isfinite(q) and (q < best["Q"]) and (q < self.fail_penalty):
                best["Q"] = float(q)
                best["sigma"] = float(s)

        if best["sigma"] is not None:
            sigma_start = best["sigma"]
        else:
            # fallback if every grid point failed
            sigma_start = min(max(sigma_init, sigma_min), sigma_max_grid)

        # -----------------------------
        # Step 1: Nelder–Mead with W1
        # -----------------------------
        best1 = {"Q": np.inf, "sigma": None}

        def obj1(theta_vec):
            sigma = float(np.exp(theta_vec[0]))
            q = self._safe_gmm_objective(sigma, W1)
            if np.isfinite(q) and (q < best1["Q"]) and (q < self.fail_penalty):
                best1["Q"] = float(q)
                best1["sigma"] = float(sigma)
            return float(q)

        res1 = minimize(
            fun=obj1,
            x0=np.array([np.log(sigma_start)]),
            method="Nelder-Mead",
        )

        sigma_hat_1 = best1["sigma"]
        if sigma_hat_1 is None:
            sigma_hat_1 = float(np.exp(res1.x[0]))
        sigma_hat_1 = float(min(max(sigma_hat_1, sigma_min), self.sigma_max))

        # -----------------------------
        # Build W2 from Omega_hat at sigma_hat_1
        # -----------------------------
        delta_1 = self._invert_demand(sigma_hat_1)
        beta_1 = self._estimate_beta(delta_1)
        E_1 = self._compute_E_hat(delta_1, beta_1)

        _, Omega_hat = self._moments_and_omega(E_1)
        W2 = np.linalg.pinv(Omega_hat)

        # -----------------------------
        # Step 2: Nelder–Mead with W2 (no grid)
        # -----------------------------
        best2 = {"Q": np.inf, "sigma": None}

        def obj2(theta_vec):
            sigma = float(np.exp(theta_vec[0]))
            q = self._safe_gmm_objective(sigma, W2)
            if np.isfinite(q) and (q < best2["Q"]) and (q < self.fail_penalty):
                best2["Q"] = float(q)
                best2["sigma"] = float(sigma)
            return float(q)

        res2 = minimize(
            fun=obj2,
            x0=np.array([np.log(sigma_hat_1)]),
            method="Nelder-Mead",
        )

        sigma_hat_2 = best2["sigma"]
        if sigma_hat_2 is None:
            sigma_hat_2 = float(np.exp(res2.x[0]))
        sigma_hat_2 = float(min(max(sigma_hat_2, sigma_min), self.sigma_max))

        # -----------------------------
        # Final recomputation at sigma_hat_2
        # -----------------------------
        self.sigma_hat = sigma_hat_2

        delta_hat = self._invert_demand(self.sigma_hat)
        self.beta_hat = self._estimate_beta(delta_hat)
        self.E_hat = self._compute_E_hat(delta_hat, self.beta_hat)

        print(f"[BLP] Fit complete")
        self.success = True

    def get_results(self):
        """
        Return estimator outputs in a single, consistent bundle.

        Keys:
          - "success":   bool
          - "sigma_hat": scalar float or None
          - "beta_hat":  (Kx, 1) array or None
          - "E_hat":     (T, J) array or None
        """
        return {
            "success": self.success,
            "sigma_hat": self.sigma_hat,
            "beta_hat": self.beta_hat,
            "E_hat": self.E_hat,
        }
