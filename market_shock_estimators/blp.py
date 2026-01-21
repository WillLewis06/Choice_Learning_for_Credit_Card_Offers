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
        damping=0.25,
    ):
        print("[BLP] Initializing estimator")

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

        print("[BLP] Data stored successfully")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invert_demand(self, sigma):
        """
        Berry inversion:
            (sjt, s0t, pjt, sigma) -> delta_jt

        Uses warm start and damping if available.
        """
        print(f"[BLP] Inverting demand at sigma = {sigma:.6f}")

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

        print("[BLP] Demand inversion completed")
        return delta

    def _estimate_beta(self, delta):
        """
        IV regression:
            delta_jt = X_jt beta + E_jt
        """
        print("[BLP] Estimating beta via IV regression")

        delta_vec = delta.reshape(-1, 1)  # (n_obs, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kx)
        Z = self.Zjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kz)

        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        Pz = Z @ ZTZ_inv @ Z.T

        XPZX = X.T @ Pz @ X
        XPZy = X.T @ Pz @ delta_vec

        beta_hat = np.linalg.pinv(XPZX) @ XPZy  # (Kx,1)

        print("[BLP] beta_hat estimated")
        return beta_hat

    def _compute_E_hat(self, delta, beta_hat):
        """
        Compute recovered demand shocks:
            E_hat_jt = delta_jt - X_jt beta_hat
        """
        print("[BLP] Computing recovered shocks E_hat")

        delta_vec = delta.reshape(-1, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)

        E_vec = delta_vec - X @ beta_hat
        E_hat = E_vec.reshape(self.sjt.shape)

        print("[BLP] E_hat computed")
        return E_hat

    def _gmm_objective(self, sigma):
        """
        One-step GMM objective:
            Q(sigma) = g_bar(sigma)' W g_bar(sigma),
            g_bar(sigma) = mean(Z_jt * E_hat_jt(sigma)),
            W = (Z'Z)^{+}.
        """
        print(f"[BLP] Evaluating GMM objective at sigma = {sigma:.6f}")

        delta = self._invert_demand(sigma)
        beta_hat = self._estimate_beta(delta)
        E_hat = self._compute_E_hat(delta, beta_hat)

        E_vec = E_hat.reshape(-1, 1)
        Z = self.Zjt.reshape(E_vec.shape[0], -1)

        g_bar = (Z * E_vec).mean(axis=0)  # (K,)

        W = np.linalg.pinv(Z.T @ Z)
        Q = float(g_bar @ W @ g_bar)

        print(f"[BLP] GMM objective value = {Q:.6e}")
        return Q

    def _safe_gmm_objective(self, sigma):
        """
        Safe wrapper around _gmm_objective:
          - Enforces sigma bounds.
          - Converts inversion/linear-algebra failures into a large penalty.
          - Converts NaN/inf objective values into a large penalty.
        """
        sigma = float(sigma)

        if not np.isfinite(sigma) or sigma <= 0.0 or sigma > self.sigma_max:
            return self.fail_penalty

        try:
            q = self._gmm_objective(sigma)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            return self.fail_penalty

        if (q is None) or (not np.isfinite(q)):
            return self.fail_penalty

        return float(q)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, sigma_init):
        """
        Estimate sigma via GMM and store implied E_hat_jt.

        Robustness behavior:
          - Objective evaluation never throws due to inversion failure; it returns a penalty.
          - sigma is restricted to (0, sigma_max] inside the objective.
          - The final reported sigma_hat is the best feasible sigma encountered.
          - Inversion uses warm starts and damping.
        """
        sigma_init = float(sigma_init)
        print(f"[BLP] Starting GMM optimization from sigma_init = {sigma_init:.6f}")

        best = {"Q": np.inf, "sigma": None}

        def objective_theta(theta_vec):
            sigma = float(np.exp(theta_vec[0]))
            q = self._safe_gmm_objective(sigma)
            if np.isfinite(q) and (q < best["Q"]) and (q < self.fail_penalty):
                best["Q"] = float(q)
                best["sigma"] = float(sigma)
            return float(q)

        res = minimize(
            fun=objective_theta,
            x0=np.array([np.log(sigma_init)]),
            method="Nelder-Mead",
        )

        if best["sigma"] is not None:
            self.sigma_hat = best["sigma"]
        else:
            self.sigma_hat = float(np.exp(res.x[0]))
            if not np.isfinite(self.sigma_hat) or self.sigma_hat <= 0.0:
                self.sigma_hat = min(max(sigma_init, 1e-12), self.sigma_max)
            if self.sigma_hat > self.sigma_max:
                self.sigma_hat = self.sigma_max

        print(f"[BLP] Optimization completed: sigma_hat = {self.sigma_hat:.6f}")

        # Ensure we don't rely on any warm start from failed objective evaluations:
        # recompute delta_hat at the chosen sigma_hat (this call will refresh warm start on success)
        delta_hat = self._invert_demand(self.sigma_hat)
        self.beta_hat = self._estimate_beta(delta_hat)
        self.E_hat = self._compute_E_hat(delta_hat, self.beta_hat)

        print("[BLP] Estimation pipeline completed successfully")

    def get_E_hat(self):
        """
        Return recovered demand shocks E_hat_jt.
        """
        if self.E_hat is None:
            raise RuntimeError("Estimator must be fit() before calling get_E_hat().")

        print("[BLP] Returning E_hat_jt")
        return self.E_hat
