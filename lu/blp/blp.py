"""
BLP-style demand estimator used in the Lu simulation pipeline.

This module provides:
  - Simple instrument builders (strong vs weak instruments).
  - A minimal two-step GMM estimator for sigma in a random-coefficient logit.
  - Recovery of mean utilities via Berry inversion (contraction mapping).
  - IV regression of delta on X = [pjt, wjt] to recover (beta_p, beta_w).
  - Recovery of demand shocks E_hat = delta - X beta_hat.

What this estimator does (high level):
  1) For a candidate sigma, invert observed shares to get delta (mean utilities).
  2) Run IV regression: delta = X beta + E (E is the demand shock).
  3) Form moments: g_bar(sigma) = mean(z * E).
  4) Minimize Q(sigma) = g_bar' W g_bar using a two-step weighting matrix.

This is a simulation-oriented implementation: it prioritizes clarity and stable
behavior (warm starts, damping, penalties) over econometric completeness.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from lu.blp.inversion import invert_all_markets


def build_strong_IVs(wjt, ujt):
    """Build 'strong' instruments using both wjt and a cost shifter ujt.

    Returns Zjt with last axis holding instrument components:
      Z_strong = [1, wjt, wjt^2, ujt, ujt^2]

    This matches the simulation-side instrument construction used in the Lu-style
    experiments: ujt is treated as an exogenous cost shifter correlated with pjt.
    """
    wjt = np.asarray(wjt, dtype=float)
    ujt = np.asarray(ujt, dtype=float)
    ones = np.ones_like(wjt)
    return np.stack([ones, wjt, wjt**2, ujt, ujt**2], axis=2)


def build_weak_IVs(wjt):
    """Build 'weak' instruments as polynomials of wjt only.

    Returns Zjt with last axis holding instrument components:
      Z_weak = [1, wjt, wjt^2, wjt^3, wjt^4]

    These instruments are deliberately weaker in the simulation design: they
    contain no additional excluded shifter like ujt.
    """
    wjt = np.asarray(wjt, dtype=float)
    ones = np.ones_like(wjt)
    return np.stack([ones, wjt, wjt**2, wjt**3, wjt**4], axis=2)


class BLPEstimator:
    """Minimal BLP-style estimator for the simulation environment.

    Demand-side specification (aligned with the simulation regressors):
      delta_jt = beta_p * pjt + beta_w * wjt + E_hat_jt

    Implementation outline:
      - For each candidate sigma, recover delta via Berry inversion.
      - Estimate beta by IV regression using instruments Zjt.
      - Construct moment conditions from Zjt and E_hat.
      - Minimize the GMM objective over sigma via a grid + Nelder–Mead.

    Notes:
      - X is constructed internally as [pjt, wjt] (no constant).
      - All computations are done with numpy; inversion is imported from
        `market_shock_estimators.inversion`.
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
        """Store data, configure inversion controls, and validate inputs.

        Key stored objects:
          - Observed shares: sjt (inside) and s0t (outside)
          - Regressors: pjt, wjt, and instruments Zjt
          - Simulation draws used to integrate the random coefficient on price

        The estimator keeps a warm-start delta from the last successful inversion
        to speed up repeated objective evaluations during optimization.
        """
        self.sjt = np.asarray(sjt, dtype=float)
        self.s0t = np.asarray(s0t, dtype=float)
        self.pjt = np.asarray(pjt, dtype=float)
        self.wjt = np.asarray(wjt, dtype=float)
        self.Zjt = np.asarray(Zjt, dtype=float)

        self.n_draws = int(n_draws)
        self.seed = int(seed)

        # Demand regressors: Xjt[t,j,:] = [pjt[t,j], wjt[t,j]]
        self.Xjt = np.stack([self.pjt, self.wjt], axis=2)

        # Fixed simulation draws for the RC integration (owned by the estimator).
        rng = np.random.default_rng(self.seed)
        self.v_draws = rng.standard_normal(self.n_draws).astype(float)

        # Inversion controls (shared by all objective evaluations).
        self.tol = float(tol)
        self.share_tol = float(share_tol)
        self.max_iter = int(max_iter)
        self.damping = float(damping)

        # A global cap on sigma (separate from per-fit bounds).
        self.sigma_max = float(sigma_max)

        # Penalty value used when sigma is invalid or inversion fails.
        self.fail_penalty = float(fail_penalty)

        # Warm-start for inversion once the first successful delta is computed.
        self._delta_warm_start = None

        # Active sigma bounds (set within fit()).
        self._sigma_lo = None
        self._sigma_hi = None

        # Outputs populated after fit().
        self.sigma_hat = None
        self.beta_hat = None
        self.beta_p_hat = None
        self.beta_w_hat = None
        self.E_hat = None
        self.success = False

        self._check_inputs()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_inputs(self):
        """Validate shapes, finiteness, and basic share identities."""
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

        # Shape consistency across observed objects.
        if self.pjt.shape != (T, J):
            raise ValueError(f"pjt must have shape {(T, J)}. Got {self.pjt.shape}.")
        if self.wjt.shape != (T, J):
            raise ValueError(f"wjt must have shape {(T, J)}. Got {self.wjt.shape}.")
        if self.s0t.shape != (T,):
            raise ValueError(f"s0t must have shape {(T,)}. Got {self.s0t.shape}.")
        if self.Xjt.shape != (T, J, 2):
            raise ValueError(f"Xjt must have shape (T,J,2). Got {self.Xjt.shape}.")
        if self.Zjt.shape[0] != T or self.Zjt.shape[1] != J:
            raise ValueError(
                f"Zjt must have shape (T,J,Kz) with T={T}, J={J}. Got {self.Zjt.shape}."
            )
        if self.v_draws.size < 1:
            raise ValueError("n_draws must be >= 1 (v_draws non-empty).")

        # All inputs must be finite.
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

        # Basic share validity.
        if np.any(self.sjt <= 0.0):
            raise ValueError("sjt must be strictly positive.")
        if np.any(self.s0t <= 0.0):
            raise ValueError("s0t must be strictly positive.")
        if np.any(self.sjt >= 1.0):
            raise ValueError("sjt must be strictly less than 1.")
        if np.any(self.s0t >= 1.0):
            raise ValueError("s0t must be strictly less than 1.")

        # Share identity: s0t + sum_j sjt = 1 per market.
        share_id_tol = 1e-8
        share_err = np.max(np.abs(self.s0t + self.sjt.sum(axis=1) - 1.0))
        if not np.isfinite(share_err) or share_err > share_id_tol:
            raise ValueError(
                "Share identity violated: "
                f"max|s0t+sum(sjt)-1|={share_err:.3e} > {share_id_tol:.1e}"
            )

        # Parameter sanity.
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

    # ------------------------------------------------------------------
    # Core estimation primitives
    # ------------------------------------------------------------------

    def _invert_demand(self, sigma):
        """Recover delta for all markets via Berry inversion.

        This wraps `invert_all_markets(...)` and maintains a warm-start delta so
        repeated objective evaluations are faster and more stable.
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
        """Estimate (beta_p, beta_w) by IV regression given delta.

        This runs the 2SLS estimator:
          beta_hat = (X' Pz X)^+ (X' Pz delta)

        where:
          X is stacked [pjt, wjt] across all (t,j),
          Z is stacked instruments across all (t,j),
          Pz = Z (Z'Z)^+ Z'.

        Returns:
            beta_hat: Array of shape (2, 1) with entries [beta_p, beta_w].
        """
        delta_vec = delta.reshape(-1, 1)  # (n_obs, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)  # (n_obs, 2)
        Z = self.Zjt.reshape(delta_vec.shape[0], -1)  # (n_obs, Kz)

        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        Pz = Z @ ZTZ_inv @ Z.T

        XPZX = X.T @ Pz @ X
        XPZy = X.T @ Pz @ delta_vec

        beta_hat = np.linalg.pinv(XPZX) @ XPZy
        return beta_hat

    def _compute_E_hat(self, delta, beta_hat):
        """Compute recovered demand shocks E_hat = delta - X beta_hat."""
        delta_vec = delta.reshape(-1, 1)
        X = self.Xjt.reshape(delta_vec.shape[0], -1)

        E_vec = delta_vec - X @ beta_hat
        return E_vec.reshape(self.sjt.shape)

    def _moments_and_omega(self, E_hat):
        """Compute mean moments and the moment covariance estimate.

        Moments are formed per observation i=(t,j):
          m_i = z_i * E_i

        Returned objects:
          - g_bar: sample mean of m_i across i
          - Omega_hat: sample covariance of m_i across i

        Returns:
            g_bar: Array of shape (Kz,).
            Omega_hat: Array of shape (Kz, Kz).
        """
        E_vec = E_hat.reshape(-1, 1)  # (n, 1)
        Z = self.Zjt.reshape(E_vec.shape[0], -1)  # (n, Kz)

        m = Z * E_vec  # (n, Kz)
        g_bar = m.mean(axis=0)  # (Kz,)

        m_centered = m - g_bar
        Omega_hat = (m_centered.T @ m_centered) / float(m.shape[0])  # (Kz, Kz)
        return g_bar, Omega_hat

    def _gmm_objective(self, sigma, W):
        """Compute Q(sigma) = g_bar(sigma)' W g_bar(sigma).

        This method also updates cached scalars beta_p_hat and beta_w_hat as a
        convenience for inspection during optimization.
        """
        delta = self._invert_demand(sigma)
        beta_hat = self._estimate_beta(delta)

        # Store split coefficients: X = [p, w].
        self.beta_p_hat = float(beta_hat[0, 0])
        self.beta_w_hat = float(beta_hat[1, 0])

        E_hat = self._compute_E_hat(delta, beta_hat)
        g_bar, _ = self._moments_and_omega(E_hat)
        return float(g_bar @ W @ g_bar)

    def _safe_gmm_objective(self, sigma, W):
        """Evaluate the GMM objective with hard penalties on failure.

        This is used inside the optimizer. Any invalid sigma, out-of-bounds sigma,
        inversion failure, or linear algebra error returns `fail_penalty`.
        """
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return self.fail_penalty

        # Active bounds are set by fit().
        sigma_lo = self._sigma_lo
        sigma_hi = self._sigma_hi
        if sigma_lo is not None and sigma < float(sigma_lo):
            return self.fail_penalty
        if sigma_hi is not None and sigma > float(sigma_hi):
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
        """Fit sigma via two-step GMM and compute final beta and E_hat.

        Step 1:
          - Use W1 = (Z'Z)^+.
          - Choose a starting value by log-spaced grid search over [sigma_min, sigma_max].
          - Refine with Nelder–Mead over log(sigma).

        Step 2:
          - Recompute E_hat at sigma_hat_1.
          - Estimate Omega_hat from m_i = z_i * E_i and set W2 = Omega_hat^+.
          - Re-optimize sigma with Nelder–Mead over log(sigma).

        Results stored on the instance:
          - sigma_hat: final sigma estimate (step 2)
          - beta_hat: (2,1) estimate for [beta_p, beta_w]
          - E_hat: (T,J) recovered demand shocks
        """
        sigma_init = float(sigma_init)
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        grid_step = int(grid_step)

        # Apply a hard cap (self.sigma_max) to avoid exploring unstable regions.
        sigma_lo = sigma_min
        sigma_hi = min(sigma_max, self.sigma_max)

        if not np.isfinite(sigma_lo) or sigma_lo <= 0.0:
            raise ValueError("sigma_min must be a finite positive scalar.")
        if not np.isfinite(sigma_hi) or sigma_hi <= 0.0:
            raise ValueError("sigma_max must be a finite positive scalar.")
        if sigma_hi < sigma_lo:
            raise ValueError("sigma_max must be >= sigma_min after applying caps.")

        self._sigma_lo = float(sigma_lo)
        self._sigma_hi = float(sigma_hi)

        # Step-1 weighting matrix W1 = (Z'Z)^+ using stacked instruments.
        Z = self.Zjt.reshape(-1, self.Zjt.shape[2])
        W1 = np.linalg.pinv(Z.T @ Z)

        # -------------------------
        # Step 1: grid search
        # -------------------------
        sigmas = np.logspace(
            np.log10(self._sigma_lo), np.log10(self._sigma_hi), grid_step
        )
        best = {"Q": np.inf, "sigma": None}
        for s in sigmas:
            q = self._safe_gmm_objective(float(s), W1)
            if np.isfinite(q) and (q < best["Q"]) and (q < self.fail_penalty):
                best["Q"] = float(q)
                best["sigma"] = float(s)

        sigma_start = best["sigma"]
        if sigma_start is None:
            sigma_start = min(max(sigma_init, self._sigma_lo), self._sigma_hi)

        # -------------------------
        # Step 1: Nelder–Mead on log(sigma)
        # -------------------------
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
        sigma_hat_1 = float(min(max(sigma_hat_1, self._sigma_lo), self._sigma_hi))

        # Build W2 at sigma_hat_1.
        delta_1 = self._invert_demand(sigma_hat_1)
        beta_1 = self._estimate_beta(delta_1)
        E_1 = self._compute_E_hat(delta_1, beta_1)

        _, Omega_hat = self._moments_and_omega(E_1)
        W2 = np.linalg.pinv(Omega_hat)

        # -------------------------
        # Step 2: Nelder–Mead on log(sigma)
        # -------------------------
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
        sigma_hat_2 = float(min(max(sigma_hat_2, self._sigma_lo), self._sigma_hi))

        # -------------------------
        # Final recomputation at sigma_hat_2
        # -------------------------
        self.sigma_hat = sigma_hat_2

        delta_hat = self._invert_demand(self.sigma_hat)
        self.beta_hat = self._estimate_beta(delta_hat)
        self.E_hat = self._compute_E_hat(delta_hat, self.beta_hat)

        print("[BLP] Fit complete")
        self.success = True

    def get_results(self):
        """Return the minimal result dictionary used by downstream code."""
        return {
            "success": self.success,
            "sigma_hat": self.sigma_hat,
            "beta_p_hat": self.beta_p_hat,
            "beta_w_hat": self.beta_w_hat,
            "E_hat": self.E_hat,
        }
