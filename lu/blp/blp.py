"""
Minimal BLP-style demand estimator for simulation use.

Implements:
  - Instrument construction helpers.
  - Berry inversion to recover mean utilities delta given sigma.
  - 2SLS regression of delta on X = [pjt, wjt] using instruments Zjt.
  - Two-step GMM over sigma using moments E[z * E_hat] = 0.

Contract:
  - Data arrays are assumed to be produced internally and already consistent.
  - The `config` mapping is assumed to be validated upstream (no defaults here).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from lu.blp.inversion import invert_all_markets


def build_strong_IVs(wjt: np.ndarray, ujt: np.ndarray) -> np.ndarray:
    """Return strong instruments Z = [1, w, w^2, u, u^2] with shape (T, J, 5)."""
    w = np.asarray(wjt)
    u = np.asarray(ujt)
    ones = np.ones_like(w)
    return np.stack([ones, w, w**2, u, u**2], axis=2)


def build_weak_IVs(wjt: np.ndarray) -> np.ndarray:
    """Return weak instruments Z = [1, w, w^2, w^3, w^4] with shape (T, J, 5)."""
    w = np.asarray(wjt)
    ones = np.ones_like(w)
    return np.stack([ones, w, w**2, w**3, w**4], axis=2)


class BLPEstimator:
    """Two-step GMM estimator for sigma with IV recovery of (beta_p, beta_w)."""

    def __init__(
        self,
        sjt: np.ndarray,
        s0t: np.ndarray,
        pjt: np.ndarray,
        wjt: np.ndarray,
        Zjt: np.ndarray,
        config: dict,
    ) -> None:
        """
        Store data/config and precompute objects reused across sigma evaluations.

        Args:
            sjt: Inside shares, shape (T, J).
            s0t: Outside shares, shape (T,).
            pjt: Prices, shape (T, J).
            wjt: Observed characteristic, shape (T, J).
            Zjt: Instruments, shape (T, J, Kz).
            config: Validated configuration mapping (validated upstream).
        """
        # Core data arrays.
        self.sjt = np.asarray(sjt)
        self.s0t = np.asarray(s0t)
        self.pjt = np.asarray(pjt)
        self.wjt = np.asarray(wjt)
        self.Zjt = np.asarray(Zjt)

        # Validated config mapping (no validation here by design).
        self.config = config

        # Demand regressors (no constant): X[t, j, :] = [pjt, wjt].
        self.Xjt = np.stack([self.pjt, self.wjt], axis=2)

        # Fixed simulation draws for integrating the random coefficient on price.
        rng = np.random.default_rng(self.config["seed"])
        self.v_draws = rng.standard_normal(self.config["n_draws"])

        # Deterministic logit starting values for each market: log(sjt) - log(s0t).
        self._delta_init0 = np.log(self.sjt) - np.log(self.s0t)[:, None]
        self._delta_warm_start = self._delta_init0

        # Active sigma bounds used by the safe objective during fit().
        self._sigma_lo: float | None = None
        self._sigma_hi: float | None = None

        # Outputs populated after fit().
        self.success = False
        self.sigma_hat: float | None = None
        self.beta_hat: np.ndarray | None = None
        self.beta_p_hat: float | None = None
        self.beta_w_hat: float | None = None
        self.E_hat: np.ndarray | None = None

    def _invert_demand(self, sigma: float) -> np.ndarray:
        """Invert shares to recover delta for all markets at a given sigma."""
        # Berry contraction mapping (warm-started from the last successful delta).
        delta = invert_all_markets(
            sjt=self.sjt,
            pjt=self.pjt,
            sigma=float(sigma),
            v_draws=self.v_draws,
            delta_init=self._delta_warm_start,
            damping=self.config["damping"],
            tol=self.config["tol"],
            share_tol=self.config["share_tol"],
            max_iter=self.config["max_iter"],
        )
        self._delta_warm_start = delta
        return delta

    def _estimate_beta(self, delta: np.ndarray) -> np.ndarray:
        """Compute 2SLS beta for delta = X beta + E using instruments Z."""
        # Stack markets/products into a single regression.
        y = delta.reshape(-1, 1)  # (n, 1)
        X = self.Xjt.reshape(y.shape[0], -1)  # (n, 2)
        Z = self.Zjt.reshape(y.shape[0], -1)  # (n, Kz)

        # Closed-form 2SLS: beta = (X' Pz X)^-1 X' Pz y, with Pz = Z(Z'Z)^-1 Z'.
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        XTZ = X.T @ Z

        A = XTZ @ ZTZ_inv @ XTZ.T
        B = XTZ @ ZTZ_inv @ (Z.T @ y)
        return np.linalg.pinv(A) @ B  # (2, 1)

    def _compute_E_hat(self, delta: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
        """Compute E_hat = delta - X beta_hat with shape (T, J)."""
        y = delta.reshape(-1, 1)
        X = self.Xjt.reshape(y.shape[0], -1)
        E = y - X @ beta_hat
        return E.reshape(self.sjt.shape)

    def _moments_and_omega(self, E_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return g_bar = mean(z * E) and Omega_hat = cov(z * E)."""
        E = E_hat.reshape(-1, 1)  # (n, 1)
        Z = self.Zjt.reshape(E.shape[0], -1)  # (n, Kz)

        # Moment per observation: m_i = z_i * E_i.
        m = Z * E
        g_bar = m.mean(axis=0)

        # Sample covariance of the moment vector.
        m_centered = m - g_bar
        Omega_hat = (m_centered.T @ m_centered) / float(m.shape[0])
        return g_bar, Omega_hat

    def _gmm_objective(self, sigma: float, W: np.ndarray) -> float:
        """Return Q(sigma) = g_bar(sigma)' W g_bar(sigma)."""
        delta = self._invert_demand(float(sigma))
        beta_hat = self._estimate_beta(delta)
        E_hat = self._compute_E_hat(delta, beta_hat)
        g_bar, _ = self._moments_and_omega(E_hat)
        return float(g_bar @ W @ g_bar)

    def _safe_gmm_objective(self, sigma: float, W: np.ndarray) -> float:
        """Return the objective, or fail_penalty on any numerical failure."""
        fail = self.config["fail_penalty"]

        s = float(sigma)
        if (not np.isfinite(s)) or (s <= 0.0):
            return float(fail)

        # Enforce the sigma bounds during search (set by fit()).
        if (self._sigma_lo is not None) and (s < self._sigma_lo):
            return float(fail)
        if (self._sigma_hi is not None) and (s > self._sigma_hi):
            return float(fail)

        try:
            q = self._gmm_objective(s, W)
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
            return float(fail)

        if (not np.isfinite(q)) or (q >= fail):
            return float(fail)
        return float(q)

    def fit(self) -> None:
        """
        Run two-step GMM for sigma and store final beta and E_hat.

        Uses:
          - W1 = (Z'Z)^-1 for step 1.
          - W2 = Omega_hat(sigma_hat_1)^-1 for step 2.
        """
        sigma_lo = self.config["sigma_lower"]
        sigma_hi = self.config["sigma_upper"]
        self._sigma_lo = float(sigma_lo)
        self._sigma_hi = float(sigma_hi)

        # Step-1 weighting matrix: W1 = (Z'Z)^-1 using stacked instruments.
        Z = self.Zjt.reshape(-1, self.Zjt.shape[2])
        W1 = np.linalg.pinv(Z.T @ Z)

        # Step 1a: log-spaced grid search for a feasible starting point.
        sigmas = np.logspace(
            np.log10(self._sigma_lo),
            np.log10(self._sigma_hi),
            self.config["sigma_grid_points"],
        )
        best_sigma: float | None = None
        best_q = np.inf
        for s in sigmas:
            q = self._safe_gmm_objective(float(s), W1)
            if q < best_q:
                best_q = float(q)
                best_sigma = float(s)

        if (
            (best_sigma is None)
            or (not np.isfinite(best_q))
            or (best_q >= self.config["fail_penalty"])
        ):
            raise RuntimeError(
                "No feasible sigma found in the configured grid search bounds."
            )

        # Step 1b: Nelder–Mead on theta = log(sigma) to enforce positivity.
        def obj1(theta_vec: np.ndarray) -> float:
            return self._safe_gmm_objective(float(np.exp(theta_vec[0])), W1)

        res1 = minimize(
            fun=obj1,
            x0=np.array([np.log(best_sigma)]),
            method="Nelder-Mead",
            options={
                "maxiter": self.config["nelder_mead_maxiter"],
                "xatol": self.config["nelder_mead_xatol"],
                "fatol": self.config["nelder_mead_fatol"],
            },
        )
        sigma_hat_1 = float(np.exp(res1.x[0]))
        if (sigma_hat_1 < self._sigma_lo) or (sigma_hat_1 > self._sigma_hi):
            raise RuntimeError(
                "Step-1 optimization ended outside the configured sigma bounds."
            )

        # Build W2 at sigma_hat_1 (Omega_hat^-1).
        delta_1 = self._invert_demand(sigma_hat_1)
        beta_1 = self._estimate_beta(delta_1)
        E_1 = self._compute_E_hat(delta_1, beta_1)
        _, Omega_hat = self._moments_and_omega(E_1)
        W2 = np.linalg.pinv(Omega_hat)

        # Step 2: Nelder–Mead on theta = log(sigma).
        def obj2(theta_vec: np.ndarray) -> float:
            return self._safe_gmm_objective(float(np.exp(theta_vec[0])), W2)

        res2 = minimize(
            fun=obj2,
            x0=np.array([np.log(sigma_hat_1)]),
            method="Nelder-Mead",
            options={
                "maxiter": self.config["nelder_mead_maxiter"],
                "xatol": self.config["nelder_mead_xatol"],
                "fatol": self.config["nelder_mead_fatol"],
            },
        )
        sigma_hat_2 = float(np.exp(res2.x[0]))
        if (sigma_hat_2 < self._sigma_lo) or (sigma_hat_2 > self._sigma_hi):
            raise RuntimeError(
                "Step-2 optimization ended outside the configured sigma bounds."
            )

        # Final recomputation at sigma_hat_2 to store outputs used downstream.
        self.sigma_hat = sigma_hat_2

        delta_hat = self._invert_demand(self.sigma_hat)
        self.beta_hat = self._estimate_beta(delta_hat)
        self.beta_p_hat = float(self.beta_hat[0, 0])
        self.beta_w_hat = float(self.beta_hat[1, 0])
        self.E_hat = self._compute_E_hat(delta_hat, self.beta_hat)

        self.success = True

    def get_results(self) -> dict:
        """Return a minimal result dictionary for downstream simulation code."""
        return {
            "success": self.success,
            "sigma_hat": self.sigma_hat,
            "beta_p_hat": self.beta_p_hat,
            "beta_w_hat": self.beta_w_hat,
            "E_hat": self.E_hat,
        }
