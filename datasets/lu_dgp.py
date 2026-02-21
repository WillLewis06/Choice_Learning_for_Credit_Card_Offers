"""
Synthetic data generation for Lu-style random-coefficient logit experiments.

This module provides:
  - Market primitives (wjt, Ejt, ujt, alpha) under several DGP variants.
  - A simple random-coefficient logit simulator (normal heterogeneity in price).
  - Aggregation from simulated utilities to expected shares and realized counts.
"""

import numpy as np


def generate_market_conditions(T: int, J: int, dgp_type: int, seed: int):
    """Generate market primitives and a price-endogeneity shifter.

    The demand shock is constructed as:
      Ejt = E_bar_t + n_jt, with E_bar_t fixed at -1 for all markets.

    DGP variants:
      1) Sparse n_jt, exogenous prices (alpha = 0)
      2) Sparse n_jt, endogenous prices (alpha depends on sign of n_jt)
      3) Dense n_jt ~ Normal(0, (1/3)^2), exogenous prices (alpha = 0)
      4) Dense n_jt, endogenous prices (alpha depends on thresholded n_jt)

    Args:
        T: Number of markets.
        J: Number of products.
        dgp_type: Integer in {1, 2, 3, 4}.
        seed: RNG seed.

    Returns:
        wjt: Exogenous characteristic, shape (T, J), Uniform(1, 2).
        Ejt: Total demand shock E_bar_t[:, None] + njt, shape (T, J).
        ujt: Cost shock, shape (T, J), Normal(0, 0.7^2).
        alpha: Price endogeneity shifter, shape (T, J).
        E_bar_t: Common market shock component (Lu table "Int"), shape (T,).
        njt: Market-product deviation shocks (Lu table "\u03be"), shape (T, J).
        support_true: For DGP1/2, boolean mask indicating nonzero njt (used for Lu table
            "Prob."). For DGP3/4, this is None.
    """
    if dgp_type not in (1, 2, 3, 4):
        raise ValueError("dgp_type must be 1, 2, 3, or 4")

    rng = np.random.default_rng(int(seed))

    # Product characteristic and baseline market component of the shock.
    wjt = rng.uniform(1.0, 2.0, size=(T, J))
    E_bar_t = -1.0 * np.ones(T, dtype=float)

    # Market-product shock: sparse (deterministic pattern) or dense (Gaussian).
    if dgp_type in (1, 2):
        njt = np.zeros((T, J), dtype=float)
        n_active = int(0.4 * J)
        for t in range(T):
            for j in range(n_active):
                njt[t, j] = 1.0 if (j % 2 == 0) else -1.0
    else:
        njt = rng.normal(0.0, 1.0 / 3.0, size=(T, J))

    # Total demand shock and cost shock used in the pricing rule.
    Ejt = E_bar_t[:, None] + njt
    ujt = rng.normal(0.0, 0.7, size=(T, J))

    # alpha creates correlation between prices and the demand shock for endogenous-price DGPs.
    alpha = np.zeros((T, J), dtype=float)
    if dgp_type == 2:
        alpha[njt == 1.0] = 0.3
        alpha[njt == -1.0] = -0.3
    elif dgp_type == 4:
        thr = 1.0 / 3.0
        alpha[njt >= thr] = 0.3
        alpha[njt <= -thr] = -0.3

    support_true = (njt != 0.0) if dgp_type in (1, 2) else None

    return wjt, Ejt, ujt, alpha, E_bar_t, njt, support_true


class BasicLuChoiceModel:
    """Random-coefficient logit with normal heterogeneity in price coefficient."""

    def __init__(
        self,
        N: int,
        beta_p: float,
        beta_w: float,
        sigma: float,
        seed: int,
    ):
        """Initialize simulated consumers via beta_{p,i} draws.

        Args:
            N: Number of simulated consumers.
            beta_p: Mean price coefficient.
            beta_w: Coefficient on wjt.
            sigma: Std. dev. of price coefficient heterogeneity.
            seed: RNG seed.
        """
        self.N = int(N)
        self.beta_w = float(beta_w)

        rng = np.random.default_rng(int(seed))
        self.beta_p_i = rng.normal(float(beta_p), float(sigma), size=self.N)

    def utilities(
        self, pjt: np.ndarray, wjt: np.ndarray, Ejt: np.ndarray
    ) -> np.ndarray:
        """Compute utilities u[i,t,j] for inside goods (no outside option utility)."""
        pjt = np.asarray(pjt, dtype=float)
        wjt = np.asarray(wjt, dtype=float)
        Ejt = np.asarray(Ejt, dtype=float)

        if pjt.shape != wjt.shape or pjt.shape != Ejt.shape:
            raise ValueError("pjt, wjt, and Ejt must have the same shape (T, J)")

        T, J = pjt.shape
        # u[i,t,j] = beta_{p,i} * p[t,j] + beta_w * w[t,j] + E[t,j]
        uijt = (
            self.beta_p_i[:, None, None] * pjt[None, :, :]
            + self.beta_w * wjt[None, :, :]
            + Ejt[None, :, :]
        )
        return uijt


def generate_market(uijt: np.ndarray, N: int, seed: int):
    """Convert utilities into expected shares and realized multinomial counts."""
    uijt = np.asarray(uijt, dtype=float)
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (N, T, J)")

    N = int(N)
    rng = np.random.default_rng(int(seed))

    # Choice probabilities with outside option utility normalized to 0.
    exp_u = np.exp(uijt)
    denom = 1.0 + exp_u.sum(axis=2, keepdims=True)
    pij = exp_u / denom
    p0 = 1.0 / denom[..., 0]

    # Expected shares.
    sjt = pij.mean(axis=0)
    s0t = p0.mean(axis=0)

    # Realized counts per market.
    T, J = sjt.shape
    qjt = np.zeros((T, J), dtype=int)
    q0t = np.zeros(T, dtype=int)
    for t in range(T):
        probs = np.concatenate([s0t[t : t + 1], sjt[t]], axis=0)
        draws = rng.multinomial(N, probs)
        q0t[t] = int(draws[0])
        qjt[t, :] = draws[1:]

    return sjt, s0t, qjt, q0t
