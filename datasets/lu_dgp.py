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
        Ejt: Demand shock, shape (T, J).
        ujt: Cost shock, shape (T, J), Normal(0, 0.7^2).
        alpha: Price endogeneity shifter, shape (T, J).
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

    return wjt, Ejt, ujt, alpha


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
        """Compute systematic utilities for each market, consumer, and product.

        Utility used here:
          u_{i,j,t} = beta_{p,i} * p_{j,t} + beta_w * w_{j,t} + E_{j,t}

        Args:
            pjt: Prices, shape (T, J).
            wjt: Characteristics, shape (T, J).
            Ejt: Demand shocks, shape (T, J).

        Returns:
            uijt: Systematic utilities, shape (T, N, J).
        """
        pjt = np.asarray(pjt, dtype=float)
        wjt = np.asarray(wjt, dtype=float)
        Ejt = np.asarray(Ejt, dtype=float)

        if pjt.ndim != 2 or wjt.ndim != 2 or Ejt.ndim != 2:
            raise ValueError("pjt, wjt, Ejt must be 2D arrays of shape (T, J).")
        if pjt.shape != wjt.shape or pjt.shape != Ejt.shape:
            raise ValueError("pjt, wjt, Ejt must have the same shape (T, J).")

        T, J = pjt.shape
        uijt = np.zeros((T, self.N, J), dtype=float)

        # Market loop keeps memory usage predictable and mirrors the model structure.
        for t in range(T):
            uijt[t] = (
                self.beta_p_i[:, None] * pjt[t][None, :]
                + self.beta_w * wjt[t][None, :]
                + Ejt[t][None, :]
            )

        return uijt


def _generate_market_shares(uijt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected inside and outside shares from simulated utilities.

    This computes, for each market t:
      s_{j,t} = E_i[ exp(u_{i,j,t}) / (1 + sum_k exp(u_{i,k,t})) ]
      s_{0,t} = E_i[ 1 / (1 + sum_k exp(u_{i,k,t})) ]

    Args:
        uijt: Systematic utilities, shape (T, N, J).

    Returns:
        sjt: Expected inside shares, shape (T, J).
        s0t: Expected outside shares, shape (T,).
    """
    uijt = np.asarray(uijt, dtype=float)
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (T, N, J).")

    # Log-sum-exp stabilization per (t, i) to reduce overflow in exp(u).
    m_inside = np.max(uijt, axis=2, keepdims=True)
    m = np.maximum(0.0, m_inside)

    exp_u = np.exp(uijt - m)
    exp_out = np.exp(-m[..., 0])
    denom = exp_out + np.sum(exp_u, axis=2)

    probs = exp_u / denom[..., None]
    out_prob = exp_out / denom

    sjt = np.mean(probs, axis=1)
    s0t = np.mean(out_prob, axis=1)
    return sjt, s0t


def generate_market(
    uijt: np.ndarray, N: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate expected shares and realized multinomial counts for each market.

    The counts correspond to N independent consumers choosing among:
      {outside option} ∪ {J inside goods},
    where choice probabilities are computed from uijt.

    Args:
        uijt: Systematic utilities, shape (T, N_sim, J).
        N: Number of consumers per market for multinomial sampling.
        seed: RNG seed.

    Returns:
        sjt: Expected inside shares, shape (T, J).
        s0t: Expected outside shares, shape (T,).
        qjt: Realized inside counts, shape (T, J).
        q0t: Realized outside counts, shape (T,).
    """
    uijt = np.asarray(uijt, dtype=float)
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (T, N_sim, J).")

    rng = np.random.default_rng(int(seed))
    sjt, s0t = _generate_market_shares(uijt)

    T, J = sjt.shape
    qjt = np.zeros((T, J), dtype=int)
    q0t = np.zeros(T, dtype=int)

    # Market-by-market multinomial draw of counts given expected shares.
    for t in range(T):
        probs = np.concatenate([[s0t[t]], sjt[t]])
        probs = probs / probs.sum()
        draw = rng.multinomial(int(N), probs)
        q0t[t] = int(draw[0])
        qjt[t, :] = draw[1:]

    return sjt, s0t, qjt, q0t
