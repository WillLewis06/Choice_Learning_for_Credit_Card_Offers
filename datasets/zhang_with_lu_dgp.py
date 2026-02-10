"""
Minimal DGP for the "Zhang baseline logits + Lu market/product shocks" experiment,
with a fixed choice set (same J products in every market) and NO exogenous context.

Observed (shared features):
- xj:        (num_products,)                      product feature
- group_id:  (num_products,) int                  product -> group mapping

Observed (counts):
Baseline phase (single market draw):
- qj_base:    (num_products,)                     inside-good counts (shockless)
- q0_base:    ()                                  outside count (shockless)
- p_base:     (num_products,)                     true inside probabilities (shockless)
- p0_base:    ()                                  true outside probability (shockless)

Shock phase (many markets):
- qjt_shock:  (num_markets, num_products)         inside-good counts (with shocks)
- q0t_shock:  (num_markets,)                      outside counts (with shocks)

Truth (baseline):
- a_true:     scalar
- b_true:     scalar
- g_true:     (num_products,) sparse loadings
- delta_true: (num_products,) baseline utilities (no shocks)

Truth (shocks):
- E_bar_true: (num_markets,) market-wide shocks
- njt_true:   (num_markets, num_products) product shocks implied by groups

Baseline utility (fixed across markets):
  let S = sum_k xk, and S_{-j} = S - xj
  delta_true[j] = a*xj[j] + b*xj[j]^2 + g_true[j] * xj[j] * S_{-j}

Shocked utilities for market t:
  u_shock[t, j] = delta_true[j] + E_bar_true[t] + njt_true[t, j]

Outside option has fixed utility 0 and is included in the multinomial draw.
"""

from __future__ import annotations

import numpy as np


def _assign_groups(num_products: int, num_groups: int) -> np.ndarray:
    """
    Contiguous assignment into num_groups blocks.
    Returns group_id with shape (num_products,) and values 0..num_groups-1.
    """
    base = num_products // num_groups
    rem = num_products % num_groups
    sizes = [base + (1 if g < rem else 0) for g in range(num_groups)]

    group_id = np.empty(num_products, dtype=np.int64)
    idx = 0
    for g, sz in enumerate(sizes):
        group_id[idx : idx + sz] = g
        idx += sz

    return group_id


def compute_delta_true(
    xj: np.ndarray,
    a_true: float,
    b_true: float,
    g_true: np.ndarray,
) -> np.ndarray:
    """
    Baseline utilities with product-sparse cross-product interactions.

    delta_true[j] = a*xj[j] + b*xj[j]^2 + g_true[j] * xj[j] * sum_{k!=j} xj[k]
    Returns delta_true with shape (J,).
    """
    xj = np.asarray(xj)
    g_true = np.asarray(g_true)

    s_all = float(np.sum(xj))
    s_excl = s_all - xj  # (J,)

    return a_true * xj + b_true * (xj**2) + g_true * xj * s_excl


def compute_njt_true(
    group_id: np.ndarray, z_true: np.ndarray, u_true_group: np.ndarray
) -> np.ndarray:
    """
    njt_true[t, j] = z_true[t, g(j)] * u_true_group[t, g(j)]
    Returns (T, J).
    """
    group_effect = z_true * u_true_group  # (T, G) -> float via upcast
    return group_effect[:, group_id]  # (T, J)


def probs_with_outside(u_j: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Given inside utilities u_j (J,), return (p_inside (J,), p0) with outside utility 0.
    Numerically stable by shifting by max(max(u_j), 0).
    """
    u_j = np.asarray(u_j)
    m = max(0.0, float(np.max(u_j)))  # include outside option (0)

    exp_inside = np.exp(u_j - m)
    exp_out = np.exp(-m)

    denom = exp_out + float(np.sum(exp_inside))
    return exp_inside / denom, float(exp_out / denom)


def sample_counts_single(
    rng: np.random.Generator, u_j: np.ndarray, N: int
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """
    One multinomial draw over (outside + J inside).

    Returns:
      qj:      (J,) inside counts
      q0:      ()  outside count
      p_inside:(J,) true inside probabilities
      p0:      ()  true outside probability
    """
    u_j = np.asarray(u_j)

    p_inside, p0 = probs_with_outside(u_j)
    p = np.empty(p_inside.shape[0] + 1, dtype=np.float64)
    p[0] = p0
    p[1:] = p_inside

    counts = rng.multinomial(int(N), p)
    q0 = int(counts[0])
    qj = counts[1:]  # already integer dtype
    return qj, q0, p_inside, p0


def sample_counts_markets(
    rng: np.random.Generator, u_tj: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each market t, sample multinomial counts over (outside + J inside).

    Returns:
      qjt: (T, J)
      q0t: (T,)
    """
    u_tj = np.asarray(u_tj)
    T, J = u_tj.shape

    qjt = np.zeros((T, J), dtype=np.int64)
    q0t = np.zeros((T,), dtype=np.int64)

    p = np.empty(J + 1, dtype=np.float64)
    for t in range(T):
        p_inside, p0 = probs_with_outside(u_tj[t])
        p[0] = p0
        p[1:] = p_inside
        counts = rng.multinomial(int(N), p)
        q0t[t] = counts[0]
        qjt[t] = counts[1:]

    return qjt, q0t


def generate_choice_learn_market_shocks_dgp(
    seed: int,
    num_markets: int,
    num_products: int,
    num_groups: int,
    N_base: int,
    N_shock: int,
    # feature generation (normal)
    x_sd: float = 1.0,
    # baseline coefficient generation (normal)
    coef_sd: float = 1.0,
    # product-sparse interaction loadings
    p_g_active: float = 0.2,
    g_sd: float | None = None,
    # shocks
    sd_E: float = 0.5,
    p_active: float = 0.2,
    sd_u: float = 0.5,
) -> dict[str, object]:
    """
    Returns a flat dict of arrays/scalars.

    Baseline phase is a single multinomial draw with N_base.
    Shock phase draws num_markets markets with N_shock each.
    """
    rng = np.random.default_rng(int(seed))

    # Shared observed features
    xj = rng.normal(0.0, x_sd, size=(num_products,))
    group_id = _assign_groups(num_products, num_groups)

    # Baseline coefficients (truth)
    a_true, b_true = rng.normal(0.0, coef_sd, size=(2,)).tolist()

    # Product-sparse interaction loadings g_true
    if g_sd is None:
        g_sd = coef_sd

    active = rng.binomial(1, p_g_active, size=(num_products,)).astype(np.int64)
    g_true = rng.normal(0.0, g_sd, size=(num_products,)) * active

    # Baseline utilities (truth) - fixed across markets
    delta_true = compute_delta_true(xj, a_true, b_true, g_true)

    # Baseline (shockless) single-market draw
    qj_base, q0_base, p_base, p0_base = sample_counts_single(rng, delta_true, N=N_base)

    # Shocks (truth) across markets
    E_bar_true = rng.normal(0.0, sd_E, size=(num_markets,))
    z_true = rng.binomial(1, p_active, size=(num_markets, num_groups))  # int
    u_true_group = rng.normal(0.0, sd_u, size=(num_markets, num_groups))  # float
    njt_true = compute_njt_true(group_id, z_true, u_true_group)

    # Shocked utilities and counts
    u_shock = delta_true[None, :] + E_bar_true[:, None] + njt_true
    qjt_shock, q0t_shock = sample_counts_markets(rng, u_shock, N=N_shock)

    return {
        "xj": xj,
        "group_id": group_id,
        # baseline observed
        "qj_base": qj_base,
        "q0_base": q0_base,
        "p_base": p_base,
        "p0_base": p0_base,
        # shock observed
        "qjt_shock": qjt_shock,
        "q0t_shock": q0t_shock,
        # baseline truth
        "a_true": float(a_true),
        "b_true": float(b_true),
        "g_true": g_true,
        "delta_true": delta_true,
        # shock truth
        "E_bar_true": E_bar_true,
        "njt_true": njt_true,
    }
