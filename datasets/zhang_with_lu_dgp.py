"""
Minimal DGP for the "Zhang baseline logits + Lu market/product shocks" experiment,
with a fixed choice set (same J products in every market) and NO exogenous context.

Observed (shared features):
- xj:        (num_products, num_features)         product feature matrix
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
  Let xj have shape (J, D). Define, for each product j,

    L_j = sum_d x_{j,d}
    Q_j = sum_d x_{j,d}^2
    I_j = sum_d x_{j,d} * sum_{k!=j} x_{k,d}

  Then

    delta_true[j] = a_true * L_j + b_true * Q_j + g_true[j] * I_j

This preserves the original structure:
- one global linear coefficient,
- one global quadratic coefficient,
- one product-specific sparse interaction loading,
- one scalar baseline utility per product,

while allowing all feature dimensions to contribute to the truth.

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


def _as_feature_matrix(xj: np.ndarray) -> np.ndarray:
    """
    Normalize product features to a 2D matrix of shape (J, D).

    - If xj is 1D with shape (J,), it is promoted to shape (J, 1).
    - If xj is 2D with shape (J, D), it is returned as-is.

    Returns:
      x_mat: (J, D) feature matrix
    """
    xj = np.asarray(xj, dtype=np.float64)

    if xj.ndim == 1:
        return xj[:, None]

    if xj.ndim == 2:
        if xj.shape[1] < 1:
            raise ValueError("xj must have at least 1 feature column.")
        return xj

    raise ValueError(f"xj must be 1D or 2D. Got shape {xj.shape}.")


def compute_delta_true(
    xj: np.ndarray,
    a_true: float,
    b_true: float,
    g_true: np.ndarray,
) -> np.ndarray:
    """
    Baseline utilities with product-sparse cross-product interactions.

    All feature columns contribute to the scalar baseline utility for each product.
    With xj normalized to shape (J, D), define

      L_j = sum_d x_{j,d}
      Q_j = sum_d x_{j,d}^2
      I_j = sum_d x_{j,d} * sum_{k!=j} x_{k,d}

    and compute

      delta_true[j] = a_true * L_j + b_true * Q_j + g_true[j] * I_j

    Returns:
      delta_true: (J,)
    """
    x_mat = _as_feature_matrix(xj)
    g_true = np.asarray(g_true, dtype=np.float64).reshape(-1)

    num_products = x_mat.shape[0]
    if g_true.shape[0] != num_products:
        raise ValueError(
            f"g_true must have shape ({num_products},); got {g_true.shape}."
        )

    feature_sum_all = np.sum(x_mat, axis=0, keepdims=True)  # (1, D)
    feature_sum_excl = feature_sum_all - x_mat  # (J, D)

    linear_term = np.sum(x_mat, axis=1)  # (J,)
    quadratic_term = np.sum(x_mat * x_mat, axis=1)  # (J,)
    interaction_term = np.sum(x_mat * feature_sum_excl, axis=1)  # (J,)

    return a_true * linear_term + b_true * quadratic_term + g_true * interaction_term


def compute_njt_true(
    group_id: np.ndarray, z_true: np.ndarray, u_true_group: np.ndarray
) -> np.ndarray:
    """
    Compute product-level shocks from group-level activation and magnitudes.

    njt_true[t, j] = z_true[t, g(j)] * u_true_group[t, g(j)]

    Returns:
      njt_true: (T, J)
    """
    group_effect = z_true * u_true_group
    return group_effect[:, group_id]


def probs_with_outside(u_j: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Given inside utilities u_j (J,), return (p_inside (J,), p0) with outside utility 0.

    The computation is numerically stable by shifting by max(max(u_j), 0).
    """
    u_j = np.asarray(u_j, dtype=np.float64)
    m = max(0.0, float(np.max(u_j)))

    exp_inside = np.exp(u_j - m)
    exp_out = np.exp(-m)

    denom = exp_out + float(np.sum(exp_inside))
    return exp_inside / denom, float(exp_out / denom)


def sample_counts_single(
    rng: np.random.Generator, u_j: np.ndarray, N: int
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """
    Draw one multinomial sample over (outside + J inside goods).

    Returns:
      qj:       (J,) inside-good counts
      q0:       ()  outside-good count
      p_inside: (J,) true inside-good probabilities
      p0:       ()  true outside-good probability
    """
    u_j = np.asarray(u_j, dtype=np.float64)

    p_inside, p0 = probs_with_outside(u_j)
    p = np.empty(p_inside.shape[0] + 1, dtype=np.float64)
    p[0] = p0
    p[1:] = p_inside

    counts = rng.multinomial(int(N), p)
    q0 = int(counts[0])
    qj = counts[1:]
    return qj, q0, p_inside, p0


def sample_counts_markets(
    rng: np.random.Generator, u_tj: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each market t, draw multinomial counts over (outside + J inside goods).

    Returns:
      qjt: (T, J)
      q0t: (T,)
    """
    u_tj = np.asarray(u_tj, dtype=np.float64)
    num_markets, num_products = u_tj.shape

    qjt = np.zeros((num_markets, num_products), dtype=np.int64)
    q0t = np.zeros((num_markets,), dtype=np.int64)

    p = np.empty(num_products + 1, dtype=np.float64)
    for market_index in range(num_markets):
        p_inside, p0 = probs_with_outside(u_tj[market_index])
        p[0] = p0
        p[1:] = p_inside
        counts = rng.multinomial(int(N), p)
        q0t[market_index] = counts[0]
        qjt[market_index] = counts[1:]

    return qjt, q0t


def generate_choice_learn_market_shocks_dgp(
    seed: int,
    num_markets: int,
    num_products: int,
    num_groups: int,
    N_base: int,
    N_shock: int,
    num_features: int = 1,
    x_sd: float = 1.0,
    coef_sd: float = 1.0,
    p_g_active: float = 0.2,
    g_sd: float | None = None,
    sd_E: float = 0.5,
    p_active: float = 0.2,
    sd_u: float = 0.5,
) -> dict[str, object]:
    """
    Generate a minimal baseline-plus-shocks DGP for the Zhang-with-Lu experiment.

    Baseline phase is a single multinomial draw with N_base.
    Shock phase draws num_markets markets with N_shock each.

    Notes:
      - xj is returned as (J, num_features).
      - delta_true is computed from all feature columns in xj via compute_delta_true.
    """
    if int(num_features) < 1:
        raise ValueError("num_features must be >= 1.")

    rng = np.random.default_rng(int(seed))

    # Shared observed product features across all markets.
    xj = rng.normal(0.0, x_sd, size=(num_products, int(num_features)))
    group_id = _assign_groups(num_products, num_groups)

    # Baseline coefficients governing the feature-based utilities.
    a_true, b_true = rng.normal(0.0, coef_sd, size=(2,)).tolist()

    # Product-specific sparse interaction loadings.
    if g_sd is None:
        g_sd = coef_sd

    active = rng.binomial(1, p_g_active, size=(num_products,)).astype(np.int64)
    g_true = rng.normal(0.0, g_sd, size=(num_products,)) * active

    # Baseline utilities are fixed across markets.
    delta_true = compute_delta_true(xj, a_true, b_true, g_true)

    # Baseline single-market draw with no market/product shocks.
    qj_base, q0_base, p_base, p0_base = sample_counts_single(rng, delta_true, N=N_base)

    # Market-wide and grouped product shocks.
    E_bar_true = rng.normal(0.0, sd_E, size=(num_markets,))
    z_true = rng.binomial(1, p_active, size=(num_markets, num_groups))
    u_true_group = rng.normal(0.0, sd_u, size=(num_markets, num_groups))
    njt_true = compute_njt_true(group_id, z_true, u_true_group)

    # Shocked utilities and corresponding market counts.
    u_shock = delta_true[None, :] + E_bar_true[:, None] + njt_true
    qjt_shock, q0t_shock = sample_counts_markets(rng, u_shock, N=N_shock)

    return {
        "xj": xj,
        "group_id": group_id,
        "qj_base": qj_base,
        "q0_base": q0_base,
        "p_base": p_base,
        "p0_base": p0_base,
        "qjt_shock": qjt_shock,
        "q0t_shock": q0t_shock,
        "a_true": float(a_true),
        "b_true": float(b_true),
        "g_true": g_true,
        "delta_true": delta_true,
        "E_bar_true": E_bar_true,
        "njt_true": njt_true,
    }
