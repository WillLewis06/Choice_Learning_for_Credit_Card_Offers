"""
End-to-end orchestration for:

- Phase 1: feature-based baseline choice model (delta_hat)
- Phase 2: Lu-style shrinkage on market-product shocks (E_bar_hat, njt_hat, alpha_hat)
- Phase 3 (stockpiling): generate seller-observed purchases under a forward-looking
  inventory model, then estimate stockpiling parameters treating utilities as fixed.

Notes:
- This orchestration layer stays NumPy/Python only. Any TensorFlow conversion is handled
  inside the stockpiling estimator.
- Input validation and detailed diagnostics live elsewhere; this file prints high-level
  milestones plus existing Phase 1–2 diagnostics.
"""

from __future__ import annotations

import os

import numpy as np

# Keep TF logs quiet for upstream modules that may import TF internally.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets.ching_dgp import generate_dgp
from run_zhang_with_lu import (
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
    run_market_shock_estimator,
)
from ching.stockpiling_estimator import StockpilingEstimator
from ching.stockpiling_evaluate import evaluate_stockpiling, format_evaluation_summary

# =============================================================================
# Phase 1: baseline choice model hyperparameters
# =============================================================================

seed = 123
num_products = 15
num_groups = 5
num_markets = 5

N_base = 2_000
N_shock = 1_000

x_sd = 1.0
coef_sd = 1.0
p_g_active = 0.2
g_sd = None

sd_E = 0.5
p_active = 0.25
sd_u = 0.5

depth = 10
width = 64
heads = 8

epochs = 5
batch_size = 64
learning_rate = 1e-3
shuffle_buffer = 10_000

# Diagnostics controls (used by the Phase 1–2 diagnostics printers)
eval_include_outside = True
eval_against_empirical = True


# =============================================================================
# Phase 2: market shock estimator hyperparameters
# =============================================================================

shrink_seed = 0
shrink_n_iter = 10
shrink_pilot_length = 20
shrink_max_rounds = 50
shrink_target_low = 0.3
shrink_target_high = 0.5
shrink_factor_rw = 1.2
shrink_factor_tmh = 1.2
shrink_ridge = 1e-6


# =============================================================================
# Stockpiling model configuration + estimation hyperparameters
# =============================================================================

product_index = 0  # chosen product j*

# Stockpiling DGP/likelihood config (formerly stock dict)
stock_N = 500
stock_T = 100
stock_I_max = 10
stock_P_price = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
stock_price_vals = np.array([1.0, 0.8], dtype=np.float64)
stock_waste_cost = 1.0
stock_dp_tol = 1e-10
stock_dp_max_iter = 50_000
stock_eps = 1e-12

# MCMC config (formerly mcmc dict); keep k and sigmas as dicts
mcmc_seed = 0
mcmc_n_iter = 4

k = {
    "beta": 1,
    "alpha": 0.7,
    "v": 1,
    "fc": 0.7,
    "lambda": 1,
    "u_scale": 0.05,
}

sigmas = {
    "z_beta": 2.0,
    "z_alpha": 2.0,
    "z_v": 2.0,
    "z_fc": 2.0,
    "z_lambda": 2.0,
    "z_u_scale": 2.0,
}


# =============================================================================
# Helpers
# =============================================================================


def _uniform_pi_I0(I_max: int) -> np.ndarray:
    """
    Uniform initial-inventory belief over {0, ..., I_max}.

    Returns:
      ndarray: shape (I_max + 1,), float64.
    """
    return np.full((I_max + 1,), 1.0 / (I_max + 1), dtype=np.float64)


def run_stockpiling_dgp(
    delta_used: np.ndarray,
    E_bar_used: np.ndarray,
    njt_used: np.ndarray,
    seed_dgp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate seller-observed stockpiling data using fixed utilities.

    Fixed per-market intercept for product_index:
      u_m = delta_used[product_index] + E_bar_used[m] + njt_used[m, product_index].

    Returns:
      a_imt      (M, N, T) int64 purchases
      p_state_mt (M, T)    int64 price states
      u_m_true   (M,)      float64 market intercepts
      theta_true dict of (M, N) float64 arrays
    """
    delta_used = np.asarray(delta_used, dtype=np.float64)
    E_bar_used = np.asarray(E_bar_used, dtype=np.float64)
    njt_used = np.asarray(njt_used, dtype=np.float64)

    a_imt, p_state_mt, u_m_true, theta_true = generate_dgp(
        seed=seed_dgp,
        delta_true=delta_used,
        E_bar_true=E_bar_used,
        njt_true=njt_used,
        product_index=product_index,
        N=stock_N,
        T=stock_T,
        I_max=stock_I_max,
        P_price=stock_P_price,
        price_vals=stock_price_vals,
        waste_cost=stock_waste_cost,
        tol=stock_dp_tol,
        max_iter=stock_dp_max_iter,
    )
    return a_imt, p_state_mt, u_m_true, theta_true


def run_stockpiling_estimation(
    a_imt: np.ndarray,
    p_state_mt: np.ndarray,
    u_m: np.ndarray,
) -> dict[str, object]:
    """
    Fit stockpiling parameters from seller-observed data.

    Returns:
      dict with keys used downstream, including:
        - "theta_hat": fitted parameter summary (structure defined by estimator)
        - "n_saved": number of saved posterior draws
        - "accept": acceptance rate diagnostics
    """
    est = StockpilingEstimator(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=stock_price_vals,
        P_price=stock_P_price,
        I_max=stock_I_max,
        pi_I0=_uniform_pi_I0(stock_I_max),
        waste_cost=stock_waste_cost,
        eps=stock_eps,
        tol=stock_dp_tol,
        max_iter=stock_dp_max_iter,
        sigmas=sigmas,
        seed=mcmc_seed,
    )
    print("=== Stockpiling Estimator built ===")

    est.fit(
        n_iter=mcmc_n_iter,
        k_beta=k["beta"],
        k_alpha=k["alpha"],
        k_v=k["v"],
        k_fc=k["fc"],
        k_lambda=k["lambda"],
        k_u_scale=k["u_scale"],
    )
    print("=== Stockpiling Estimator fitted ===")

    return est.get_results()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run Phase 1–2 utilities, then stockpiling DGP, estimation, and evaluation."""
    # -------------------------------------------------------------------------
    # Phase 1
    # -------------------------------------------------------------------------
    print("=== Phase 1: Baseline choice model ===")
    out1 = run_choice_model(
        seed=seed,
        num_products=num_products,
        num_groups=num_groups,
        num_markets=num_markets,
        N_base=N_base,
        N_shock=N_shock,
        x_sd=x_sd,
        coef_sd=coef_sd,
        p_g_active=p_g_active,
        g_sd=g_sd,
        sd_E=sd_E,
        p_active=p_active,
        sd_u=sd_u,
        depth=depth,
        width=width,
        heads=heads,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shuffle_buffer=shuffle_buffer,
    )
    print("=== Baseline choice model fitted ===")

    dgp = out1["dgp"]
    delta_hat = np.asarray(out1["delta_hat"], dtype=np.float64)

    print_choice_model_diagnostics(
        delta_hat=delta_hat,
        delta_true=dgp["delta_true"],
        qj_base=dgp["qj_base"],
        q0_base=int(dgp["q0_base"]),
        p_base=dgp["p_base"],
        p0_base=float(dgp["p0_base"]),
        N_base=N_base,
        eval_include_outside=eval_include_outside,
        eval_against_empirical=eval_against_empirical,
    )
    print("=== Phase 1 complete ===")

    # -------------------------------------------------------------------------
    # Phase 2
    # -------------------------------------------------------------------------
    print("=== Phase 2: Market shock estimator ===")
    res2 = run_market_shock_estimator(
        delta_hat=delta_hat,
        qjt_shock=dgp["qjt_shock"],
        q0t_shock=dgp["q0t_shock"],
        seed=shrink_seed,
        n_iter=shrink_n_iter,
        pilot_length=shrink_pilot_length,
        max_rounds=shrink_max_rounds,
        target_low=shrink_target_low,
        target_high=shrink_target_high,
        factor_rw=shrink_factor_rw,
        factor_tmh=shrink_factor_tmh,
        ridge=shrink_ridge,
    )
    print("=== Market shock estimator fitted ===")

    print_market_shock_diagnostics(
        delta_hat=delta_hat,
        dgp=dgp,
        res=res2,
        eval_include_outside=eval_include_outside,
    )
    print("=== Phase 2 complete ===")

    # -------------------------------------------------------------------------
    # Stockpiling DGP: use Phase-2 posterior means as fixed utilities.
    # -------------------------------------------------------------------------
    alpha_hat = float(res2["alpha_hat"])
    E_bar_hat = np.asarray(res2["E_bar_hat"], dtype=np.float64)
    njt_hat = np.asarray(res2["njt_hat"], dtype=np.float64)

    # Rescale Phase-1 utilities by the Phase-2 normalization (alpha_hat) so the
    # stockpiling DGP/likelihood sees utilities on the intended scale.
    delta_used = alpha_hat * delta_hat

    # Generate seller-observed stockpiling panel data using fixed utilities.
    print("=== Stockpiling DGP: Generating seller-observed data ===")
    a_imt, p_state_mt, u_m_true, theta_true = run_stockpiling_dgp(
        delta_used=delta_used,
        E_bar_used=E_bar_hat,
        njt_used=njt_hat,
        seed_dgp=seed + 999,
    )

    buy_rate = float(np.mean(a_imt))
    print("=== Stockpiling data generated ===")
    print(
        f"a_imt shape: {a_imt.shape} | p_state_mt shape: {p_state_mt.shape} | u_m shape: {u_m_true.shape}"
    )
    print(f"overall buy rate: {buy_rate:.4f}")

    # -------------------------------------------------------------------------
    # Stockpiling estimation
    # -------------------------------------------------------------------------
    print("=== Stockpiling estimation ===")
    res3 = run_stockpiling_estimation(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m_true,
    )

    pi_I0 = _uniform_pi_I0(stock_I_max)

    # Evaluate fitted parameters versus (optional) oracle truth and summarize MCMC
    # diagnostics (acceptance, saved draws) in a single reviewer-friendly report.
    eval_out = evaluate_stockpiling(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m_true,
        price_vals=stock_price_vals,
        P_price=stock_P_price,
        I_max=stock_I_max,
        pi_I0=pi_I0,
        waste_cost=stock_waste_cost,
        eps=stock_eps,
        tol=stock_dp_tol,
        max_iter=stock_dp_max_iter,
        theta_hat=res3["theta_hat"],
        theta_true=theta_true,
        mcmc={"n_saved": res3["n_saved"], "accept": res3["accept"]},
    )

    print("=== Stockpiling evaluation ===")
    print(format_evaluation_summary(eval_out))


if __name__ == "__main__":
    main()
