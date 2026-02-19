"""
run_bonus2.py

End-to-end orchestration for Bonus Q2 (updated spec):

- Phase 1: Zhang feature-based baseline choice model -> delta_hat (J,)
- Bonus2 DGP: simulate y_mit under habit + peer + weekend + market seasonality MNL using delta_mj
- Bonus2 estimation: RW-MH over z-blocks -> theta_hat
- Bonus2 evaluation: summary via bonus2_evaluate.py

Notes:
- We do NOT run the Lu market-shock phase.
- The within-market social network (neighbors) is treated as known and passed to the estimator.
- Habit decay is a known scalar (decay) passed to both DGP and estimator.
- Time feature naming:
    w_t                 (T,) in {0,1} where 1=weekend
    season_sin_kt       (K,T)
    season_cos_kt       (K,T)
- All input validation is centralized in bonus2/bonus2_input_validation.py.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Keep TF logs quiet for downstream modules that may import TF internally.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from datasets.bonus2_dgp import simulate_bonus2_dgp  # noqa: E402
from run_zhang_with_lu import (  # noqa: E402
    print_choice_model_diagnostics,
    run_choice_model,
)
from bonus2.bonus2_estimator import Bonus2Estimator  # noqa: E402
from bonus2 import bonus2_model as b2_model  # noqa: E402
from bonus2.bonus2_evaluate import (  # noqa: E402
    evaluate_bonus2,
    format_evaluation_summary,
)
from bonus2.bonus2_diagnostics import (  # noqa: E402
    report_theta_summary,
    report_known_summary,
)

from bonus2.bonus2_input_validation import (  # noqa: E402
    validate_phase1_delta_hat,
    validate_bonus2_panel,
)

# =============================================================================
# Configuration
# =============================================================================

# Phase 1: baseline choice model (Zhang feature-based)
CFG_PHASE1 = {
    "seed": 123,
    "num_products": 15,
    "num_groups": 15,
    "num_markets": 3,
    "N_base": 2_000,
    "N_shock": 1_000,
    "num_features": 4,
    "x_sd": 1.0,
    "coef_sd": 1.0,
    "p_g_active": 0.2,
    "g_sd": None,
    "sd_E": 0.5,
    "p_active": 0.25,
    "sd_u": 0.5,
    "depth": 5,
    "width": 64,
    "heads": 8,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "shuffle_buffer": 10_000,
    "eval_include_outside": True,
    "eval_against_empirical": True,
}

# Bonus2: DGP + estimation (updated spec)
CFG_BONUS2 = {
    # panel size
    "N": 100,
    "T": (365 * 10),
    # time features
    "season_period": 365,
    "K": 1,
    "peer_lookback_L": 1,
    # known scalar habit decay
    "decay": 0.8,
    # network hyperparams (DGP)
    "avg_friends": 3.0,
    "friends_sd": 1.0,
    # DGP parameter dispersion (required keys; no defaults in DGP)
    "params_true": {
        "habit_mean": 0.60,
        "habit_sd": 0.05,
        "peer_mean": 0.5,
        "peer_sd": 0.15,
        "mktprod_sd": 0.5,  # product intercept sd
        "dow_prod_sd": 0.2,  # product weekend effect sd
        "season_mkt_sd": 0.2,  # market seasonality sd
    },
    # MCMC config
    "mcmc_seed": 0,
    "mcmc_n_iter": 200,
    # estimator init (scalar fills)
    "init_theta": {
        "beta_market": 0.0,
        "beta_habit": 0.0,
        "beta_peer": 0.0,
        "beta_dow_j": 0.0,
        "a_m": 0.0,
        "b_m": 0.0,
    },
    # z-space prior scales (keys must match Bonus2Estimator expectation)
    "sigmas": {
        "z_beta_market_j": 2.0,
        "z_beta_habit_j": 2.0,
        "z_beta_peer_j": 2.0,
        "z_beta_dow_j": 2.0,
        "z_a_m": 2.0,
        "z_b_m": 2.0,
    },
    # RW step sizes (keys must match Bonus2Estimator.fit expectation)
    "k": {
        "beta_market": 0.02,
        "beta_habit": 0.02,
        "beta_peer": 0.02,
        "beta_dow_j": 0.01,
        "a_m": 0.05,
        "b_m": 0.05,
    },
}

# =============================================================================
# Utilities
# =============================================================================


def _tile_delta_mj(delta_hat: np.ndarray, M: int) -> np.ndarray:
    """Tile phase-1 delta_hat (J,) into delta_mj (M,J)."""
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    return np.tile(delta_hat[None, :], reps=(int(M), 1)).astype(np.float64)


def _theta_true_from_panel(panel: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract true theta arrays from a Bonus2 panel dict (updated spec)."""
    return {
        "beta_market_j": np.asarray(panel["beta_market_j"], dtype=np.float64),
        "beta_habit_j": np.asarray(panel["beta_habit_j"], dtype=np.float64),
        "beta_peer_j": np.asarray(panel["beta_peer_j"], dtype=np.float64),
        "beta_dow_j": np.asarray(panel["beta_dow_j"], dtype=np.float64),
        "a_m": np.asarray(panel["a_m"], dtype=np.float64),
        "b_m": np.asarray(panel["b_m"], dtype=np.float64),
    }


def _theta_init_full(
    init_theta: dict[str, float], M: int, J: int, K: int
) -> dict[str, np.ndarray]:
    """Expand scalar init_theta fills into full theta arrays (updated spec)."""
    return {
        "beta_market_j": np.full(
            (J,), float(init_theta["beta_market"]), dtype=np.float64
        ),
        "beta_habit_j": np.full(
            (J,), float(init_theta["beta_habit"]), dtype=np.float64
        ),
        "beta_peer_j": np.full((J,), float(init_theta["beta_peer"]), dtype=np.float64),
        "beta_dow_j": np.full(
            (J, 2), float(init_theta["beta_dow_j"]), dtype=np.float64
        ),
        "a_m": np.full((M, K), float(init_theta["a_m"]), dtype=np.float64),
        "b_m": np.full((M, K), float(init_theta["b_m"]), dtype=np.float64),
    }


# =============================================================================
# Bonus2 summaries
# =============================================================================


def summarize_bonus2_panel(panel: dict[str, Any], init_theta: dict[str, float]) -> None:
    y = np.asarray(panel["y"], dtype=np.int64)  # (M,N,T)
    delta = np.asarray(panel["delta"], dtype=np.float64)  # (M,J)
    w_t = np.asarray(panel["w"], dtype=np.int64)  # (T,)
    season_sin_kt = np.asarray(panel["season_sin_kt"], dtype=np.float64)  # (K,T)
    nbrs = panel["nbrs"]

    M, N, T = y.shape
    J = delta.shape[1]
    K = int(panel["K"])

    outside_share = float(np.mean(y == 0))
    inside_shares = np.array(
        [np.mean(y == (j + 1)) for j in range(J)], dtype=np.float64
    )

    # average out-degree
    degs: list[int] = []
    for m in range(M):
        rows = nbrs[m]
        for i in range(N):
            degs.append(int(len(rows[i])))
    avg_deg = float(np.mean(degs)) if degs else 0.0

    print("=== Bonus2 data generated ===")
    print(
        "shapes: "
        f"y={y.shape} | delta={delta.shape} | w_t={w_t.shape} | season_sin_kt={season_sin_kt.shape}"
    )
    print(f"outside_share: {outside_share:.4f} | avg_out_degree: {avg_deg:.2f}")
    print(
        f"inside_share_mean: {float(inside_shares.mean()):.4f} | inside_share_max: {float(inside_shares.max()):.4f}"
    )

    theta_true = _theta_true_from_panel(panel)
    theta_init = _theta_init_full(init_theta, M=M, J=J, K=K)

    report_theta_summary("True", theta_true)
    report_theta_summary("Init", theta_init)
    report_known_summary(
        L=int(panel["peer_lookback_L"]),
        K=int(panel["K"]),
        season_period=int(panel["season_period"]),
        decay=float(panel["decay"]),
    )


# =============================================================================
# Phase runners
# =============================================================================


def run_phase1(cfg: dict[str, Any]) -> dict[str, Any]:
    print("=== Phase 1: Baseline choice model (Zhang feature-based) ===")

    out1 = run_choice_model(
        seed=int(cfg["seed"]),
        num_products=int(cfg["num_products"]),
        num_groups=int(cfg["num_groups"]),
        num_markets=int(cfg["num_markets"]),
        N_base=int(cfg["N_base"]),
        N_shock=int(cfg["N_shock"]),
        num_features=int(cfg["num_features"]),
        x_sd=float(cfg["x_sd"]),
        coef_sd=float(cfg["coef_sd"]),
        p_g_active=float(cfg["p_g_active"]),
        g_sd=cfg["g_sd"],
        sd_E=float(cfg["sd_E"]),
        p_active=float(cfg["p_active"]),
        sd_u=float(cfg["sd_u"]),
        depth=int(cfg["depth"]),
        width=int(cfg["width"]),
        heads=int(cfg["heads"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        shuffle_buffer=int(cfg["shuffle_buffer"]),
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
        N_base=int(cfg["N_base"]),
        eval_include_outside=bool(cfg["eval_include_outside"]),
        eval_against_empirical=bool(cfg["eval_against_empirical"]),
    )

    print("=== Phase 1 complete ===")
    return {"dgp": dgp, "delta_hat": delta_hat}


def run_bonus2_dgp(
    cfg: dict[str, Any], delta_mj: np.ndarray, seed: Any
) -> dict[str, Any]:
    print(
        "=== Bonus2 DGP: Generating y_mit under habit + peer + weekend + market seasonality ==="
    )

    panel = simulate_bonus2_dgp(
        delta=delta_mj,
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        avg_friends=float(cfg["avg_friends"]),
        friends_sd=float(cfg["friends_sd"]),
        params_true=cfg["params_true"],
        decay=float(cfg["decay"]),
        seed=seed,
        season_period=int(cfg["season_period"]),
        K=int(cfg["K"]),
        peer_lookback_L=int(cfg["peer_lookback_L"]),
    )

    print("=== Bonus2 DGP complete ===")
    return panel


def run_bonus2_estimation(cfg: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    print("=== Bonus2 estimation ===")

    validate_bonus2_panel(panel)

    est = Bonus2Estimator(
        y_mit=panel["y"],
        delta_mj=panel["delta"],
        weekend_t=panel["w"],
        season_sin_kt=panel["season_sin_kt"],
        season_cos_kt=panel["season_cos_kt"],
        neighbors=panel["nbrs"],
        L=int(panel["peer_lookback_L"]),
        decay=float(panel["decay"]),
        init_theta=cfg["init_theta"],
        sigmas=cfg["sigmas"],
        seed=int(cfg["mcmc_seed"]),
    )
    print("=== Bonus2 Estimator built ===")

    est.fit(
        n_iter=int(cfg["mcmc_n_iter"]), k={k: float(v) for k, v in cfg["k"].items()}
    )

    print("=== Bonus2 Estimator fitted ===")
    return est.get_results()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # Phase 1 (Zhang)
    out1 = run_phase1(CFG_PHASE1)

    delta_hat = out1["delta_hat"]
    validate_phase1_delta_hat(delta_hat, num_products=CFG_PHASE1["num_products"])

    M = int(CFG_PHASE1["num_markets"])

    # Build delta_mj for Bonus2 (fixed baseline utilities)
    delta_mj = _tile_delta_mj(delta_hat=delta_hat, M=M)

    # Bonus2 DGP
    seed_dgp = int(CFG_PHASE1["seed"]) + 999
    panel = run_bonus2_dgp(CFG_BONUS2, delta_mj=delta_mj, seed=seed_dgp)

    # Build peer adjacency once (reuse in evaluation)
    N = int(panel["y"].shape[1])
    panel["peer_adj_m"] = b2_model.build_peer_adjacency(nbrs_m=panel["nbrs"], N=N)

    summarize_bonus2_panel(panel, init_theta=CFG_BONUS2["init_theta"])

    # Bonus2 estimation
    res = run_bonus2_estimation(CFG_BONUS2, panel=panel)

    # Evaluation (baseline is δ-only internally)
    theta_hat = res["theta_hat"]
    theta_true = _theta_true_from_panel(panel)

    mcmc: dict[str, Any] = {
        "n_saved": int(res.get("n_saved", 0)),
        "accept": res.get("accept", {}),
    }

    ev = evaluate_bonus2(
        y_mit=panel["y"],
        delta_mj=panel["delta"],
        theta_hat=theta_hat,
        theta_true=theta_true,
        w_t=panel["w"],
        season_sin_kt=panel["season_sin_kt"],
        season_cos_kt=panel["season_cos_kt"],
        peer_adj_m=panel["peer_adj_m"],
        L=int(panel["peer_lookback_L"]),
        decay=float(panel["decay"]),
        mcmc=mcmc if mcmc["accept"] else None,
    )
    print(format_evaluation_summary(ev))


if __name__ == "__main__":
    main()
