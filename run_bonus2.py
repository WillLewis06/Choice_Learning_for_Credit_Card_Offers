"""
run_bonus2.py

End-to-end orchestration for Bonus Q2:

- Phase 1: Zhang feature-based baseline choice model -> delta_hat (J,)
- Bonus2 DGP: simulate y_mit under habit + peer + DOW + seasonal MNL using delta_mj
- Bonus2 estimation: RW-MH over z-blocks -> theta_hat
- Bonus2 evaluation: summary via bonus2_evaluate.py

Notes:
- We do NOT run the Lu market-shock phase.
- The within-market social network (neighbors) is treated as known and passed to the estimator.
- The decay prior hyperparameter kappa_decay is treated as known and passed to the estimator.
- Time feature naming:
    season_sin_kt, season_cos_kt  (K,T)
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
from bonus2.bonus2_diagnostics import report_theta_summary, report_known_summary

# =============================================================================
# Configuration
# =============================================================================

# Phase 1: baseline choice model (Zhang feature-based)
CFG_PHASE1 = {
    "seed": 123,
    "num_products": 10,
    "num_groups": 2,
    "num_markets": 5,
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
    "width": 32,
    "heads": 8,
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "shuffle_buffer": 10_000,
    "eval_include_outside": True,
    "eval_against_empirical": True,
}

# Bonus2: DGP + estimation
CFG_BONUS2 = {
    # panel size
    "N": 200,
    "T": (365 * 15),
    # time features
    "season_period": 365,
    "K": 1,
    "peer_lookback_L": 2,
    # network hyperparams (DGP)
    "avg_friends": 5.0,
    "friends_sd": 2.0,
    # decay prior/DGP hyperparam
    "average_decay_rate": 0.875,
    # DGP parameter dispersion (required keys; no defaults in DGP)
    "params_true": {
        "habit_mean": 0.6,
        "habit_sd": 0.25,
        "peer_mean": 0.2,
        "peer_sd": 0.15,
        "mktprod_sd": 0.5,
        "dow_mkt_sd": 0.2,
        "dow_prod_sd": 0.2,
        "season_mkt_sd": 0.2,
        "season_prod_sd": 0.2,
        "decay_rate_eps": 1e-6,
    },
    # MCMC config
    "mcmc_seed": 0,
    "mcmc_n_iter": 30,
    # estimator init (scalar fills)
    "init_theta": {
        "beta_market": 0.0,
        "beta_habit": 0.0,
        "beta_peer": 0.0,
        "decay_rate": 0.875,
        "beta_dow_m": 0.0,
        "beta_dow_j": 0.0,
        "a_m": 0.0,
        "b_m": 0.0,
        "a_j": 0.0,
        "b_j": 0.0,
    },
    # z-space prior scales (keys must match Bonus2Estimator expectation)
    "sigmas": {
        "z_beta_market_mj": 2.0,
        "z_beta_habit_j": 2.0,
        "z_beta_peer_j": 2.0,
        "z_decay_rate_j": 2.0,
        "z_beta_dow_m": 2.0,
        "z_beta_dow_j": 2.0,
        "z_a_m": 2.0,
        "z_b_m": 2.0,
        "z_a_j": 2.0,
        "z_b_j": 2.0,
    },
    # RW step sizes (keys must match Bonus2Estimator.fit expectation)
    "k": {
        "beta_market": 0.05,
        "beta_habit": 0.05,
        "beta_peer": 0.05,
        "decay_rate": 0.02,
        "beta_dow_m": 0.001,
        "beta_dow_j": 0.001,
        "a_m": 0.05,
        "b_m": 0.05,
        "a_j": 0.05,
        "b_j": 0.05,
    },
}


# =============================================================================
# Utilities
# =============================================================================


def _tile_delta_mj(delta_hat: np.ndarray, M: int) -> np.ndarray:
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    if delta_hat.ndim != 1:
        raise ValueError("delta_hat must be 1D (J,)")
    return np.tile(delta_hat[None, :], reps=(int(M), 1)).astype(np.float64)


def _delta_only_baseline_probs_mc(delta_mj: np.ndarray) -> np.ndarray:
    """
    Construct δ-only baseline probabilities:
      p_m(c) = softmax([0, δ_m1, ..., δ_mJ]) over c=0..J,
    returning (M,J+1).
    """
    delta_mj = np.asarray(delta_mj, dtype=np.float64)
    if delta_mj.ndim != 2:
        raise ValueError("delta_mj must be 2D (M,J)")
    M, J = (int(delta_mj.shape[0]), int(delta_mj.shape[1]))
    if M < 1 or J < 1:
        raise ValueError(f"delta_mj must have M>=1 and J>=1, got {delta_mj.shape}")

    logits = np.concatenate(
        [np.zeros((M, 1), dtype=np.float64), delta_mj], axis=1
    )  # (M,J+1)
    logits = logits - logits.max(axis=1, keepdims=True)  # stable softmax
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (M,J+1)


def _broadcast_market_probs_to_mntc(p_mc: np.ndarray, N: int, T: int) -> np.ndarray:
    """Broadcast market-level probs (M,C) to panel probs (M,N,T,C)."""
    p_mc = np.asarray(p_mc, dtype=np.float64)
    if p_mc.ndim != 2:
        raise ValueError("p_mc must be 2D (M,C)")
    M, C = (int(p_mc.shape[0]), int(p_mc.shape[1]))
    if M < 1 or C < 1:
        raise ValueError(f"p_mc must have M>=1 and C>=1, got {p_mc.shape}")
    if int(N) < 1 or int(T) < 1:
        raise ValueError(f"N and T must be >=1, got N={N}, T={T}")
    return np.broadcast_to(p_mc[:, None, None, :], (M, int(N), int(T), C)).copy()


def _theta_true_from_panel(panel: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "beta_market_mj": np.asarray(panel["beta_market_mj"], dtype=np.float64),
        "beta_habit_j": np.asarray(panel["beta_habit_j"], dtype=np.float64),
        "beta_peer_j": np.asarray(panel["beta_peer_j"], dtype=np.float64),
        "decay_rate_j": np.asarray(panel["decay_rate_j"], dtype=np.float64),
        "beta_dow_m": np.asarray(panel["beta_dow_m"], dtype=np.float64),
        "beta_dow_j": np.asarray(panel["beta_dow_j"], dtype=np.float64),
        "a_m": np.asarray(panel["a_m"], dtype=np.float64),
        "b_m": np.asarray(panel["b_m"], dtype=np.float64),
        "a_j": np.asarray(panel["a_j"], dtype=np.float64),
        "b_j": np.asarray(panel["b_j"], dtype=np.float64),
    }


def _to_tf_theta(theta_np: dict[str, Any]) -> dict[str, tf.Tensor]:
    return {
        k: tf.convert_to_tensor(np.asarray(v, dtype=np.float64), dtype=tf.float64)
        for k, v in theta_np.items()
    }


# =============================================================================
# Bonus2 summaries
# =============================================================================


def summarize_bonus2_panel(panel: dict[str, Any], init_theta: dict[str, float]) -> None:
    y = np.asarray(panel["y"], dtype=np.int64)  # (M,N,T)
    delta = np.asarray(panel["delta"], dtype=np.float64)  # (M,J)
    dow = np.asarray(panel["dow"], dtype=np.int64)  # (T,)
    season_sin_kt = np.asarray(panel["season_sin_kt"], dtype=np.float64)  # (K,T)
    nbrs = panel["nbrs"]

    M, N, T = y.shape
    J = delta.shape[1]

    outside_share = float(np.mean(y == 0))
    inside_shares = np.array(
        [np.mean(y == (j + 1)) for j in range(J)], dtype=np.float64
    )

    # average out-degree
    degs = []
    for m in range(M):
        rows = nbrs[m]
        for i in range(N):
            degs.append(int(len(rows[i])))
    avg_deg = float(np.mean(degs)) if degs else 0.0

    print("=== Bonus2 data generated ===")
    print(
        "shapes: "
        f"y={y.shape} | delta={delta.shape} | dow={dow.shape} | season_sin_kt={season_sin_kt.shape}"
    )
    print(f"outside_share: {outside_share:.4f} | avg_out_degree: {avg_deg:.2f}")
    print(
        f"inside_share_mean: {float(inside_shares.mean()):.4f} | inside_share_max: {float(inside_shares.max()):.4f}"
    )

    theta_true = _theta_true_from_panel(panel)
    report_theta_summary("True", theta_true)
    report_theta_summary("Init", init_theta)
    report_known_summary(
        L=int(panel["peer_lookback_L"]),
        K=int(panel["K"]),
        season_period=int(panel["season_period"]),
        kappa_decay=float(panel["kappa_decay"]),
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
    print("=== Bonus2 DGP: Generating y_mit under habit + peer + DOW + seasonality ===")

    panel = simulate_bonus2_dgp(
        delta=delta_mj,
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        avg_friends=float(cfg["avg_friends"]),
        friends_sd=float(cfg["friends_sd"]),
        params_true=cfg["params_true"],
        average_decay_rate=float(cfg["average_decay_rate"]),
        seed=seed,
        season_period=int(cfg["season_period"]),
        K=int(cfg["K"]),
        peer_lookback_L=int(cfg["peer_lookback_L"]),
    )

    print("=== Bonus2 DGP complete ===")
    return panel


def run_bonus2_estimation(cfg: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    print("=== Bonus2 estimation ===")

    kappa_decay = float(panel["kappa_decay"])
    decay_rate_eps = float(panel["params_true"]["decay_rate_eps"])

    est = Bonus2Estimator(
        y_mit=panel["y"],
        delta_mj=panel["delta"],
        dow_t=panel["dow"],
        season_sin_kt=panel["season_sin_kt"],
        season_cos_kt=panel["season_cos_kt"],
        neighbors=panel["nbrs"],
        L=int(panel["peer_lookback_L"]),
        init_theta=cfg["init_theta"],
        sigmas=cfg["sigmas"],
        seed=int(cfg["mcmc_seed"]),
        kappa_decay=kappa_decay,
        decay_rate_eps=decay_rate_eps,
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
    M = int(CFG_PHASE1["num_markets"])
    J = int(CFG_PHASE1["num_products"])
    if int(delta_hat.shape[0]) != J:
        raise ValueError(
            "Phase 1 delta_hat length does not match CFG_PHASE1['num_products']"
        )

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

    # Build probabilities for evaluation (fitted + oracle; baseline is δ-only in NumPy)
    y_tf = tf.convert_to_tensor(panel["y"], dtype=tf.int32)
    delta_tf = tf.convert_to_tensor(panel["delta"], dtype=tf.float64)
    dow_tf = tf.convert_to_tensor(panel["dow"], dtype=tf.int32)
    season_sin_tf = tf.convert_to_tensor(panel["season_sin_kt"], dtype=tf.float64)
    season_cos_tf = tf.convert_to_tensor(panel["season_cos_kt"], dtype=tf.float64)

    T = int(panel["y"].shape[2])
    L_tf = tf.convert_to_tensor(int(panel["peer_lookback_L"]), dtype=tf.int32)
    decay_rate_eps_tf = tf.convert_to_tensor(
        float(panel["params_true"]["decay_rate_eps"]), dtype=tf.float64
    )

    peer_adj_m = panel["peer_adj_m"]

    theta_hat = res["theta_hat"]
    theta_true = _theta_true_from_panel(panel)

    p_hat = b2_model.predict_choice_probs_from_theta(
        theta=_to_tf_theta(theta_hat),
        y_mit=y_tf,
        delta_mj=delta_tf,
        dow_t=dow_tf,
        season_sin_kt=season_sin_tf,
        season_cos_kt=season_cos_tf,
        peer_adj_m=peer_adj_m,
        L=L_tf,
        decay_rate_eps=decay_rate_eps_tf,
    ).numpy()

    p_oracle = b2_model.predict_choice_probs_from_theta(
        theta=_to_tf_theta(theta_true),
        y_mit=y_tf,
        delta_mj=delta_tf,
        dow_t=dow_tf,
        season_sin_kt=season_sin_tf,
        season_cos_kt=season_cos_tf,
        peer_adj_m=peer_adj_m,
        L=L_tf,
        decay_rate_eps=decay_rate_eps_tf,
    ).numpy()

    # δ-only baseline probabilities for evaluation (compact -> broadcast)
    p_delta_only_mc = _delta_only_baseline_probs_mc(panel["delta"])  # (M,J+1)
    p_delta_only = _broadcast_market_probs_to_mntc(p_delta_only_mc, N=N, T=T)

    # Bonus2 evaluation
    mcmc: dict[str, Any] = {}
    if "n_saved" in res:
        mcmc["n_saved"] = int(res["n_saved"])
    if "accept" in res:
        mcmc["accept"] = res["accept"]
    elif "accept_rates" in res:
        mcmc["accept"] = res["accept_rates"]

    ev = evaluate_bonus2(
        y_mit=panel["y"],
        p_choice_hat_mntc=p_hat,
        p_choice_baseline_mntc=p_delta_only,
        theta_hat=theta_hat,
        theta_true=theta_true,
        p_choice_oracle_mntc=p_oracle,
        mcmc=mcmc if mcmc else None,
    )
    print(format_evaluation_summary(ev))


if __name__ == "__main__":
    main()
