"""
run_bonus2.py

End-to-end orchestration for Bonus Q2:

- Phase 1: Zhang feature-based baseline choice model -> delta_hat (J,)
- Bonus2 DGP: simulate y_mit under habit + peer + DOW + seasonal MNL using delta_mj
- Bonus2 estimation: RW-MH over z-blocks (Ching-style architecture) -> theta_hat
- Bonus2 evaluation: NLL comparisons + parameter recovery metrics

Notes:
- We do NOT run the Lu market-shock phase.
- The within-market social network (neighbors) is treated as known and passed to the estimator.
- The decay prior hyperparameter kappa_decay is treated as known and passed to the estimator
  (computed from average_decay_rate; the estimator uses it in the Beta(kappa_decay,1) prior).
"""

from __future__ import annotations

import os
import math
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
    "T": 500,
    # time features
    "season_period": 365,
    "K": 3,
    "peer_lookback_L": 7,
    # network hyperparams (DGP)
    "avg_friends": 10.0,
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
    "mcmc_n_iter": 50,
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
        "beta_dow_m": 0.05,
        "beta_dow_j": 0.05,
        "a_m": 0.05,
        "b_m": 0.05,
        "a_j": 0.05,
        "b_j": 0.05,
    },
}


# =============================================================================
# Utilities
# =============================================================================


def _kappa_from_average_decay_rate(mu: float) -> float:
    mu = float(mu)
    if not (0.0 < mu < 1.0):
        raise ValueError("average_decay_rate must be in (0,1)")
    return mu / (1.0 - mu)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def _tile_delta_mj(delta_hat: np.ndarray, M: int) -> np.ndarray:
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    if delta_hat.ndim != 1:
        raise ValueError("delta_hat must be 1D (J,)")
    return np.tile(delta_hat[None, :], reps=(int(M), 1)).astype(np.float64)


def _apply_identifiability_constraints_np(
    beta_dow_m: np.ndarray,
    beta_dow_j: np.ndarray,
    a_j: np.ndarray,
    b_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_dow_m = np.asarray(beta_dow_m, dtype=np.float64)
    beta_dow_j = np.asarray(beta_dow_j, dtype=np.float64)
    a_j = np.asarray(a_j, dtype=np.float64)
    b_j = np.asarray(b_j, dtype=np.float64)

    if beta_dow_m.size:
        beta_dow_m = beta_dow_m - beta_dow_m.mean(axis=1, keepdims=True)

    if beta_dow_j.size:
        beta_dow_j = beta_dow_j - beta_dow_j.mean(axis=1, keepdims=True)
        beta_dow_j = beta_dow_j - beta_dow_j.mean(axis=0, keepdims=True)

    # K may be 0; operations are safe with (J,0).
    if a_j.ndim == 2:
        a_j = a_j - a_j.mean(axis=0, keepdims=True)
    if b_j.ndim == 2:
        b_j = b_j - b_j.mean(axis=0, keepdims=True)

    return beta_dow_m, beta_dow_j, a_j, b_j


def _theta_hat_centered_for_reporting(
    theta_hat: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    th = {k: np.asarray(v, dtype=np.float64) for k, v in theta_hat.items()}
    th["beta_dow_m"], th["beta_dow_j"], th["a_j"], th["b_j"] = (
        _apply_identifiability_constraints_np(
            th["beta_dow_m"], th["beta_dow_j"], th["a_j"], th["b_j"]
        )
    )
    return th


# =============================================================================
# Bonus2 summaries
# =============================================================================


def summarize_bonus2_panel(panel: dict[str, Any], init_theta: dict[str, float]) -> None:
    y = np.asarray(panel["y"], dtype=np.int64)  # (M,N,T)
    delta = np.asarray(panel["delta"], dtype=np.float64)  # (M,J)
    dow = np.asarray(panel["dow"], dtype=np.int64)  # (T,)
    sin_k_theta = np.asarray(panel["sin_k_theta"], dtype=np.float64)  # (K,T)
    nbrs = panel["nbrs"]

    M, N, T = y.shape
    J = delta.shape[1]
    K = sin_k_theta.shape[0]

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
        f"y={y.shape} | delta={delta.shape} | dow={dow.shape} | sin_k_theta={sin_k_theta.shape}"
    )
    print(f"outside_share: {outside_share:.4f} | avg_out_degree: {avg_deg:.2f}")
    print(
        f"inside_share_mean: {float(inside_shares.mean()):.4f} | inside_share_max: {float(inside_shares.max()):.4f}"
    )

    # True parameter means
    mean_beta_market = float(np.mean(panel["beta_market_mj"]))
    mean_beta_habit = float(np.mean(panel["beta_habit_j"]))
    mean_beta_peer = float(np.mean(panel["beta_peer_j"]))
    mean_decay = float(np.mean(panel["decay_rate_j"]))
    mean_dow_m = (
        float(np.mean(panel["beta_dow_m"]))
        if np.asarray(panel["beta_dow_m"]).size
        else 0.0
    )
    mean_dow_j = (
        float(np.mean(panel["beta_dow_j"]))
        if np.asarray(panel["beta_dow_j"]).size
        else 0.0
    )
    mean_a_m = float(np.mean(panel["a_m"])) if np.asarray(panel["a_m"]).size else 0.0
    mean_b_m = float(np.mean(panel["b_m"])) if np.asarray(panel["b_m"]).size else 0.0
    mean_a_j = float(np.mean(panel["a_j"])) if np.asarray(panel["a_j"]).size else 0.0
    mean_b_j = float(np.mean(panel["b_j"])) if np.asarray(panel["b_j"]).size else 0.0

    print(
        "[Bonus2] True | "
        f'mean(beta_market)= "{mean_beta_market:.4f}" , '
        f'mean(beta_habit)= "{mean_beta_habit:.4f}" , '
        f'mean(beta_peer)= "{mean_beta_peer:.4f}" , '
        f'mean(decay)= "{mean_decay:.4f}" , '
        f'mean(dow_m)= "{mean_dow_m:.4f}" , '
        f'mean(dow_j)= "{mean_dow_j:.4f}" , '
        f'mean(a_m)= "{mean_a_m:.4f}" , '
        f'mean(b_m)= "{mean_b_m:.4f}" , '
        f'mean(a_j)= "{mean_a_j:.4f}" , '
        f'mean(b_j)= "{mean_b_j:.4f}"'
    )

    print(
        "[Bonus2] Init | "
        f'mean(beta_market)= "{float(init_theta["beta_market"]):.4f}" , '
        f'mean(beta_habit)= "{float(init_theta["beta_habit"]):.4f}" , '
        f'mean(beta_peer)= "{float(init_theta["beta_peer"]):.4f}" , '
        f'mean(decay)= "{float(init_theta["decay_rate"]):.4f}" , '
        f'mean(beta_dow_m)= "{float(init_theta["beta_dow_m"]):.4f}" , '
        f'mean(beta_dow_j)= "{float(init_theta["beta_dow_j"]):.4f}" , '
        f'mean(a_m)= "{float(init_theta["a_m"]):.4f}" , '
        f'mean(b_m)= "{float(init_theta["b_m"]):.4f}" , '
        f'mean(a_j)= "{float(init_theta["a_j"]):.4f}" , '
        f'mean(b_j)= "{float(init_theta["b_j"]):.4f}"'
    )

    print(
        f'[Bonus2] Known | L={int(panel["peer_lookback_L"])} , '
        f'K={int(panel["K"])} , season_period={int(panel["season_period"])} , '
        f'kappa_decay={float(panel["kappa_decay"]):.4f}'
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


def run_bonus2_estimation(
    cfg: dict[str, Any], panel: dict[str, Any], M: int, J: int
) -> dict[str, Any]:
    print("=== Bonus2 estimation ===")

    kappa_decay = float(panel["kappa_decay"])
    eps_decay = float(panel["params_true"]["decay_rate_eps"])

    est = Bonus2Estimator(
        y_mit=panel["y"],
        delta_mj=panel["delta"],
        dow_t=panel["dow"],
        sin_k_theta=panel["sin_k_theta"],
        cos_k_theta=panel["cos_k_theta"],
        neighbors=panel["nbrs"],
        L=int(panel["peer_lookback_L"]),
        init_theta=cfg["init_theta"],
        sigmas=cfg["sigmas"],
        seed=int(cfg["mcmc_seed"]),
        kappa_decay=kappa_decay,
        eps_decay=eps_decay,
    )
    print("=== Bonus2 Estimator built ===")

    est.fit(
        n_iter=int(cfg["mcmc_n_iter"]),
        k={
            "beta_market": float(cfg["k"]["beta_market"]),
            "beta_habit": float(cfg["k"]["beta_habit"]),
            "beta_peer": float(cfg["k"]["beta_peer"]),
            "decay_rate": float(cfg["k"]["decay_rate"]),
            "beta_dow_m": float(cfg["k"]["beta_dow_m"]),
            "beta_dow_j": float(cfg["k"]["beta_dow_j"]),
            "a_m": float(cfg["k"]["a_m"]),
            "b_m": float(cfg["k"]["b_m"]),
            "a_j": float(cfg["k"]["a_j"]),
            "b_j": float(cfg["k"]["b_j"]),
        },
    )

    print("=== Bonus2 Estimator fitted ===")
    return est.get_results()


def evaluate_bonus2(
    panel: dict[str, Any], theta_hat: dict[str, np.ndarray]
) -> dict[str, Any]:
    y = tf.convert_to_tensor(np.asarray(panel["y"], dtype=np.int32))
    delta = tf.convert_to_tensor(np.asarray(panel["delta"], dtype=np.float64))
    dow = tf.convert_to_tensor(np.asarray(panel["dow"], dtype=np.int32))
    sin_k_theta = tf.convert_to_tensor(
        np.asarray(panel["sin_k_theta"], dtype=np.float64)
    )
    cos_k_theta = tf.convert_to_tensor(
        np.asarray(panel["cos_k_theta"], dtype=np.float64)
    )

    M = int(panel["y"].shape[0])
    N = int(panel["y"].shape[1])
    T = int(panel["y"].shape[2])
    J = int(panel["delta"].shape[1])
    K = int(panel["K"])
    L = tf.convert_to_tensor(int(panel["peer_lookback_L"]), dtype=tf.int32)

    eps_decay = tf.convert_to_tensor(
        float(panel["params_true"]["decay_rate_eps"]), dtype=tf.float64
    )

    peer_adj_m = b2_model.build_peer_adjacency(panel["nbrs"], N=N)

    def to_tf(theta_np: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
        return {
            k: tf.convert_to_tensor(np.asarray(v, dtype=np.float64), dtype=tf.float64)
            for k, v in theta_np.items()
        }

    # Fitted
    ll_hat = b2_model.loglik_from_theta(
        theta=to_tf(theta_hat),
        y_mit=y,
        delta_mj=delta,
        dow_t=dow,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    ).numpy()
    nll_hat = -float(ll_hat) / float(M * N * T)

    # Baseline-only (theta = 0, decay arbitrary)
    theta_base = {
        "beta_market_mj": np.zeros((M, J), dtype=np.float64),
        "beta_habit_j": np.zeros((J,), dtype=np.float64),
        "beta_peer_j": np.zeros((J,), dtype=np.float64),
        "decay_rate_j": np.full((J,), 0.5, dtype=np.float64),
        "beta_dow_m": np.zeros((M, 7), dtype=np.float64),
        "beta_dow_j": np.zeros((J, 7), dtype=np.float64),
        "a_m": np.zeros((M, K), dtype=np.float64),
        "b_m": np.zeros((M, K), dtype=np.float64),
        "a_j": np.zeros((J, K), dtype=np.float64),
        "b_j": np.zeros((J, K), dtype=np.float64),
    }
    ll_base = b2_model.loglik_from_theta(
        theta=to_tf(theta_base),
        y_mit=y,
        delta_mj=delta,
        dow_t=dow,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    ).numpy()
    nll_base = -float(ll_base) / float(M * N * T)

    # Oracle (truth)
    theta_true = {
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
    ll_true = b2_model.loglik_from_theta(
        theta=to_tf(theta_true),
        y_mit=y,
        delta_mj=delta,
        dow_t=dow,
        sin_k_theta=sin_k_theta,
        cos_k_theta=cos_k_theta,
        peer_adj_m=peer_adj_m,
        L=L,
        eps_decay=eps_decay,
    ).numpy()
    nll_true = -float(ll_true) / float(M * N * T)

    # Parameter recovery (apply identifiability constraints before comparing)
    th_hat_c = _theta_hat_centered_for_reporting(theta_hat)
    th_true_c = _theta_hat_centered_for_reporting(theta_true)

    rec = {
        "rmse_beta_market_mj": rmse(
            th_hat_c["beta_market_mj"], th_true_c["beta_market_mj"]
        ),
        "rmse_beta_habit_j": rmse(th_hat_c["beta_habit_j"], th_true_c["beta_habit_j"]),
        "rmse_beta_peer_j": rmse(th_hat_c["beta_peer_j"], th_true_c["beta_peer_j"]),
        "rmse_decay_rate_j": rmse(th_hat_c["decay_rate_j"], th_true_c["decay_rate_j"]),
        "rmse_beta_dow_m": rmse(th_hat_c["beta_dow_m"], th_true_c["beta_dow_m"]),
        "rmse_beta_dow_j": rmse(th_hat_c["beta_dow_j"], th_true_c["beta_dow_j"]),
        "rmse_a_m": (
            rmse(th_hat_c["a_m"], th_true_c["a_m"])
            if th_true_c["a_m"].size
            else float("nan")
        ),
        "rmse_b_m": (
            rmse(th_hat_c["b_m"], th_true_c["b_m"])
            if th_true_c["b_m"].size
            else float("nan")
        ),
        "rmse_a_j": (
            rmse(th_hat_c["a_j"], th_true_c["a_j"])
            if th_true_c["a_j"].size
            else float("nan")
        ),
        "rmse_b_j": (
            rmse(th_hat_c["b_j"], th_true_c["b_j"])
            if th_true_c["b_j"].size
            else float("nan")
        ),
    }

    return {
        "nll_per_obs": {
            "baseline_only": nll_base,
            "theta_hat": nll_hat,
            "oracle": nll_true,
        },
        "recovery_rmse": rec,
    }


def format_bonus2_evaluation_summary(ev: dict[str, Any]) -> str:
    nll = ev["nll_per_obs"]
    rec = ev["recovery_rmse"]

    lines = []
    lines.append("")
    lines.append("============================================================")
    lines.append("BONUS2 EVALUATION")
    lines.append("============================================================")
    lines.append(f'avg NLL per obs | baseline-only: {nll["baseline_only"]:.6f}')
    lines.append(f'avg NLL per obs | theta_hat:      {nll["theta_hat"]:.6f}')
    lines.append(f'avg NLL per obs | oracle:         {nll["oracle"]:.6f}')
    lines.append("")
    lines.append("Parameter RMSE (after identifiability centering):")
    for k in sorted(rec.keys()):
        v = rec[k]
        if np.isnan(v):
            lines.append(f"  {k}: nan")
        else:
            lines.append(f"  {k}: {v:.6f}")
    return "\n".join(lines)


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

    summarize_bonus2_panel(panel, init_theta=CFG_BONUS2["init_theta"])

    # Bonus2 estimation
    res = run_bonus2_estimation(CFG_BONUS2, panel=panel, M=M, J=J)

    # Bonus2 evaluation
    ev = evaluate_bonus2(panel=panel, theta_hat=res["theta_hat"])
    print(format_bonus2_evaluation_summary(ev))

    # Also print acceptance rates (Ching-style)
    accept = res["accept"]
    print("")
    print("Acceptance rates (per-element, averaged over saved draws):")
    for k in sorted(accept.keys()):
        print(f"  {k}: {accept[k]:.6f}")


if __name__ == "__main__":
    main()
