"""
run_bonus2.py

End-to-end orchestration for Bonus Q2.

Pipeline
--------
1) Phase 1 baseline choice model (Zhang feature-based) -> delta_hat (J,)
2) Bonus2 DGP -> {panel, theta_true}
   - panel is in canonical estimator schema and is validated inside the DGP
3) Bonus2 estimation (RW-MH) -> theta_hat
4) Predict probabilities via bonus2_model and evaluate vs delta-only baseline

Notes
-----
- External input validation is centralized in bonus2_input_validation.py.
- No fallbacks are used in orchestration: configs must provide all required keys.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Keep TF logs quiet for downstream modules that import TF internally.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from datasets.bonus2_dgp import simulate_bonus2_dgp  # noqa: E402
from run_zhang_with_lu import (
    print_choice_model_diagnostics,
    run_choice_model,
)  # noqa: E402

from bonus2 import bonus2_model as b2_model  # noqa: E402
from bonus2.bonus2_diagnostics import (
    report_known_summary,
    report_theta_summary,
)  # noqa: E402
from bonus2.bonus2_estimator import Bonus2Estimator  # noqa: E402
from bonus2.bonus2_evaluate import (
    evaluate_bonus2,
    format_evaluation_summary,
)  # noqa: E402
from bonus2.bonus2_input_validation import validate_phase1_delta_hat  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

CFG_PHASE1: dict[str, Any] = {
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

CFG_BONUS2: dict[str, Any] = {
    # panel size
    "N": 100,
    "T": 365 * 10,
    # observed time features
    "season_period": 365,
    "K": 1,
    "lookback": 1,
    # known scalar habit decay
    "decay": 0.8,
    # network hyperparams (DGP)
    "avg_friends": 3.0,
    "friends_sd": 1.0,
    # DGP hyperparams
    "params_true": {
        "habit_mean": 0.60,
        "habit_sd": 0.05,
        "peer_mean": 0.50,
        "peer_sd": 0.15,
        "mktprod_sd": 0.50,
        "weekend_prod_sd": 0.20,
        "season_mkt_sd": 0.20,
    },
    # MCMC
    "mcmc_seed": 0,
    "mcmc_n_iter": 200,
    # estimator init (scalar fills)
    "init_theta": {
        "beta_intercept": 0.0,
        "beta_habit": 0.0,
        "beta_peer": 0.0,
        "beta_weekend_weekday": 0.0,
        "beta_weekend_weekend": 0.0,
        "a_m": 0.0,
        "b_m": 0.0,
    },
    # z-space prior scales (keys must match estimator Z_KEYS)
    "sigmas": {
        "z_beta_intercept_j": 2.0,
        "z_beta_habit_j": 2.0,
        "z_beta_peer_j": 2.0,
        "z_beta_weekend_jw": 2.0,
        "z_a_m": 2.0,
        "z_b_m": 2.0,
    },
    # RW step sizes in z-space (keys must match estimator Z_KEYS)
    "step_size_z": {
        "z_beta_intercept_j": 0.02,
        "z_beta_habit_j": 0.02,
        "z_beta_peer_j": 0.02,
        "z_beta_weekend_jw": 0.01,
        "z_a_m": 0.05,
        "z_b_m": 0.05,
    },
    # evaluation numerics
    "eps": 1e-12,
}

# =============================================================================
# Utilities
# =============================================================================


def _tile_delta_mj(delta_hat: np.ndarray, M: int) -> np.ndarray:
    """Tile phase-1 delta_hat (J,) into delta_mj (M,J)."""
    dh = np.asarray(delta_hat, dtype=np.float64)
    return np.tile(dh[None, :], reps=(int(M), 1)).astype(np.float64)


def _theta_init_full(
    init_theta: dict[str, float], M: int, J: int, K: int
) -> dict[str, np.ndarray]:
    """Expand scalar init_theta fills into full theta arrays (canonical keys)."""
    return {
        "beta_intercept_j": np.full(
            (J,), float(init_theta["beta_intercept"]), dtype=np.float64
        ),
        "beta_habit_j": np.full(
            (J,), float(init_theta["beta_habit"]), dtype=np.float64
        ),
        "beta_peer_j": np.full((J,), float(init_theta["beta_peer"]), dtype=np.float64),
        "beta_weekend_jw": np.stack(
            [
                np.full(
                    (J,), float(init_theta["beta_weekend_weekday"]), dtype=np.float64
                ),
                np.full(
                    (J,), float(init_theta["beta_weekend_weekend"]), dtype=np.float64
                ),
            ],
            axis=1,
        ),
        "a_m": np.full((M, K), float(init_theta["a_m"]), dtype=np.float64),
        "b_m": np.full((M, K), float(init_theta["b_m"]), dtype=np.float64),
    }


def _theta_np_to_tf(theta_np: dict[str, Any]) -> dict[str, tf.Tensor]:
    """Convert numpy theta dict to float64 TF tensors."""
    return {
        k: tf.convert_to_tensor(np.asarray(v, dtype=np.float64), dtype=tf.float64)
        for k, v in theta_np.items()
    }


# =============================================================================
# Reporting
# =============================================================================


def summarize_bonus2_panel(
    panel: dict[str, Any],
    theta_true: dict[str, Any],
    init_theta: dict[str, float],
    season_period: int,
    K: int,
) -> None:
    """Print a compact dataset + parameter summary for examiner-facing logs."""
    y = np.asarray(panel["y_mit"], dtype=np.int64)
    delta = np.asarray(panel["delta_mj"], dtype=np.float64)
    w = np.asarray(panel["is_weekend_t"], dtype=np.int64)
    sin_kt = np.asarray(panel["season_sin_kt"], dtype=np.float64)
    neighbors_m = panel["neighbors_m"]

    M, N, T = y.shape
    J = int(delta.shape[1])

    outside_share = float(np.mean(y == 0))
    inside_shares = np.array(
        [np.mean(y == (j + 1)) for j in range(J)], dtype=np.float64
    )

    degs: list[int] = []
    for m in range(M):
        rows = neighbors_m[m]
        for i in range(N):
            degs.append(int(len(rows[i])))
    avg_deg = float(np.mean(degs)) if degs else 0.0

    print("=== Bonus2 data generated ===")
    print(
        "shapes: "
        f"y_mit={y.shape} | delta_mj={delta.shape} | is_weekend_t={w.shape} | season_sin_kt={sin_kt.shape}"
    )
    print(f"outside_share: {outside_share:.4f} | avg_out_degree: {avg_deg:.2f}")
    print(
        f"inside_share_mean: {float(inside_shares.mean()):.4f} | inside_share_max: {float(inside_shares.max()):.4f}"
    )

    theta_init = _theta_init_full(init_theta=init_theta, M=M, J=J, K=int(K))

    report_theta_summary("True", _theta_np_to_tf(theta_true))
    report_theta_summary("Init", _theta_np_to_tf(theta_init))

    report_known_summary(
        lookback=tf.constant(int(panel["lookback"]), dtype=tf.int32),
        K=tf.constant(int(K), dtype=tf.int32),
        season_period=tf.constant(int(season_period), dtype=tf.int32),
        decay=tf.constant(float(panel["decay"]), dtype=tf.float64),
    )


# =============================================================================
# Phase runners
# =============================================================================


def run_phase1(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 1 baseline choice model and return delta_hat."""
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
    cfg: dict[str, Any], delta_mj: np.ndarray, seed: int
) -> dict[str, Any]:
    """Run Bonus2 DGP and return {panel, theta_true} plus metadata."""
    print(
        "=== Bonus2 DGP: Generating y_mit under habit + peer + weekend + market seasonality ==="
    )

    out = simulate_bonus2_dgp(
        delta_mj=delta_mj,
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        avg_friends=float(cfg["avg_friends"]),
        friends_sd=float(cfg["friends_sd"]),
        params_true=cfg["params_true"],
        decay=float(cfg["decay"]),
        seed=int(seed),
        season_period=int(cfg["season_period"]),
        K=int(cfg["K"]),
        lookback=int(cfg["lookback"]),
    )

    panel = out["panel"]
    theta_true = out["theta_true"]

    print("=== Bonus2 DGP complete ===")
    return {
        "panel": panel,
        "theta_true": theta_true,
        "season_period": int(cfg["season_period"]),
        "K": int(cfg["K"]),
    }


def run_bonus2_estimation(cfg: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    """Run RW-MH estimation for Bonus2 and return estimator results."""
    print("=== Bonus2 estimation ===")

    est = Bonus2Estimator(
        y_mit=panel["y_mit"],
        delta_mj=panel["delta_mj"],
        is_weekend_t=panel["is_weekend_t"],
        season_sin_kt=panel["season_sin_kt"],
        season_cos_kt=panel["season_cos_kt"],
        neighbors_m=panel["neighbors_m"],
        lookback=int(panel["lookback"]),
        decay=float(panel["decay"]),
        init_theta=cfg["init_theta"],
        sigmas=cfg["sigmas"],
        step_size_z=cfg["step_size_z"],
        seed=int(cfg["mcmc_seed"]),
    )

    print("=== Bonus2 Estimator built ===")
    est.fit(n_iter=int(cfg["mcmc_n_iter"]))
    print("=== Bonus2 Estimator fitted ===")

    return est.get_results()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # Phase 1 (Zhang baseline)
    out1 = run_phase1(CFG_PHASE1)
    delta_hat = out1["delta_hat"]
    validate_phase1_delta_hat(delta_hat, num_products=int(CFG_PHASE1["num_products"]))

    # Build delta_mj for Bonus2 (fixed baseline utilities)
    M = int(CFG_PHASE1["num_markets"])
    delta_mj = _tile_delta_mj(delta_hat=delta_hat, M=M)

    # Bonus2 DGP
    seed_dgp = int(CFG_PHASE1["seed"]) + 999
    dgp_out = run_bonus2_dgp(CFG_BONUS2, delta_mj=delta_mj, seed=seed_dgp)

    panel = dgp_out["panel"]
    theta_true = dgp_out["theta_true"]
    season_period = int(dgp_out["season_period"])
    K = int(dgp_out["K"])

    summarize_bonus2_panel(
        panel=panel,
        theta_true=theta_true,
        init_theta=CFG_BONUS2["init_theta"],
        season_period=season_period,
        K=K,
    )

    # Bonus2 estimation
    res = run_bonus2_estimation(CFG_BONUS2, panel=panel)
    theta_hat = res["theta_hat"]

    # Predict probabilities explicitly via the model (for evaluation)
    y_tf = tf.convert_to_tensor(
        np.asarray(panel["y_mit"], dtype=np.int32), dtype=tf.int32
    )
    delta_tf = tf.convert_to_tensor(
        np.asarray(panel["delta_mj"], dtype=np.float64), dtype=tf.float64
    )
    w_tf = tf.convert_to_tensor(
        np.asarray(panel["is_weekend_t"], dtype=np.int32), dtype=tf.int32
    )
    sin_tf = tf.convert_to_tensor(
        np.asarray(panel["season_sin_kt"], dtype=np.float64), dtype=tf.float64
    )
    cos_tf = tf.convert_to_tensor(
        np.asarray(panel["season_cos_kt"], dtype=np.float64), dtype=tf.float64
    )

    N = int(np.asarray(panel["y_mit"]).shape[1])
    peer_adj_m = b2_model.build_peer_adjacency(
        neighbors_m=panel["neighbors_m"], n_consumers=N
    )

    lookback_tf = tf.constant(int(panel["lookback"]), dtype=tf.int32)
    decay_tf = tf.constant(float(panel["decay"]), dtype=tf.float64)

    p_choice_hat = b2_model.predict_choice_probs_from_theta(
        theta=_theta_np_to_tf(theta_hat),
        y_mit=y_tf,
        delta_mj=delta_tf,
        is_weekend_t=w_tf,
        season_sin_kt=sin_tf,
        season_cos_kt=cos_tf,
        peer_adj_m=peer_adj_m,
        lookback=lookback_tf,
        decay=decay_tf,
    ).numpy()

    p_choice_oracle = b2_model.predict_choice_probs_from_theta(
        theta=_theta_np_to_tf(theta_true),
        y_mit=y_tf,
        delta_mj=delta_tf,
        is_weekend_t=w_tf,
        season_sin_kt=sin_tf,
        season_cos_kt=cos_tf,
        peer_adj_m=peer_adj_m,
        lookback=lookback_tf,
        decay=decay_tf,
    ).numpy()

    mcmc = {"n_saved": int(res.get("n_saved", 0)), "accept": res.get("accept", {})}
    mcmc_out = mcmc if mcmc.get("accept") else None

    ev = evaluate_bonus2(
        y_mit=np.asarray(panel["y_mit"], dtype=np.int64),
        delta_mj=np.asarray(panel["delta_mj"], dtype=np.float64),
        p_choice_hat_mntc=p_choice_hat,
        p_choice_oracle_mntc=p_choice_oracle,
        theta_hat=theta_hat,
        theta_true=theta_true,
        mcmc=mcmc_out,
        eps=float(CFG_BONUS2["eps"]),
    )
    print(format_evaluation_summary(ev))


if __name__ == "__main__":
    main()
