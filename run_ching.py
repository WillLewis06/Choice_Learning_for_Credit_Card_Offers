"""
run_ching.py

End-to-end orchestration for the Phase-3 stockpiling (Ching-style) layer on top of
Phase 1–2 utilities (run_zhang_with_lu).

Key objects:
  - Phase 1 output: delta_hat (J,)
  - Phase 2 output: E_bar_hat (M,), njt_hat (M,J)
  - Phase 3 intercepts: u_mj = delta_hat[None,:] + E_bar_hat[:,None] + njt_hat

Phase 3 observed panel:
  - a_mnjt      (M,N,J,T) purchases
  - p_state_mjt (M,J,T)   exogenous price states

Known to estimator:
  - P_price_mj    (M,J,S,S) price Markov transitions
  - price_vals_mj (M,J,S)   price levels by state
  - u_mj          (M,J)     intercepts
  - lambda_mn     (M,N)     consumption probabilities (treated as known input)

Phase 3 estimated parameters (MCMC):
  - beta      (1,)
  - alpha     (J,)
  - v         (J,)
  - fc        (J,)
  - u_scale   (M,)   (estimation-only nuisance; DGP uses u_mj unscaled)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Set TF log level before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

import ching.stockpiling_model as sp_model
from datasets.ching_dgp import generate_dgp

from ching.stockpiling_estimator import StockpilingEstimator
from ching.stockpiling_evaluate import evaluate_stockpiling, format_evaluation_summary
from ching.stockpiling_posterior import StockpilingInputs, predict_p_buy_mnjt_from_theta

# Phase 1–2 orchestration
from run_zhang_with_lu import (
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
    run_market_shock_estimator,
)


# =============================================================================
# Configuration
# =============================================================================

CFG_PHASE1: dict[str, Any] = {
    "seed": 123,
    "num_products": 10,
    "num_groups": 2,
    "num_markets": 5,
    "N_base": 2_000,
    "N_shock": 1_000,
    "num_features": 10,
    "x_sd": 1.0,
    "coef_sd": 1.0,
    "p_g_active": 0.8,
    "g_sd": 0.5,
    "sd_E": 0.5,
    "p_active": 0.5,
    "sd_u": 0.5,
    "depth": 2,
    "width": 64,
    "heads": 4,
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "shuffle_buffer": 1_000,
    "eval_include_outside": True,
    "eval_against_empirical": True,
}

CFG_PHASE2: dict[str, Any] = {
    "seed": 0,
    "n_iter": 50,
    "pilot_length": 20,
    "max_rounds": 50,
    "target_low": 0.30,
    "target_high": 0.50,
    "factor_rw": 1.2,
    "factor_tmh": 1.2,
    "ridge": 1e-6,
}

CFG_PHASE3: dict[str, Any] = {
    "N": 500,
    "T": 500,
    "I_max": 10,
    "S": 2,
    "waste_cost": 1.0,
    "dp_tol": 1e-5,
    "dp_max_iter": 200,
    # price process construction
    "price_seed": 777,
    "p_stay": 0.85,
    "P_noise_sd": 0.05,
    "P_min_prob": 1e-6,
    "price_base_low": 0.7,
    "price_base_high": 1.3,
    "discount_low": 0.10,
    "discount_high": 0.35,
    "price_noise_sd": 0.02,
    # MCMC
    "mcmc_seed": 0,
    "mcmc_n_iter": 200,
    "init_theta": {
        "beta": 0.5,
        "alpha": 0.5,
        "v": 0.5,
        "fc": 2.0,
        "u_scale": 1.0,
    },
    # NOTE: StockpilingEstimator requires z-space prior scales keyed by z_*.
    "sigmas": {
        "z_beta": 2.0,
        "z_alpha": 2.0,
        "z_v": 2.0,
        "z_fc": 2.0,
        "z_u_scale": 2.0,
    },
    "k": {
        "beta": 0.1,
        "alpha": 0.05,
        "v": 0.2,
        "fc": 0.05,
        # freeze u_scale by setting to 0.0
        "u_scale": 0.0,
    },
}


# =============================================================================
# Helpers
# =============================================================================


def uniform_pi_I0(I_max: int) -> np.ndarray:
    """Uniform initial inventory distribution over {0,...,I_max}."""
    I_max = int(I_max)
    pi = np.ones(I_max + 1, dtype=np.float64)
    return pi / float(pi.sum())


def row_normalize(P: np.ndarray, min_prob: float) -> np.ndarray:
    """Row-normalize a 2D array after clipping entries below min_prob."""
    P = np.asarray(P, dtype=np.float64)
    P = np.maximum(P, float(min_prob))
    return P / P.sum(axis=-1, keepdims=True)


def build_price_transitions(
    rng: np.random.Generator,
    M: int,
    J: int,
    S: int,
    p_stay: float,
    noise_sd: float,
    min_prob: float,
) -> np.ndarray:
    """Construct (M,J,S,S) row-stochastic Markov transitions with mild randomization."""
    M, J, S = int(M), int(J), int(S)
    P = np.zeros((M, J, S, S), dtype=np.float64)

    p_stay = float(p_stay)
    base = np.full((S, S), (1.0 - p_stay) / max(S - 1, 1), dtype=np.float64)
    np.fill_diagonal(base, p_stay)

    for m in range(M):
        for j in range(J):
            noise = rng.normal(0.0, float(noise_sd), size=(S, S))
            P[m, j] = row_normalize(base + noise, min_prob=min_prob)

    return P


def build_price_levels(
    rng: np.random.Generator,
    M: int,
    J: int,
    S: int,
    base_low: float,
    base_high: float,
    discount_low: float,
    discount_high: float,
    noise_sd: float,
) -> np.ndarray:
    """Construct (M,J,S) price levels with per-(m,j) base and state discounts."""
    M, J, S = int(M), int(J), int(S)
    base = rng.uniform(float(base_low), float(base_high), size=(M, J, 1))
    disc = rng.uniform(float(discount_low), float(discount_high), size=(M, J, S))
    noise = rng.normal(0.0, float(noise_sd), size=(M, J, S))
    return np.maximum(1e-6, base * (1.0 - disc) + noise).astype(np.float64, copy=False)


def build_price_processes(
    M: int,
    J: int,
    S: int,
    seed_price: int,
    p_stay: float,
    P_noise_sd: float,
    P_min_prob: float,
    price_base_low: float,
    price_base_high: float,
    discount_low: float,
    discount_high: float,
    price_noise_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Markov transitions and price levels for each (market, product)."""
    rng = np.random.default_rng(int(seed_price))
    P_price_mj = build_price_transitions(
        rng=rng,
        M=M,
        J=J,
        S=S,
        p_stay=p_stay,
        noise_sd=P_noise_sd,
        min_prob=P_min_prob,
    )
    price_vals_mj = build_price_levels(
        rng=rng,
        M=M,
        J=J,
        S=S,
        base_low=price_base_low,
        base_high=price_base_high,
        discount_low=discount_low,
        discount_high=discount_high,
        noise_sd=price_noise_sd,
    )
    return P_price_mj, price_vals_mj


def summarize_stockpiling_panel(
    panel: dict[str, object], init_theta: dict[str, float]
) -> None:
    """Print basic panel diagnostics and true/init means (excluding lambda)."""
    a_mnjt = np.asarray(panel["a_mnjt"])
    p_state_mjt = np.asarray(panel["p_state_mjt"])
    u_mj = np.asarray(panel["u_mj"])

    print("=== Stockpiling data generated ===")
    print(
        "shapes: "
        f"a_mnjt={a_mnjt.shape} | p_state_mjt={p_state_mjt.shape} | u_mj={u_mj.shape}"
    )
    print(f"overall buy rate: {float(np.mean(a_mnjt)):.4f}")

    theta_true = panel.get("theta_true", None)
    if isinstance(theta_true, dict):
        print(
            "[Stockpiling] True | "
            f'mean(beta)= "{float(np.mean(theta_true["beta"])):.4f}" , '
            f'mean(alpha)= "{float(np.mean(theta_true["alpha"])):.4f}" , '
            f'mean(v)= "{float(np.mean(theta_true["v"])):.4f}" , '
            f'mean(fc)= "{float(np.mean(theta_true["fc"])):.4f}"'
        )

    print(
        "[Stockpiling] Init | "
        f'mean(beta)= "{float(init_theta["beta"]):.4f}" , '
        f'mean(alpha)= "{float(init_theta["alpha"]):.4f}" , '
        f'mean(v)= "{float(init_theta["v"]):.4f}" , '
        f'mean(fc)= "{float(init_theta["fc"]):.4f}" , '
        f'mean(u_scale)= "{float(init_theta["u_scale"]):.4f}"'
    )


# =============================================================================
# Phase runners
# =============================================================================


def run_phase1(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 1 baseline choice model and print diagnostics."""
    print("=== Phase 1: Baseline choice model ===")
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
        g_sd=None if cfg["g_sd"] is None else float(cfg["g_sd"]),
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
    print_choice_model_diagnostics(
        delta_hat=out1["delta_hat"],
        delta_true=dgp["delta_true"],
        qj_base=dgp["qj_base"],
        q0_base=int(dgp["q0_base"]),
        p_base=dgp["p_base"],
        p0_base=float(dgp["p0_base"]),
        N_base=int(cfg["N_base"]),
        eval_include_outside=bool(cfg["eval_include_outside"]),
        eval_against_empirical=bool(cfg["eval_against_empirical"]),
    )
    return out1


def run_phase2(
    cfg: dict[str, Any],
    dgp: dict[str, Any],
    delta_hat: np.ndarray,
    eval_include_outside: bool,
) -> dict[str, Any]:
    """Run Phase 2 Lu-style market-shock estimator and print diagnostics."""
    print("=== Phase 2: Market-shock estimator ===")
    res2 = run_market_shock_estimator(
        delta_hat=np.asarray(delta_hat, dtype=np.float64),
        qjt_shock=dgp["qjt_shock"],
        q0t_shock=dgp["q0t_shock"],
        seed=int(cfg["seed"]),
        n_iter=int(cfg["n_iter"]),
        pilot_length=int(cfg["pilot_length"]),
        max_rounds=int(cfg["max_rounds"]),
        target_low=float(cfg["target_low"]),
        target_high=float(cfg["target_high"]),
        factor_rw=float(cfg["factor_rw"]),
        factor_tmh=float(cfg["factor_tmh"]),
        ridge=float(cfg["ridge"]),
    )
    print_market_shock_diagnostics(
        delta_hat=np.asarray(delta_hat, dtype=np.float64),
        dgp=dgp,
        res=res2,
        eval_include_outside=bool(eval_include_outside),
    )
    return res2


def build_phase3_inputs(delta_hat: np.ndarray, res2: dict[str, Any]) -> dict[str, Any]:
    """Construct Phase-3 fixed inputs from Phase 1–2 outputs."""
    return {
        "delta_used": np.asarray(delta_hat, dtype=np.float64),
        "E_bar_used": np.asarray(res2["E_bar_hat"], dtype=np.float64),
        "njt_used": np.asarray(res2["njt_hat"], dtype=np.float64),
    }


def run_phase3_dgp(
    cfg: dict[str, Any],
    delta_used: np.ndarray,
    E_bar_used: np.ndarray,
    njt_used: np.ndarray,
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    seed_dgp: int,
) -> dict[str, Any]:
    """Generate the Phase-3 seller-observed panel via datasets.ching_dgp.generate_dgp."""
    a_mnjt, p_state_mjt, u_mj, theta_true = generate_dgp(
        seed=int(seed_dgp),
        delta_true=np.asarray(delta_used, dtype=np.float64),
        E_bar_true=np.asarray(E_bar_used, dtype=np.float64),
        njt_true=np.asarray(njt_used, dtype=np.float64),
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        I_max=int(cfg["I_max"]),
        P_price_mj=np.asarray(P_price_mj, dtype=np.float64),
        price_vals_mj=np.asarray(price_vals_mj, dtype=np.float64),
        waste_cost=float(cfg["waste_cost"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
    )
    theta_true = dict(theta_true)
    lambda_mn = np.asarray(theta_true["lambda"], dtype=np.float64)

    return {
        "a_mnjt": a_mnjt,
        "p_state_mjt": p_state_mjt,
        "u_mj": u_mj,
        "lambda_mn": lambda_mn,
        "theta_true": theta_true,
    }


def run_phase3_estimation(
    cfg: dict[str, Any],
    panel: dict[str, Any],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
) -> dict[str, Any]:
    """Fit stockpiling parameters from observed data, treating lambda_mn as known."""
    est = StockpilingEstimator(
        a_mnjt=np.asarray(panel["a_mnjt"]),
        p_state_mjt=np.asarray(panel["p_state_mjt"]),
        u_mj=np.asarray(panel["u_mj"]),
        lambda_mn=np.asarray(panel["lambda_mn"]),
        P_price_mj=np.asarray(P_price_mj),
        price_vals_mj=np.asarray(price_vals_mj),
        pi_I0=uniform_pi_I0(int(cfg["I_max"])),
        I_max=int(cfg["I_max"]),
        waste_cost=float(cfg["waste_cost"]),
        sigmas=dict(cfg["sigmas"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
        rng_seed=int(cfg["mcmc_seed"]),
    )

    res = est.fit(
        n_iter=int(cfg["mcmc_n_iter"]),
        init_theta=dict(cfg["init_theta"]),
        k=dict(cfg["k"]),
    )

    return {
        "theta_hat": res["theta_mean"],
        "accept": res["accept"],
        "n_saved": res["n_saved"],
        "z_last": res["z_last"],
    }


def run_phase3_evaluation(
    cfg: dict[str, Any],
    panel: dict[str, Any],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    theta_hat: dict[str, np.ndarray],
    theta_true: dict[str, np.ndarray],
    mcmc_diag: dict[str, Any],
) -> dict[str, Any]:
    """Compute one-step-ahead predicted probabilities and evaluate."""
    I_max = int(cfg["I_max"])
    pi0 = uniform_pi_I0(I_max)

    inputs: StockpilingInputs = {
        "a_mnjt": tf.convert_to_tensor(np.asarray(panel["a_mnjt"]), dtype=tf.int32),
        "s_mjt": tf.convert_to_tensor(np.asarray(panel["p_state_mjt"]), dtype=tf.int32),
        "u_mj": tf.convert_to_tensor(np.asarray(panel["u_mj"]), dtype=tf.float64),
        "P_price_mj": tf.convert_to_tensor(np.asarray(P_price_mj), dtype=tf.float64),
        "price_vals_mj": tf.convert_to_tensor(
            np.asarray(price_vals_mj), dtype=tf.float64
        ),
        "lambda_mn": tf.convert_to_tensor(
            np.asarray(panel["lambda_mn"]), dtype=tf.float64
        ),
        "waste_cost": tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        "tol": float(cfg["dp_tol"]),
        "max_iter": int(cfg["dp_max_iter"]),
        "init_I_dist": tf.convert_to_tensor(pi0, dtype=tf.float64),
        "inventory_maps": sp_model.build_inventory_maps(I_max),
    }

    theta_hat_tf = {
        "beta": tf.convert_to_tensor(theta_hat["beta"], dtype=tf.float64),
        "alpha": tf.convert_to_tensor(theta_hat["alpha"], dtype=tf.float64),
        "v": tf.convert_to_tensor(theta_hat["v"], dtype=tf.float64),
        "fc": tf.convert_to_tensor(theta_hat["fc"], dtype=tf.float64),
        "u_scale": tf.convert_to_tensor(theta_hat["u_scale"], dtype=tf.float64),
    }
    p_buy_hat = predict_p_buy_mnjt_from_theta(theta=theta_hat_tf, inputs=inputs).numpy()

    M = int(np.asarray(panel["u_mj"]).shape[0])
    theta_oracle_tf = {
        "beta": tf.convert_to_tensor(theta_true["beta"], dtype=tf.float64),
        "alpha": tf.convert_to_tensor(theta_true["alpha"], dtype=tf.float64),
        "v": tf.convert_to_tensor(theta_true["v"], dtype=tf.float64),
        "fc": tf.convert_to_tensor(theta_true["fc"], dtype=tf.float64),
        "u_scale": tf.ones((M,), dtype=tf.float64),
    }
    p_buy_oracle = predict_p_buy_mnjt_from_theta(
        theta=theta_oracle_tf, inputs=inputs
    ).numpy()

    theta_true_eval = dict(theta_true)
    theta_true_eval["u_scale"] = np.ones((M,), dtype=np.float64)

    return evaluate_stockpiling(
        a_mnjt=np.asarray(panel["a_mnjt"]),
        p_buy_hat_mnjt=p_buy_hat,
        p_state_mjt=np.asarray(panel["p_state_mjt"]),
        theta_hat=theta_hat,
        theta_true=theta_true_eval,
        p_buy_oracle_mnjt=p_buy_oracle,
        mcmc=mcmc_diag,
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    out1 = run_phase1(CFG_PHASE1)

    res2 = run_phase2(
        CFG_PHASE2,
        dgp=out1["dgp"],
        delta_hat=out1["delta_hat"],
        eval_include_outside=bool(CFG_PHASE1["eval_include_outside"]),
    )

    phase3_inputs = build_phase3_inputs(delta_hat=out1["delta_hat"], res2=res2)
    delta_used = phase3_inputs["delta_used"]
    E_bar_used = phase3_inputs["E_bar_used"]
    njt_used = phase3_inputs["njt_used"]

    M = int(E_bar_used.shape[0])
    J = int(delta_used.shape[0])

    P_price_mj, price_vals_mj = build_price_processes(
        M=M,
        J=J,
        S=int(CFG_PHASE3["S"]),
        seed_price=int(CFG_PHASE3["price_seed"]),
        p_stay=float(CFG_PHASE3["p_stay"]),
        P_noise_sd=float(CFG_PHASE3["P_noise_sd"]),
        P_min_prob=float(CFG_PHASE3["P_min_prob"]),
        price_base_low=float(CFG_PHASE3["price_base_low"]),
        price_base_high=float(CFG_PHASE3["price_base_high"]),
        discount_low=float(CFG_PHASE3["discount_low"]),
        discount_high=float(CFG_PHASE3["discount_high"]),
        price_noise_sd=float(CFG_PHASE3["price_noise_sd"]),
    )

    print("=== Phase 3: Stockpiling DGP ===")
    panel = run_phase3_dgp(
        CFG_PHASE3,
        delta_used=delta_used,
        E_bar_used=E_bar_used,
        njt_used=njt_used,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        seed_dgp=int(CFG_PHASE1["seed"]) + 999,
    )

    summarize_stockpiling_panel(panel, init_theta=CFG_PHASE3["init_theta"])

    print("=== Phase 3: Stockpiling estimation ===")
    res3 = run_phase3_estimation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
    )

    print("=== Phase 3: Stockpiling evaluation ===")
    eval_out = run_phase3_evaluation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        theta_hat=res3["theta_hat"],
        theta_true=panel["theta_true"],
        mcmc_diag={"accept": res3["accept"], "n_saved": res3["n_saved"]},
    )

    print(format_evaluation_summary(eval_out))


if __name__ == "__main__":
    main()
