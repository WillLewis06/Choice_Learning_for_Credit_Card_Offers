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
  - s_mjt       (M,J,T)   exogenous price states

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

from ching.stockpiling_input_validation import validate_stockpiling_phase3_config

# Phase 1–2 orchestration
from run_zhang_with_lu import (
    build_shrinkage_fit_config,
    build_shrinkage_init_config,
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
    run_market_shock_estimator,
)


def uniform_pi_I0(I_max: int) -> np.ndarray:
    """Uniform initial inventory distribution over {0,...,I_max}."""
    I_max_i = int(I_max)
    pi = np.ones(I_max_i + 1, dtype=np.float64)
    return pi / float(pi.sum())


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
    "init_config": {
        "seed": 0,
        "posterior": {
            "alpha_mean": 1.0,
            "alpha_var": 1.0,
            "E_bar_mean": 0.0,
            "E_bar_var": 1.0,
            "T0_sq": 1.0,
            "T1_sq": 2.0,
            "a_phi": 1.0,
            "b_phi": 1.0,
        },
        "init_state": {
            # Scalars; expanded to required shapes inside run_phase2().
            "alpha": 1.0,
            "E_bar": 0.0,
            "njt": 0.0,
            "gamma": 0.0,  # MUST be exactly 0.0 or 1.0
            "phi": 0.5,  # MUST be strictly in (0,1)
        },
    },
    "fit_config": {
        "n_iter": 200,
        "pilot_length": 20,
        "ridge": 1e-6,
        "target_low": 0.30,
        "target_high": 0.50,
        "max_rounds": 50,
        "factor_rw": 1.2,
        "factor_tmh": 1.2,
        "k_alpha0": 2.0,
        "k_E_bar0": 2.0,
        "k_njt0": 0.5,
        "tune_seed": 0,
    },
}

CFG_PHASE3: dict[str, Any] = {
    "N": 200,
    "T": 500,
    "I_max": 10,
    "S": 2,
    "waste_cost": 1.0,
    "dp_tol": 1e-5,
    "dp_max_iter": 200,
    "pi_I0": uniform_pi_I0(10),
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
    "mcmc_n_iter": 500,
    "init_theta": {
        "beta": 0.5,
        "alpha": np.full((int(CFG_PHASE1["num_products"]),), 0.5, dtype=np.float64),
        "v": np.full((int(CFG_PHASE1["num_products"]),), 0.5, dtype=np.float64),
        "fc": np.full((int(CFG_PHASE1["num_products"]),), 0.5, dtype=np.float64),
        "u_scale": np.ones((int(CFG_PHASE1["num_markets"]),), dtype=np.float64),
    },
    # StockpilingEstimator expects z-space prior scales keyed by z_*.
    "sigmas": {
        "z_beta": 2.0,
        "z_alpha": 2.0,
        "z_v": 2.0,
        "z_fc": 2.0,
        "z_u_scale": 2.0,
    },
    # Proposal scales (z-space random walk). Set u_scale > 0 to estimate it.
    "k": {
        "beta": 0.1,
        "alpha": 0.05,
        "v": 0.2,
        "fc": 0.05,
        "u_scale": 0.00,
    },
}

EVAL_EPS = 1e-12

# =============================================================================
# Price process helpers
# =============================================================================


def _row_normalize(P: np.ndarray, min_prob: float) -> np.ndarray:
    """Row-normalize a 2D array after clipping entries below min_prob."""
    P64 = np.asarray(P, dtype=np.float64)
    P64 = np.maximum(P64, float(min_prob))
    return P64 / P64.sum(axis=-1, keepdims=True)


def _build_price_transitions(
    rng: np.random.Generator,
    M: int,
    J: int,
    S: int,
    p_stay: float,
    noise_sd: float,
    min_prob: float,
) -> np.ndarray:
    """Construct (M,J,S,S) row-stochastic Markov transitions with mild randomization."""
    M_i, J_i, S_i = int(M), int(J), int(S)
    P = np.zeros((M_i, J_i, S_i, S_i), dtype=np.float64)

    p_stay_f = float(p_stay)
    base = np.full((S_i, S_i), (1.0 - p_stay_f) / max(S_i - 1, 1), dtype=np.float64)
    np.fill_diagonal(base, p_stay_f)

    for m in range(M_i):
        for j in range(J_i):
            noise = rng.normal(0.0, float(noise_sd), size=(S_i, S_i))
            P[m, j] = _row_normalize(base + noise, min_prob=min_prob)

    return P


def _build_price_levels(
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
    M_i, J_i, S_i = int(M), int(J), int(S)
    base = rng.uniform(float(base_low), float(base_high), size=(M_i, J_i, 1))
    disc = rng.uniform(float(discount_low), float(discount_high), size=(M_i, J_i, S_i))
    noise = rng.normal(0.0, float(noise_sd), size=(M_i, J_i, S_i))
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
    """Build P_price_mj and price_vals_mj from config."""
    rng = np.random.default_rng(int(seed_price))
    P_price_mj = _build_price_transitions(
        rng=rng,
        M=M,
        J=J,
        S=S,
        p_stay=p_stay,
        noise_sd=P_noise_sd,
        min_prob=P_min_prob,
    )
    price_vals_mj = _build_price_levels(
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


# =============================================================================
# Diagnostics / summaries
# =============================================================================


def summarize_stockpiling_panel(
    panel: dict[str, Any], init_theta: dict[str, Any]
) -> None:
    """Lightweight printout of Phase-3 panel + true/init params."""
    a = np.asarray(panel["a_mnjt"])
    s = np.asarray(panel["s_mjt"])
    u = np.asarray(panel["u_mj"])

    print("")
    print("=== Stockpiling data generated ===")
    print(f"shapes: a_mnjt={a.shape} | p_state_mjt={s.shape} | u_mj={u.shape}")
    print(f"overall buy rate: {float(a.mean()):.4f}")

    theta_true = panel.get("theta_true", None)
    if isinstance(theta_true, dict):
        print(
            "[Stockpiling] True | "
            f'mean(beta)= "{float(np.mean(theta_true["beta"])):.4f}" , '
            f'mean(alpha)= "{float(np.mean(theta_true["alpha"])):.4f}" , '
            f'mean(v)= "{float(np.mean(theta_true["v"])):.4f}" , '
            f'mean(fc)= "{float(np.mean(theta_true["fc"])):.4f}"'
        )

    beta0 = float(np.mean(np.asarray(init_theta["beta"], dtype=np.float64)))
    alpha0 = float(np.mean(np.asarray(init_theta["alpha"], dtype=np.float64)))
    v0 = float(np.mean(np.asarray(init_theta["v"], dtype=np.float64)))
    fc0 = float(np.mean(np.asarray(init_theta["fc"], dtype=np.float64)))
    u_scale0 = float(np.mean(np.asarray(init_theta["u_scale"], dtype=np.float64)))

    print(
        "[Stockpiling] Init | "
        f'mean(beta)= "{beta0:.4f}" , '
        f'mean(alpha)= "{alpha0:.4f}" , '
        f'mean(v)= "{v0:.4f}" , '
        f'mean(fc)= "{fc0:.4f}" , '
        f'mean(u_scale)= "{u_scale0:.4f}"'
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
        g_sd=float(cfg["g_sd"]),
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

    # Data tensors
    qjt_np = np.asarray(dgp["qjt_shock"], dtype=np.float64)  # (T,J)
    q0t_np = np.asarray(dgp["q0t_shock"], dtype=np.float64)  # (T,)
    delta_hat_np = np.asarray(delta_hat, dtype=np.float64)  # (J,)

    T_i, J_i = qjt_np.shape
    delta_cl_np = np.repeat(delta_hat_np[None, :], T_i, axis=0)  # (T,J)

    delta_cl = tf.convert_to_tensor(delta_cl_np, dtype=tf.float64)
    qjt = tf.convert_to_tensor(qjt_np, dtype=tf.float64)
    q0t = tf.convert_to_tensor(q0t_np, dtype=tf.float64)

    # Config blocks (assemble exact types/shapes expected by cl_validate_input.py)
    init_cfg = cfg["init_config"]
    fit_cfg = cfg["fit_config"]

    posterior_cfg = init_cfg["posterior"]
    init_state_cfg = init_cfg["init_state"]

    alpha0 = tf.convert_to_tensor(
        float(init_state_cfg["alpha"]), dtype=tf.float64
    )  # ()
    E_bar0 = tf.fill([T_i], tf.cast(float(init_state_cfg["E_bar"]), tf.float64))  # (T,)
    njt0 = tf.fill(
        [T_i, J_i], tf.cast(float(init_state_cfg["njt"]), tf.float64)
    )  # (T,J)

    gamma_init = float(init_state_cfg["gamma"])
    phi_init = float(init_state_cfg["phi"])

    gamma0 = tf.fill([T_i, J_i], tf.cast(gamma_init, tf.float64))  # (T,J), binary {0,1}
    phi0 = tf.fill([T_i], tf.cast(phi_init, tf.float64))  # (T,), in (0,1)

    init_config = build_shrinkage_init_config(
        seed=int(init_cfg["seed"]),
        posterior={
            "alpha_mean": float(posterior_cfg["alpha_mean"]),
            "alpha_var": float(posterior_cfg["alpha_var"]),
            "E_bar_mean": float(posterior_cfg["E_bar_mean"]),
            "E_bar_var": float(posterior_cfg["E_bar_var"]),
            "T0_sq": float(posterior_cfg["T0_sq"]),
            "T1_sq": float(posterior_cfg["T1_sq"]),
            "a_phi": float(posterior_cfg["a_phi"]),
            "b_phi": float(posterior_cfg["b_phi"]),
        },
        init_state={
            "alpha": alpha0,
            "E_bar": E_bar0,
            "njt": njt0,
            "gamma": gamma0,
            "phi": phi0,
        },
    )

    fit_config = build_shrinkage_fit_config(
        n_iter=int(fit_cfg["n_iter"]),
        pilot_length=int(fit_cfg["pilot_length"]),
        ridge=float(fit_cfg["ridge"]),
        target_low=float(fit_cfg["target_low"]),
        target_high=float(fit_cfg["target_high"]),
        max_rounds=int(fit_cfg["max_rounds"]),
        factor_rw=float(fit_cfg["factor_rw"]),
        factor_tmh=float(fit_cfg["factor_tmh"]),
        k_alpha0=float(fit_cfg["k_alpha0"]),
        k_E_bar0=float(fit_cfg["k_E_bar0"]),
        k_njt0=float(fit_cfg["k_njt0"]),
        tune_seed=int(fit_cfg["tune_seed"]),
    )

    res2 = run_market_shock_estimator(delta_cl, qjt, q0t, init_config, fit_config)

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
        delta_true=delta_used,
        E_bar_true=E_bar_used,
        njt_true=njt_used,
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        I_max=int(cfg["I_max"]),
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        waste_cost=float(cfg["waste_cost"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
    )
    theta_true = dict(theta_true)
    lambda_mn = np.asarray(theta_true["lambda"], dtype=np.float64)

    return {
        "a_mnjt": a_mnjt,
        "s_mjt": p_state_mjt,
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
        a_mnjt=panel["a_mnjt"],
        s_mjt=panel["s_mjt"],
        u_mj=panel["u_mj"],
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        lambda_mn=panel["lambda_mn"],
        I_max=int(cfg["I_max"]),
        pi_I0=np.asarray(cfg["pi_I0"], dtype=np.float64),
        waste_cost=float(cfg["waste_cost"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
        sigmas=dict(cfg["sigmas"]),
        rng_seed=int(cfg["mcmc_seed"]),
    )

    res = est.fit(
        n_iter=int(cfg["mcmc_n_iter"]),
        k=dict(cfg["k"]),
        init_theta=dict(cfg["init_theta"]),
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
    theta_hat: dict[str, Any],
    theta_true: dict[str, Any] | None,
    mcmc_diag: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evaluation wrapper aligned with ching.stockpiling_evaluate.evaluate_stockpiling().

    - Computes fitted p_buy_hat_mnjt from theta_hat
    - Optionally computes oracle p_buy_oracle_mnjt from theta_true (if provided)
    - Calls evaluate_stockpiling(...) with the correct argument names/signature
    """
    a_np = np.asarray(panel["a_mnjt"])
    s_np = np.asarray(panel["s_mjt"])
    u_np = np.asarray(panel["u_mj"])
    lam_np = np.asarray(panel["lambda_mn"])

    M = int(a_np.shape[0])

    # Build posterior inputs (must match StockpilingInputs keys)
    inv_maps = sp_model.build_inventory_maps(int(cfg["I_max"]))
    inputs: StockpilingInputs = {
        "a_mnjt": tf.convert_to_tensor(a_np, dtype=tf.int32),
        "s_mjt": tf.convert_to_tensor(s_np, dtype=tf.int32),
        "u_mj": tf.convert_to_tensor(u_np, dtype=tf.float64),
        "P_price_mj": tf.convert_to_tensor(np.asarray(P_price_mj), dtype=tf.float64),
        "price_vals_mj": tf.convert_to_tensor(
            np.asarray(price_vals_mj), dtype=tf.float64
        ),
        "lambda_mn": tf.convert_to_tensor(lam_np, dtype=tf.float64),
        "waste_cost": tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        "inventory_maps": inv_maps,
        "tol": float(cfg["dp_tol"]),
        "max_iter": int(cfg["dp_max_iter"]),
        "pi_I0": tf.convert_to_tensor(
            np.asarray(cfg["pi_I0"], dtype=np.float64), dtype=tf.float64
        ),
    }

    # Fitted probabilities
    theta_hat_tf = {
        "beta": tf.convert_to_tensor(theta_hat["beta"], dtype=tf.float64),
        "alpha": tf.convert_to_tensor(theta_hat["alpha"], dtype=tf.float64),
        "v": tf.convert_to_tensor(theta_hat["v"], dtype=tf.float64),
        "fc": tf.convert_to_tensor(theta_hat["fc"], dtype=tf.float64),
        "u_scale": tf.convert_to_tensor(theta_hat["u_scale"], dtype=tf.float64),
    }
    p_buy_hat_tf = predict_p_buy_mnjt_from_theta(theta=theta_hat_tf, inputs=inputs)
    p_buy_hat_mnjt = np.asarray(p_buy_hat_tf.numpy(), dtype=np.float64)

    # Prepare theta_hat for parameter_metrics (numpy arrays)
    theta_hat_np: dict[str, np.ndarray] = {
        "beta": np.asarray(theta_hat["beta"], dtype=np.float64),
        "alpha": np.asarray(theta_hat["alpha"], dtype=np.float64),
        "v": np.asarray(theta_hat["v"], dtype=np.float64),
        "fc": np.asarray(theta_hat["fc"], dtype=np.float64),
        "u_scale": np.asarray(theta_hat["u_scale"], dtype=np.float64),
    }

    # Optional oracle probabilities + parameter recovery
    p_buy_oracle_mnjt: np.ndarray | None = None
    theta_true_np: dict[str, np.ndarray] | None = None

    if theta_true is not None:
        # Ensure theta_true has all PARAM_KEYS expected by evaluation (incl. u_scale).
        # DGP may not contain u_scale; treat it as ones(M,) for evaluation compatibility.
        u_scale_true = (
            np.asarray(theta_true["u_scale"], dtype=np.float64)
            if "u_scale" in theta_true
            else np.ones((M,), dtype=np.float64)
        )

        theta_true_np = {
            "beta": np.asarray(theta_true["beta"], dtype=np.float64),
            "alpha": np.asarray(theta_true["alpha"], dtype=np.float64),
            "v": np.asarray(theta_true["v"], dtype=np.float64),
            "fc": np.asarray(theta_true["fc"], dtype=np.float64),
            "u_scale": u_scale_true,
        }

        theta_true_tf = {
            "beta": tf.convert_to_tensor(theta_true_np["beta"], dtype=tf.float64),
            "alpha": tf.convert_to_tensor(theta_true_np["alpha"], dtype=tf.float64),
            "v": tf.convert_to_tensor(theta_true_np["v"], dtype=tf.float64),
            "fc": tf.convert_to_tensor(theta_true_np["fc"], dtype=tf.float64),
            "u_scale": tf.convert_to_tensor(theta_true_np["u_scale"], dtype=tf.float64),
        }
        p_buy_oracle_tf = predict_p_buy_mnjt_from_theta(
            theta=theta_true_tf, inputs=inputs
        )
        p_buy_oracle_mnjt = np.asarray(p_buy_oracle_tf.numpy(), dtype=np.float64)

    # Optional MCMC diagnostics (must include accept and n_saved if provided)
    mcmc = None
    if mcmc_diag is not None:
        mcmc = {
            "accept": mcmc_diag["accept"],
            "n_saved": mcmc_diag["n_saved"],
        }

    eval_out = evaluate_stockpiling(
        a_mnjt=a_np,
        p_buy_hat_mnjt=p_buy_hat_mnjt,
        s_mjt=s_np,
        theta_hat=theta_hat_np if theta_true_np is not None else None,
        theta_true=theta_true_np,
        p_buy_oracle_mnjt=p_buy_oracle_mnjt,
        mcmc=mcmc,
        eps=float(EVAL_EPS),
    )

    print("=== Phase 3: Stockpiling evaluation ===")
    print(format_evaluation_summary(eval_out))

    return {
        "eval": eval_out,
        "p_buy_hat_mnjt": p_buy_hat_mnjt,
        "p_buy_oracle_mnjt": p_buy_oracle_mnjt,
        "mcmc": mcmc,
    }


def main() -> None:
    """Run Phases 1–3 end-to-end."""
    out1 = run_phase1(CFG_PHASE1)
    dgp = out1["dgp"]
    delta_hat = np.asarray(out1["delta_hat"], dtype=np.float64)

    res2 = run_phase2(
        CFG_PHASE2,
        dgp=dgp,
        delta_hat=delta_hat,
        eval_include_outside=bool(CFG_PHASE1["eval_include_outside"]),
    )

    phase3_inputs = build_phase3_inputs(delta_hat=delta_hat, res2=res2)
    delta_used = phase3_inputs["delta_used"]
    E_bar_used = phase3_inputs["E_bar_used"]
    njt_used = phase3_inputs["njt_used"]

    M = int(E_bar_used.shape[0])
    J = int(delta_used.shape[0])

    validate_stockpiling_phase3_config(cfg=CFG_PHASE3, M=M, J=J)

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

    print("")
    print("============================================================")
    print("PHASE 3: Ching - Stockpiling")
    print("============================================================")

    panel = run_phase3_dgp(
        CFG_PHASE3,
        delta_used=delta_used,
        E_bar_used=E_bar_used,
        njt_used=njt_used,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        seed_dgp=int(CFG_PHASE3["price_seed"]) + 1,
    )

    summarize_stockpiling_panel(panel=panel, init_theta=dict(CFG_PHASE3["init_theta"]))

    res3 = run_phase3_estimation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
    )

    theta_hat = dict(res3["theta_hat"])
    theta_true = dict(panel["theta_true"])

    _ = run_phase3_evaluation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        theta_hat=theta_hat,
        theta_true=theta_true,
        mcmc_diag={
            "accept": res3["accept"],
            "n_saved": res3["n_saved"],
            "z_last": res3["z_last"],
        },
    )


if __name__ == "__main__":
    main()
