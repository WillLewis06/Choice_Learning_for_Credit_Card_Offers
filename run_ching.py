"""Run the Phase 1-3 Ching stockpiling experiment."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from datasets.ching_dgp import generate_dgp

from ching.stockpiling_estimator import (
    StockpilingConfig,
    StockpilingState,
    build_initial_state,
    run_chain,
    summarize_samples,
)
from ching.stockpiling_evaluate import evaluate_stockpiling, format_evaluation_summary
from ching.stockpiling_model import build_inventory_maps, unconstrained_to_theta
from ching.stockpiling_posterior import (
    StockpilingPosteriorConfig,
    StockpilingPosteriorTF,
)

from lu.choice_learn.cl_posterior import ChoiceLearnPosteriorConfig
from lu.choice_learn.cl_shrinkage import (
    ChoiceLearnShrinkageConfig,
    run_chain as run_lu_chain,
    summarize_samples as summarize_lu_samples,
)
from run_zhang_with_lu import (
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
)


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
    "depth": 5,
    "width": 128,
    "heads": 8,
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "shuffle_buffer": 1000,
    "eval_against_empirical": True,
}

CFG_PHASE2: dict[str, Any] = {
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
    "shrinkage": {
        "num_results": 2000,
        "num_burnin_steps": 0,
        "chunk_size": 200,
        "k_alpha": 2.0,
        "k_E_bar": 2.0,
        "k_njt": 0.5,
        "pilot_length": 20,
        "target_low": 0.30,
        "target_high": 0.50,
        "max_rounds": 50,
        "factor": 1.2,
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
    "p_stay": 0.85,
    "P_noise_sd": 0.05,
    "P_min_prob": 1e-6,
    "price_base_low": 0.7,
    "price_base_high": 1.3,
    "discount_low": 0.10,
    "discount_high": 0.35,
    "price_noise_sd": 0.02,
    "posterior": {
        "tol": 1e-5,
        "max_iter": 200,
        "eps": 1e-12,
        "sigma_z_beta": 2.0,
        "sigma_z_alpha": 2.0,
        "sigma_z_v": 2.0,
        "sigma_z_fc": 2.0,
        "sigma_z_u_scale": 2.0,
    },
    "sampler": {
        "num_results": 500,
        "chunk_size": 50,
        "k_beta": 0.10,
        "k_alpha": np.full((int(CFG_PHASE1["num_products"]),), 0.05, dtype=np.float64),
        "k_v": np.full((int(CFG_PHASE1["num_products"]),), 0.20, dtype=np.float64),
        "k_fc": np.full((int(CFG_PHASE1["num_products"]),), 0.05, dtype=np.float64),
        "k_u_scale": np.zeros((int(CFG_PHASE1["num_markets"]),), dtype=np.float64),
    },
}

EVAL_EPS = 1e-12


def _row_normalize(P: np.ndarray, min_prob: float) -> np.ndarray:
    """Clip a transition matrix below min_prob and row-normalize it."""
    P = np.asarray(P, dtype=np.float64)
    P = np.maximum(P, min_prob)
    return P / P.sum(axis=-1, keepdims=True)


def _build_price_transitions(
    rng: np.random.Generator,
    M: int,
    J: int,
    S: int,
    p_stay: float,
    noise_sd: float,
    min_prob: float,
) -> np.ndarray:
    """Construct row-stochastic (M, J, S, S) price-state transitions."""
    P = np.zeros((M, J, S, S), dtype=np.float64)
    base = np.full((S, S), (1.0 - p_stay) / max(S - 1, 1), dtype=np.float64)
    np.fill_diagonal(base, p_stay)

    for m in range(M):
        for j in range(J):
            noise = rng.normal(0.0, noise_sd, size=(S, S))
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
    """Construct per-(market, product, state) price levels."""
    base = rng.uniform(base_low, base_high, size=(M, J, 1))
    disc = rng.uniform(discount_low, discount_high, size=(M, J, S))
    noise = rng.normal(0.0, noise_sd, size=(M, J, S))
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
    """Construct price-state Markov chains and their price levels."""
    rng = np.random.default_rng(seed_price)
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


def _theta_to_state(theta: dict[str, Any], M: int) -> StockpilingState:
    """Map constrained parameter values into the unconstrained sampler state."""
    beta = np.clip(np.asarray(theta["beta"], dtype=np.float64), 1e-12, 1.0 - 1e-12)
    alpha = np.clip(np.asarray(theta["alpha"], dtype=np.float64), 1e-12, None)
    v = np.clip(np.asarray(theta["v"], dtype=np.float64), 1e-12, None)
    fc = np.clip(np.asarray(theta["fc"], dtype=np.float64), 1e-12, None)
    u_scale = np.clip(
        np.asarray(
            theta.get("u_scale", np.ones((M,), dtype=np.float64)),
            dtype=np.float64,
        ),
        1e-12,
        None,
    )

    return StockpilingState(
        z_beta=tf.constant(np.log(beta) - np.log1p(-beta), dtype=tf.float64),
        z_alpha=tf.constant(np.log(alpha), dtype=tf.float64),
        z_v=tf.constant(np.log(v), dtype=tf.float64),
        z_fc=tf.constant(np.log(fc), dtype=tf.float64),
        z_u_scale=tf.constant(np.log(u_scale), dtype=tf.float64),
    )


def _build_phase2_posterior_config(cfg: dict[str, Any]) -> ChoiceLearnPosteriorConfig:
    """Construct the Phase-2 posterior config."""
    pcfg = cfg["posterior"]
    return ChoiceLearnPosteriorConfig(
        alpha_mean=float(pcfg["alpha_mean"]),
        alpha_var=float(pcfg["alpha_var"]),
        E_bar_mean=float(pcfg["E_bar_mean"]),
        E_bar_var=float(pcfg["E_bar_var"]),
        T0_sq=float(pcfg["T0_sq"]),
        T1_sq=float(pcfg["T1_sq"]),
        a_phi=float(pcfg["a_phi"]),
        b_phi=float(pcfg["b_phi"]),
    )


def _build_phase2_shrinkage_config(cfg: dict[str, Any]) -> ChoiceLearnShrinkageConfig:
    """Construct the Phase-2 shrinkage config."""
    scfg = cfg["shrinkage"]
    return ChoiceLearnShrinkageConfig(
        num_results=int(scfg["num_results"]),
        num_burnin_steps=int(scfg["num_burnin_steps"]),
        chunk_size=int(scfg["chunk_size"]),
        k_alpha=float(scfg["k_alpha"]),
        k_E_bar=float(scfg["k_E_bar"]),
        k_njt=float(scfg["k_njt"]),
        pilot_length=int(scfg["pilot_length"]),
        target_low=float(scfg["target_low"]),
        target_high=float(scfg["target_high"]),
        max_rounds=int(scfg["max_rounds"]),
        factor=float(scfg["factor"]),
    )


def _build_phase3_posterior_config(cfg: dict[str, Any]) -> StockpilingPosteriorConfig:
    """Construct the Phase-3 posterior config."""
    pcfg = cfg["posterior"]
    return StockpilingPosteriorConfig(
        tol=float(pcfg["tol"]),
        max_iter=int(pcfg["max_iter"]),
        eps=float(pcfg["eps"]),
        sigma_z_beta=float(pcfg["sigma_z_beta"]),
        sigma_z_alpha=float(pcfg["sigma_z_alpha"]),
        sigma_z_v=float(pcfg["sigma_z_v"]),
        sigma_z_fc=float(pcfg["sigma_z_fc"]),
        sigma_z_u_scale=float(pcfg["sigma_z_u_scale"]),
    )


def _build_phase3_sampler_config(cfg: dict[str, Any]) -> StockpilingConfig:
    """Construct the Phase-3 sampler config."""
    scfg = cfg["sampler"]
    return StockpilingConfig(
        num_results=int(scfg["num_results"]),
        chunk_size=int(scfg["chunk_size"]),
        k_beta=tf.constant(float(scfg["k_beta"]), dtype=tf.float64),
        k_alpha=tf.constant(
            np.asarray(scfg["k_alpha"], dtype=np.float64), dtype=tf.float64
        ),
        k_v=tf.constant(np.asarray(scfg["k_v"], dtype=np.float64), dtype=tf.float64),
        k_fc=tf.constant(np.asarray(scfg["k_fc"], dtype=np.float64), dtype=tf.float64),
        k_u_scale=tf.constant(
            np.asarray(scfg["k_u_scale"], dtype=np.float64),
            dtype=tf.float64,
        ),
    )


def summarize_stockpiling_panel(panel: dict[str, Any]) -> None:
    """Print a compact summary of the generated Phase-3 panel."""
    a = np.asarray(panel["a_mnjt"], dtype=np.float64)
    s = np.asarray(panel["s_mjt"])
    u = np.asarray(panel["u_mj"], dtype=np.float64)

    print("")
    print("=== Stockpiling data generated ===")
    print(f"shapes: a_mnjt={a.shape} | p_state_mjt={s.shape} | u_mj={u.shape}")
    print(f"overall buy rate: {a.mean():.4f}")

    theta_true = panel.get("theta_true")
    if isinstance(theta_true, dict):
        print(
            "[Stockpiling] True | "
            f"beta={float(np.asarray(theta_true['beta'], dtype=np.float64)):.4f} "
            f"mean_alpha={float(np.mean(np.asarray(theta_true['alpha'], dtype=np.float64))):.4f} "
            f"mean_v={float(np.mean(np.asarray(theta_true['v'], dtype=np.float64))):.4f} "
            f"mean_fc={float(np.mean(np.asarray(theta_true['fc'], dtype=np.float64))):.4f}"
        )

    M, _, J, _ = a.shape
    initial_state = build_initial_state(M=M, J=J)
    theta0 = unconstrained_to_theta(
        {
            "z_beta": initial_state.z_beta,
            "z_alpha": initial_state.z_alpha,
            "z_v": initial_state.z_v,
            "z_fc": initial_state.z_fc,
            "z_u_scale": initial_state.z_u_scale,
        }
    )
    print(
        "[Stockpiling] Init | "
        f"beta={float(theta0['beta'].numpy()):.4f} "
        f"mean_alpha={float(tf.reduce_mean(theta0['alpha']).numpy()):.4f} "
        f"mean_v={float(tf.reduce_mean(theta0['v']).numpy()):.4f} "
        f"mean_fc={float(tf.reduce_mean(theta0['fc']).numpy()):.4f} "
        f"mean_u_scale={float(tf.reduce_mean(theta0['u_scale']).numpy()):.4f}"
    )


def run_phase1(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 1 and print inside-choice diagnostics."""
    print("=== Phase 1: Baseline choice model ===")
    out = run_choice_model(
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

    dgp = out["dgp"]
    print_choice_model_diagnostics(
        delta_hat=out["delta_hat"],
        delta_true=dgp["delta_true"],
        qj_base=dgp["qj_base"],
        p_base=dgp["p_base"],
        eval_against_empirical=bool(cfg["eval_against_empirical"]),
    )
    return out


def run_phase2(
    cfg: dict[str, Any],
    dgp: dict[str, Any],
    delta_hat: np.ndarray,
) -> dict[str, Any]:
    """Run Phase 2 and print diagnostics."""
    print("=== Phase 2: Market-shock estimator ===")

    qjt = tf.constant(np.asarray(dgp["qjt_shock"], dtype=np.float64), dtype=tf.float64)
    q0t = tf.constant(np.asarray(dgp["q0t_shock"], dtype=np.float64), dtype=tf.float64)
    delta_cl = tf.constant(
        np.repeat(
            np.asarray(delta_hat, dtype=np.float64)[None, :], qjt.shape[0], axis=0
        ),
        dtype=tf.float64,
    )

    samples = run_lu_chain(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
        posterior_config=_build_phase2_posterior_config(cfg),
        shrinkage_config=_build_phase2_shrinkage_config(cfg),
        seed=tf.constant([int(cfg["seed"]), 0], dtype=tf.int32),
    )
    summary_tf = summarize_lu_samples(samples)
    res = {
        key: (value.numpy() if isinstance(value, tf.Tensor) else value)
        for key, value in summary_tf.items()
    }

    print_market_shock_diagnostics(
        np.asarray(delta_hat, dtype=np.float64),
        dgp,
        res,
    )
    return res


def build_phase3_inputs(
    delta_hat: np.ndarray, res2: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Build the fixed Phase-3 intercept components from Phases 1-2."""
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
    """Generate the Phase-3 seller-observed panel."""
    a_mnjt, p_state_mjt, u_mj, theta_true = generate_dgp(
        seed=seed_dgp,
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
    return {
        "a_mnjt": a_mnjt,
        "s_mjt": p_state_mjt,
        "u_mj": u_mj,
        "lambda_mn": np.asarray(theta_true["lambda"], dtype=np.float64),
        "theta_true": theta_true,
    }


def run_phase3_estimation(
    cfg: dict[str, Any],
    panel: dict[str, Any],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    mcmc_seed: int,
) -> dict[str, Any]:
    """Run the Phase-3 chain and summarize retained samples."""
    a = np.asarray(panel["a_mnjt"])
    M = int(a.shape[0])
    J = int(a.shape[2])

    initial_state = build_initial_state(M=M, J=J)

    run_result = run_chain(
        a_mnjt=tf.constant(panel["a_mnjt"], dtype=tf.int32),
        s_mjt=tf.constant(panel["s_mjt"], dtype=tf.int32),
        u_mj=tf.constant(panel["u_mj"], dtype=tf.float64),
        P_price_mj=tf.constant(P_price_mj, dtype=tf.float64),
        price_vals_mj=tf.constant(price_vals_mj, dtype=tf.float64),
        lambda_mn=tf.constant(panel["lambda_mn"], dtype=tf.float64),
        waste_cost=tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        inventory_maps=build_inventory_maps(int(cfg["I_max"])),
        posterior_config=_build_phase3_posterior_config(cfg),
        stockpiling_config=_build_phase3_sampler_config(cfg),
        initial_state=initial_state,
        seed=tf.constant([int(mcmc_seed), 0], dtype=tf.int32),
    )

    theta_hat_tf = summarize_samples(run_result.samples)
    theta_hat = {
        "beta": np.asarray(theta_hat_tf["beta"].numpy(), dtype=np.float64),
        "alpha": np.asarray(theta_hat_tf["alpha"].numpy(), dtype=np.float64),
        "v": np.asarray(theta_hat_tf["v"].numpy(), dtype=np.float64),
        "fc": np.asarray(theta_hat_tf["fc"].numpy(), dtype=np.float64),
        "u_scale": np.asarray(theta_hat_tf["u_scale"].numpy(), dtype=np.float64),
    }

    return {
        "samples": run_result.samples,
        "theta_hat": theta_hat,
        "initial_state": initial_state,
        "posterior_config": _build_phase3_posterior_config(cfg),
        "chunk_summaries": run_result.chunk_summaries,
        "mcmc_summary": run_result.mcmc_summary,
        "n_saved": int(run_result.samples.z_beta.shape[0]),
    }


def run_phase3_evaluation(
    cfg: dict[str, Any],
    panel: dict[str, Any],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    theta_hat: dict[str, Any],
    theta_true: dict[str, Any] | None,
    posterior_config: StockpilingPosteriorConfig,
    mcmc_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the fitted Phase-3 model."""
    a = np.asarray(panel["a_mnjt"], dtype=np.float64)
    s = np.asarray(panel["s_mjt"], dtype=np.int64)
    u = np.asarray(panel["u_mj"], dtype=np.float64)
    lam = np.asarray(panel["lambda_mn"], dtype=np.float64)

    M = int(a.shape[0])
    posterior = StockpilingPosteriorTF(
        config=posterior_config,
        a_mnjt=tf.constant(panel["a_mnjt"], dtype=tf.int32),
        s_mjt=tf.constant(panel["s_mjt"], dtype=tf.int32),
        u_mj=tf.constant(u, dtype=tf.float64),
        P_price_mj=tf.constant(P_price_mj, dtype=tf.float64),
        price_vals_mj=tf.constant(price_vals_mj, dtype=tf.float64),
        lambda_mn=tf.constant(lam, dtype=tf.float64),
        waste_cost=tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        inventory_maps=build_inventory_maps(int(cfg["I_max"])),
    )

    fitted_state = _theta_to_state(theta_hat, M=M)
    p_buy_hat_mnjt = posterior.predict_p_buy_mnjt(
        z_beta=fitted_state.z_beta,
        z_alpha=fitted_state.z_alpha,
        z_v=fitted_state.z_v,
        z_fc=fitted_state.z_fc,
        z_u_scale=fitted_state.z_u_scale,
    ).numpy()

    p_buy_oracle_mnjt: np.ndarray | None = None
    theta_true_eval: dict[str, np.ndarray] | None = None
    if theta_true is not None:
        theta_true_eval = {
            "beta": np.asarray(theta_true["beta"], dtype=np.float64),
            "alpha": np.asarray(theta_true["alpha"], dtype=np.float64),
            "v": np.asarray(theta_true["v"], dtype=np.float64),
            "fc": np.asarray(theta_true["fc"], dtype=np.float64),
        }
        oracle_state = _theta_to_state(theta_true_eval, M=M)
        p_buy_oracle_mnjt = posterior.predict_p_buy_mnjt(
            z_beta=oracle_state.z_beta,
            z_alpha=oracle_state.z_alpha,
            z_v=oracle_state.z_v,
            z_fc=oracle_state.z_fc,
            z_u_scale=oracle_state.z_u_scale,
        ).numpy()

    eval_out = evaluate_stockpiling(
        a_mnjt=a,
        p_buy_hat_mnjt=np.asarray(p_buy_hat_mnjt, dtype=np.float64),
        s_mjt=s,
        theta_hat=theta_hat if theta_true_eval is not None else None,
        theta_true=theta_true_eval,
        p_buy_oracle_mnjt=p_buy_oracle_mnjt,
        mcmc=mcmc_summary,
        eps=EVAL_EPS,
    )

    print("=== Phase 3: Stockpiling evaluation ===")
    print(format_evaluation_summary(eval_out))

    return {
        "eval": eval_out,
        "p_buy_hat_mnjt": np.asarray(p_buy_hat_mnjt, dtype=np.float64),
        "p_buy_oracle_mnjt": p_buy_oracle_mnjt,
        "mcmc": mcmc_summary,
    }


def main() -> None:
    """Run the full Phase 1-3 stockpiling experiment."""
    out1 = run_phase1(CFG_PHASE1)
    dgp = out1["dgp"]
    delta_hat = np.asarray(out1["delta_hat"], dtype=np.float64)

    res2 = run_phase2(CFG_PHASE2, dgp=dgp, delta_hat=delta_hat)
    phase3_inputs = build_phase3_inputs(delta_hat=delta_hat, res2=res2)

    delta_used = phase3_inputs["delta_used"]
    E_bar_used = phase3_inputs["E_bar_used"]
    njt_used = phase3_inputs["njt_used"]

    M = int(E_bar_used.shape[0])
    J = int(delta_used.shape[0])

    phase3_price_seed = 777
    phase3_dgp_seed = phase3_price_seed + 1
    phase3_mcmc_seed = 0

    P_price_mj, price_vals_mj = build_price_processes(
        M=M,
        J=J,
        S=int(CFG_PHASE3["S"]),
        seed_price=phase3_price_seed,
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
        seed_dgp=phase3_dgp_seed,
    )
    summarize_stockpiling_panel(panel)

    res3 = run_phase3_estimation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        mcmc_seed=phase3_mcmc_seed,
    )

    run_phase3_evaluation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        theta_hat=res3["theta_hat"],
        theta_true=panel["theta_true"],
        posterior_config=res3["posterior_config"],
        mcmc_summary=res3["mcmc_summary"],
    )


if __name__ == "__main__":
    main()
