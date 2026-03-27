"""Run the Phase 1–3 Ching stockpiling experiment with the refactored Phase-3 chain."""

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

from run_zhang_with_lu import (
    build_shrinkage_fit_config,
    build_shrinkage_init_config,
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
    run_market_shock_estimator,
)


def uniform_pi_I0(I_max: int) -> np.ndarray:
    """Return the uniform initial inventory distribution over {0, ..., I_max}."""
    pi = np.ones(int(I_max) + 1, dtype=np.float64)
    return pi / float(pi.sum())


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
            "alpha": 1.0,
            "E_bar": 0.0,
            "njt": 0.0,
            "gamma": 0.0,
            "phi": 0.5,
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
    "price_seed": 777,
    "p_stay": 0.85,
    "P_noise_sd": 0.05,
    "P_min_prob": 1e-6,
    "price_base_low": 0.7,
    "price_base_high": 1.3,
    "discount_low": 0.10,
    "discount_high": 0.35,
    "price_noise_sd": 0.02,
    "mcmc_seed": 0,
    "posterior": {
        "tol": 1e-5,
        "max_iter": 200,
        "eps": 1e-12,
        "sigma_z_beta": 2.0,
        "sigma_z_alpha": 2.0,
        "sigma_z_v": 2.0,
        "sigma_z_fc": 2.0,
        "sigma_z_u_scale": 2.0,
        "fix_u_scale": True,
        "fixed_z_u_scale": 0.0,
    },
    "sampler": {
        "num_results": 500,
        "num_burnin_steps": 0,
        "chunk_size": 100,
        "k_beta": 0.10,
        "k_alpha": np.full((int(CFG_PHASE1["num_products"]),), 0.05, dtype=np.float64),
        "k_v": np.full((int(CFG_PHASE1["num_products"]),), 0.20, dtype=np.float64),
        "k_fc": np.full((int(CFG_PHASE1["num_products"]),), 0.05, dtype=np.float64),
        "k_u_scale": np.full((int(CFG_PHASE1["num_markets"]),), 0.05, dtype=np.float64),
        "pilot_num_steps": 100,
        "target_accept_low": 0.20,
        "target_accept_high": 0.40,
        "grow_factor": 1.5,
        "shrink_factor": 0.75,
        "max_tuning_rounds": 8,
    },
}

EVAL_EPS = 1e-12


def _row_normalize(P: np.ndarray, min_prob: float) -> np.ndarray:
    """Clip a transition matrix below min_prob and row-normalize it."""
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
    """Construct row-stochastic (M, J, S, S) price-state transitions."""
    P = np.zeros((int(M), int(J), int(S), int(S)), dtype=np.float64)
    base = np.full(
        (int(S), int(S)),
        (1.0 - float(p_stay)) / max(int(S) - 1, 1),
        dtype=np.float64,
    )
    np.fill_diagonal(base, float(p_stay))

    for m in range(int(M)):
        for j in range(int(J)):
            noise = rng.normal(0.0, float(noise_sd), size=(int(S), int(S)))
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
    base = rng.uniform(float(base_low), float(base_high), size=(int(M), int(J), 1))
    disc = rng.uniform(
        float(discount_low),
        float(discount_high),
        size=(int(M), int(J), int(S)),
    )
    noise = rng.normal(0.0, float(noise_sd), size=(int(M), int(J), int(S)))
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
    """Construct the price-state Markov chains and their price levels."""
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


def _to_numpy_results(results: dict[str, Any]) -> dict[str, Any]:
    """Convert tensor-valued result entries into NumPy-backed outputs."""
    out: dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, tf.Tensor):
            out[key] = value.numpy()
        else:
            out[key] = value
    return out


def _theta_to_state(theta: dict[str, Any], M: int) -> StockpilingState:
    """Map constrained theta values into the unconstrained sampler state."""
    beta = np.asarray(theta["beta"], dtype=np.float64)
    beta = np.clip(beta, 1e-12, 1.0 - 1e-12)

    alpha = np.asarray(theta["alpha"], dtype=np.float64)
    v = np.asarray(theta["v"], dtype=np.float64)
    fc = np.asarray(theta["fc"], dtype=np.float64)
    u_scale = np.asarray(
        theta.get("u_scale", np.ones((M,), dtype=np.float64)),
        dtype=np.float64,
    )

    alpha = np.clip(alpha, 1e-12, None)
    v = np.clip(v, 1e-12, None)
    fc = np.clip(fc, 1e-12, None)
    u_scale = np.clip(u_scale, 1e-12, None)

    return StockpilingState(
        z_beta=tf.convert_to_tensor(np.log(beta) - np.log1p(-beta), dtype=tf.float64),
        z_alpha=tf.convert_to_tensor(np.log(alpha), dtype=tf.float64),
        z_v=tf.convert_to_tensor(np.log(v), dtype=tf.float64),
        z_fc=tf.convert_to_tensor(np.log(fc), dtype=tf.float64),
        z_u_scale=tf.convert_to_tensor(np.log(u_scale), dtype=tf.float64),
    )


def _normalize_theta_true_for_eval(
    theta_true: dict[str, Any],
    M: int,
) -> dict[str, np.ndarray]:
    """Ensure the evaluation truth dictionary contains all standard Phase-3 keys."""
    return {
        "beta": np.asarray(theta_true["beta"], dtype=np.float64),
        "alpha": np.asarray(theta_true["alpha"], dtype=np.float64),
        "v": np.asarray(theta_true["v"], dtype=np.float64),
        "fc": np.asarray(theta_true["fc"], dtype=np.float64),
        "u_scale": np.asarray(
            theta_true.get("u_scale", np.ones((M,), dtype=np.float64)),
            dtype=np.float64,
        ),
    }


def _build_posterior_config(cfg: dict[str, Any]) -> StockpilingPosteriorConfig:
    """Construct the posterior config dataclass from Phase-3 config."""
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
        fix_u_scale=bool(pcfg["fix_u_scale"]),
        fixed_z_u_scale=float(pcfg["fixed_z_u_scale"]),
    )


def _build_sampler_config(cfg: dict[str, Any]) -> StockpilingConfig:
    """Construct the sampler config dataclass from Phase-3 config."""
    scfg = cfg["sampler"]
    return StockpilingConfig(
        num_results=int(scfg["num_results"]),
        num_burnin_steps=int(scfg["num_burnin_steps"]),
        chunk_size=int(scfg["chunk_size"]),
        k_beta=tf.convert_to_tensor(float(scfg["k_beta"]), dtype=tf.float64),
        k_alpha=tf.convert_to_tensor(
            np.asarray(scfg["k_alpha"], dtype=np.float64),
            dtype=tf.float64,
        ),
        k_v=tf.convert_to_tensor(
            np.asarray(scfg["k_v"], dtype=np.float64),
            dtype=tf.float64,
        ),
        k_fc=tf.convert_to_tensor(
            np.asarray(scfg["k_fc"], dtype=np.float64),
            dtype=tf.float64,
        ),
        k_u_scale=tf.convert_to_tensor(
            np.asarray(scfg["k_u_scale"], dtype=np.float64),
            dtype=tf.float64,
        ),
        pilot_num_steps=int(scfg["pilot_num_steps"]),
        target_accept_low=float(scfg["target_accept_low"]),
        target_accept_high=float(scfg["target_accept_high"]),
        grow_factor=float(scfg["grow_factor"]),
        shrink_factor=float(scfg["shrink_factor"]),
        max_tuning_rounds=int(scfg["max_tuning_rounds"]),
    )


def summarize_stockpiling_panel(
    panel: dict[str, Any],
    initial_state: StockpilingState,
) -> None:
    """Print a lightweight summary of the generated Phase-3 panel."""
    a = np.asarray(panel["a_mnjt"], dtype=np.float64)
    s = np.asarray(panel["s_mjt"])
    u = np.asarray(panel["u_mj"], dtype=np.float64)

    print("")
    print("=== Stockpiling data generated ===")
    print(f"shapes: a_mnjt={a.shape} | p_state_mjt={s.shape} | u_mj={u.shape}")
    print(f"overall buy rate: {float(a.mean()):.4f}")

    theta_true = panel.get("theta_true")
    if isinstance(theta_true, dict):
        print(
            "[Stockpiling] True | "
            f'mean(beta)="{float(np.mean(np.asarray(theta_true["beta"], dtype=np.float64))):.4f}" , '
            f'mean(alpha)="{float(np.mean(np.asarray(theta_true["alpha"], dtype=np.float64))):.4f}" , '
            f'mean(v)="{float(np.mean(np.asarray(theta_true["v"], dtype=np.float64))):.4f}" , '
            f'mean(fc)="{float(np.mean(np.asarray(theta_true["fc"], dtype=np.float64))):.4f}"'
        )

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
        f'mean(beta)="{float(theta0["beta"].numpy()):.4f}" , '
        f'mean(alpha)="{float(tf.reduce_mean(theta0["alpha"]).numpy()):.4f}" , '
        f'mean(v)="{float(tf.reduce_mean(theta0["v"]).numpy()):.4f}" , '
        f'mean(fc)="{float(tf.reduce_mean(theta0["fc"]).numpy()):.4f}" , '
        f'mean(u_scale)="{float(tf.reduce_mean(theta0["u_scale"]).numpy()):.4f}"'
    )


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
    """Run Phase 2 Lu-style market-shock estimation and print diagnostics."""
    print("=== Phase 2: Market-shock estimator ===")

    qjt_np = np.asarray(dgp["qjt_shock"], dtype=np.float64)
    q0t_np = np.asarray(dgp["q0t_shock"], dtype=np.float64)
    delta_hat_np = np.asarray(delta_hat, dtype=np.float64)

    T_i, J_i = qjt_np.shape
    delta_cl_np = np.repeat(delta_hat_np[None, :], T_i, axis=0)

    delta_cl = tf.convert_to_tensor(delta_cl_np, dtype=tf.float64)
    qjt = tf.convert_to_tensor(qjt_np, dtype=tf.float64)
    q0t = tf.convert_to_tensor(q0t_np, dtype=tf.float64)

    init_cfg = cfg["init_config"]
    fit_cfg = cfg["fit_config"]
    posterior_cfg = init_cfg["posterior"]
    init_state_cfg = init_cfg["init_state"]

    alpha0 = tf.convert_to_tensor(float(init_state_cfg["alpha"]), dtype=tf.float64)
    E_bar0 = tf.fill([T_i], tf.cast(float(init_state_cfg["E_bar"]), tf.float64))
    njt0 = tf.fill([T_i, J_i], tf.cast(float(init_state_cfg["njt"]), tf.float64))
    gamma0 = tf.fill([T_i, J_i], tf.cast(float(init_state_cfg["gamma"]), tf.float64))
    phi0 = tf.fill([T_i], tf.cast(float(init_state_cfg["phi"]), tf.float64))

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
    """Build Phase-3 fixed utilities from the Phase 1–2 outputs."""
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
    """Generate the Phase-3 seller-observed panel from the Ching DGP."""
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
    """Run the refactored Phase-3 chain and summarize retained samples."""
    a_np = np.asarray(panel["a_mnjt"])
    M = int(a_np.shape[0])
    J = int(a_np.shape[2])

    posterior_config = _build_posterior_config(cfg)
    sampler_config = _build_sampler_config(cfg)
    initial_state = build_initial_state(M=M, J=J)
    chain_seed = tf.constant([int(cfg["mcmc_seed"]), 0], dtype=tf.int32)
    inventory_maps = build_inventory_maps(int(cfg["I_max"]))

    samples = run_chain(
        a_mnjt=tf.convert_to_tensor(panel["a_mnjt"], dtype=tf.int32),
        s_mjt=tf.convert_to_tensor(panel["s_mjt"], dtype=tf.int32),
        u_mj=tf.convert_to_tensor(panel["u_mj"], dtype=tf.float64),
        P_price_mj=tf.convert_to_tensor(P_price_mj, dtype=tf.float64),
        price_vals_mj=tf.convert_to_tensor(price_vals_mj, dtype=tf.float64),
        lambda_mn=tf.convert_to_tensor(panel["lambda_mn"], dtype=tf.float64),
        waste_cost=tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        inventory_maps=inventory_maps,
        pi_I0=tf.convert_to_tensor(
            np.asarray(cfg["pi_I0"], dtype=np.float64),
            dtype=tf.float64,
        ),
        posterior_config=posterior_config,
        stockpiling_config=sampler_config,
        initial_state=initial_state,
        seed=chain_seed,
    )

    theta_hat = _to_numpy_results(summarize_samples(samples, posterior_config))
    theta_hat = {
        "beta": np.asarray(theta_hat["beta_hat"], dtype=np.float64),
        "alpha": np.asarray(theta_hat["alpha_hat"], dtype=np.float64),
        "v": np.asarray(theta_hat["v_hat"], dtype=np.float64),
        "fc": np.asarray(theta_hat["fc_hat"], dtype=np.float64),
        "u_scale": np.asarray(theta_hat["u_scale_hat"], dtype=np.float64),
    }

    return {
        "samples": samples,
        "theta_hat": theta_hat,
        "n_saved": int(samples.z_beta.shape[0]),
        "initial_state": initial_state,
        "posterior_config": posterior_config,
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
    """Evaluate the fitted Phase-3 stockpiling model."""
    a_np = np.asarray(panel["a_mnjt"], dtype=np.float64)
    s_np = np.asarray(panel["s_mjt"])
    u_np = np.asarray(panel["u_mj"], dtype=np.float64)
    lam_np = np.asarray(panel["lambda_mn"], dtype=np.float64)

    M = int(a_np.shape[0])
    inventory_maps = build_inventory_maps(int(cfg["I_max"]))

    posterior = StockpilingPosteriorTF(
        config=posterior_config,
        a_mnjt=tf.convert_to_tensor(panel["a_mnjt"], dtype=tf.int32),
        s_mjt=tf.convert_to_tensor(panel["s_mjt"], dtype=tf.int32),
        u_mj=tf.convert_to_tensor(u_np, dtype=tf.float64),
        P_price_mj=tf.convert_to_tensor(P_price_mj, dtype=tf.float64),
        price_vals_mj=tf.convert_to_tensor(price_vals_mj, dtype=tf.float64),
        lambda_mn=tf.convert_to_tensor(lam_np, dtype=tf.float64),
        waste_cost=tf.constant(float(cfg["waste_cost"]), dtype=tf.float64),
        inventory_maps=inventory_maps,
        pi_I0=tf.convert_to_tensor(
            np.asarray(cfg["pi_I0"], dtype=np.float64),
            dtype=tf.float64,
        ),
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
    theta_true_np: dict[str, np.ndarray] | None = None
    if theta_true is not None:
        theta_true_np = _normalize_theta_true_for_eval(theta_true, M=M)
        oracle_state = _theta_to_state(theta_true_np, M=M)
        p_buy_oracle_mnjt = posterior.predict_p_buy_mnjt(
            z_beta=oracle_state.z_beta,
            z_alpha=oracle_state.z_alpha,
            z_v=oracle_state.z_v,
            z_fc=oracle_state.z_fc,
            z_u_scale=oracle_state.z_u_scale,
        ).numpy()

    theta_hat_np = {
        "beta": np.asarray(theta_hat["beta"], dtype=np.float64),
        "alpha": np.asarray(theta_hat["alpha"], dtype=np.float64),
        "v": np.asarray(theta_hat["v"], dtype=np.float64),
        "fc": np.asarray(theta_hat["fc"], dtype=np.float64),
        "u_scale": np.asarray(theta_hat["u_scale"], dtype=np.float64),
    }

    eval_out = evaluate_stockpiling(
        a_mnjt=a_np,
        p_buy_hat_mnjt=np.asarray(p_buy_hat_mnjt, dtype=np.float64),
        s_mjt=np.asarray(s_np),
        theta_hat=theta_hat_np if theta_true_np is not None else None,
        theta_true=theta_true_np,
        p_buy_oracle_mnjt=p_buy_oracle_mnjt,
        mcmc=mcmc_summary,
        eps=float(EVAL_EPS),
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
    """Run the full Phase 1–3 stockpiling experiment."""
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

    initial_state = build_initial_state(M=M, J=J)
    summarize_stockpiling_panel(panel=panel, initial_state=initial_state)

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
        posterior_config=res3["posterior_config"],
        mcmc_summary=None,
    )


if __name__ == "__main__":
    main()
