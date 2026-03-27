"""Run the end-to-end Bonus Q2 pipeline."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

from datasets.bonus2_dgp import simulate_bonus2_dgp
from run_zhang_with_lu import print_choice_model_diagnostics, run_choice_model

from bonus2 import bonus2_model as b2_model
from bonus2.bonus2_estimator import (
    Bonus2InitConfig,
    Bonus2SamplerConfig,
    run_chain,
    summarize_samples,
)
from bonus2.bonus2_evaluate import evaluate_bonus2, format_evaluation_summary
from bonus2.bonus2_posterior import Bonus2PosteriorConfig


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
    "width": 128,
    "heads": 8,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "shuffle_buffer": 10_000,
    "eval_include_outside": True,
    "eval_against_empirical": True,
}

CFG_BONUS2: dict[str, Any] = {
    "N": 100,
    "T": 365 * 10,
    "season_period": 365,
    "K": 1,
    "lookback": 1,
    "decay": 0.8,
    "avg_friends": 3.0,
    "friends_sd": 1.0,
    "params_true": {
        "habit_mean": 0.60,
        "habit_sd": 0.05,
        "peer_mean": 0.50,
        "peer_sd": 0.15,
        "mktprod_sd": 0.50,
        "weekend_prod_sd": 0.20,
        "season_mkt_sd": 0.20,
    },
    "sigmas": {
        "beta_intercept": 1.0,
        "beta_habit": 1.0,
        "beta_peer": 1.0,
        "beta_weekend": 1.0,
        "a": 1.0,
        "b": 1.0,
    },
    "num_results": 500,
    "num_burnin_steps": 1_500,
    "chunk_size": 100,
    "k_beta_intercept": 0.05,
    "k_beta_habit": 0.05,
    "k_beta_peer": 0.05,
    "k_beta_weekend": 0.05,
    "k_a": 0.05,
    "k_b": 0.05,
    "init_theta": {
        "beta_intercept": 0.0,
        "beta_habit": 0.0,
        "beta_peer": 0.0,
        "beta_weekday": 0.0,
        "beta_weekend": 0.0,
        "a": 0.0,
        "b": 0.0,
    },
    "eps": 1e-12,
}


def _validate_phase1_delta_hat(delta_hat: np.ndarray, num_products: int) -> None:
    """Validate the Phase 1 baseline utility vector."""
    delta_hat = np.asarray(delta_hat, dtype=np.float64)

    if delta_hat.ndim != 1:
        raise ValueError(f"delta_hat must have rank 1; got shape {delta_hat.shape}.")
    if delta_hat.shape[0] != int(num_products):
        raise ValueError(
            f"delta_hat must have length {int(num_products)}; got shape {delta_hat.shape}."
        )
    if not np.all(np.isfinite(delta_hat)):
        raise ValueError("delta_hat must be finite.")


def _tile_delta_mj(delta_hat: np.ndarray, num_markets: int) -> np.ndarray:
    """Tile a product-level baseline utility vector across markets."""
    delta_hat = np.asarray(delta_hat, dtype=np.float64)
    return np.tile(delta_hat[None, :], (int(num_markets), 1))


def _to_numpy_dict(x: dict[str, Any]) -> dict[str, Any]:
    """Convert tensor-valued dictionary entries to NumPy."""
    out: dict[str, Any] = {}
    for key, value in x.items():
        out[key] = value.numpy() if isinstance(value, tf.Tensor) else value
    return out


def _theta_np_to_tf(theta: dict[str, Any]) -> dict[str, tf.Tensor]:
    """Convert a NumPy-backed theta dictionary to tf.float64 tensors."""
    return {
        "beta_intercept_j": tf.convert_to_tensor(
            theta["beta_intercept_j"],
            dtype=tf.float64,
        ),
        "beta_habit_j": tf.convert_to_tensor(theta["beta_habit_j"], dtype=tf.float64),
        "beta_peer_j": tf.convert_to_tensor(theta["beta_peer_j"], dtype=tf.float64),
        "beta_weekend_jw": tf.convert_to_tensor(
            theta["beta_weekend_jw"],
            dtype=tf.float64,
        ),
        "a_m": tf.convert_to_tensor(theta["a_m"], dtype=tf.float64),
        "b_m": tf.convert_to_tensor(theta["b_m"], dtype=tf.float64),
    }


def _build_posterior_config(cfg: dict[str, Any]) -> Bonus2PosteriorConfig:
    """Build the posterior configuration."""
    sigmas = cfg["sigmas"]
    return Bonus2PosteriorConfig(
        sigma_z_beta_intercept_j=float(sigmas["beta_intercept"]),
        sigma_z_beta_habit_j=float(sigmas["beta_habit"]),
        sigma_z_beta_peer_j=float(sigmas["beta_peer"]),
        sigma_z_beta_weekend_jw=float(sigmas["beta_weekend"]),
        sigma_z_a_m=float(sigmas["a"]),
        sigma_z_b_m=float(sigmas["b"]),
    )


def _build_sampler_config(cfg: dict[str, Any]) -> Bonus2SamplerConfig:
    """Build the sampler configuration."""
    return Bonus2SamplerConfig(
        num_results=int(cfg["num_results"]),
        num_burnin_steps=int(cfg["num_burnin_steps"]),
        chunk_size=int(cfg["chunk_size"]),
        k_beta_intercept=float(cfg["k_beta_intercept"]),
        k_beta_habit=float(cfg["k_beta_habit"]),
        k_beta_peer=float(cfg["k_beta_peer"]),
        k_beta_weekend=float(cfg["k_beta_weekend"]),
        k_a=float(cfg["k_a"]),
        k_b=float(cfg["k_b"]),
    )


def _build_init_config(cfg: dict[str, Any]) -> Bonus2InitConfig:
    """Build the initial-state configuration."""
    init_theta = cfg["init_theta"]
    return Bonus2InitConfig(
        init_beta_intercept=float(init_theta["beta_intercept"]),
        init_beta_habit=float(init_theta["beta_habit"]),
        init_beta_peer=float(init_theta["beta_peer"]),
        init_beta_weekday=float(init_theta["beta_weekday"]),
        init_beta_weekend=float(init_theta["beta_weekend"]),
        init_a=float(init_theta["a"]),
        init_b=float(init_theta["b"]),
    )


def summarize_bonus2_panel(
    panel: dict[str, Any],
    theta_true: dict[str, Any],
    init_config: Bonus2InitConfig,
    season_period: int,
    num_harmonics: int,
) -> None:
    """Print a lightweight summary of the simulated Bonus2 panel."""
    y = np.asarray(panel["y_mit"], dtype=np.int64)
    delta = np.asarray(panel["delta_mj"], dtype=np.float64)
    is_weekend = np.asarray(panel["is_weekend_t"], dtype=np.int64)
    season_sin_kt = np.asarray(panel["season_sin_kt"], dtype=np.float64)
    neighbors_m = panel["neighbors_m"]

    num_markets, num_consumers, num_periods = y.shape
    num_products = int(delta.shape[1])

    outside_share = float(np.mean(y == 0))
    inside_shares = np.array(
        [np.mean(y == (j + 1)) for j in range(num_products)],
        dtype=np.float64,
    )

    degrees: list[int] = []
    for m in range(num_markets):
        for i in range(num_consumers):
            degrees.append(int(len(neighbors_m[m][i])))
    avg_degree = float(np.mean(degrees)) if degrees else 0.0

    print("=== Bonus2 data generated ===")
    print(
        "shapes: "
        f"y_mit={y.shape} | delta_mj={delta.shape} | "
        f"is_weekend_t={is_weekend.shape} | season_sin_kt={season_sin_kt.shape}"
    )
    print(
        f"outside_share={outside_share:.4f} | "
        f"inside_share_mean={float(inside_shares.mean()):.4f} | "
        f"inside_share_max={float(inside_shares.max()):.4f}"
    )
    print(
        f"avg_out_degree={avg_degree:.2f} | "
        f"weekend_share={float(np.mean(is_weekend)):.4f} | "
        f"lookback={int(panel['lookback'])} | "
        f"decay={float(panel['decay']):.4f} | "
        f"season_period={int(season_period)} | "
        f"K={int(num_harmonics)}"
    )

    print("true parameter means:")
    print(
        f"  intercept={float(np.mean(theta_true['beta_intercept_j'])):.4f} | "
        f"habit={float(np.mean(theta_true['beta_habit_j'])):.4f} | "
        f"peer={float(np.mean(theta_true['beta_peer_j'])):.4f}"
    )
    print(
        f"  weekend_weekday={float(np.mean(theta_true['beta_weekend_jw'][:, 0])):.4f} | "
        f"weekend_weekend={float(np.mean(theta_true['beta_weekend_jw'][:, 1])):.4f} | "
        f"a_mean={float(np.mean(theta_true['a_m'])):.4f} | "
        f"b_mean={float(np.mean(theta_true['b_m'])):.4f}"
    )

    print("initial parameter fills:")
    print(
        f"  intercept={init_config.init_beta_intercept:.4f} | "
        f"habit={init_config.init_beta_habit:.4f} | "
        f"peer={init_config.init_beta_peer:.4f}"
    )
    print(
        f"  weekday={init_config.init_beta_weekday:.4f} | "
        f"weekend={init_config.init_beta_weekend:.4f} | "
        f"a={init_config.init_a:.4f} | "
        f"b={init_config.init_b:.4f}"
    )


def _predict_choice_probs(panel: dict[str, Any], theta: dict[str, Any]) -> np.ndarray:
    """Predict choice probabilities from a structural theta dictionary."""
    y = np.asarray(panel["y_mit"], dtype=np.int32)
    delta = np.asarray(panel["delta_mj"], dtype=np.float64)
    is_weekend = np.asarray(panel["is_weekend_t"], dtype=np.int32)
    season_sin_kt = np.asarray(panel["season_sin_kt"], dtype=np.float64)
    season_cos_kt = np.asarray(panel["season_cos_kt"], dtype=np.float64)

    y_tf = tf.convert_to_tensor(y, dtype=tf.int32)
    delta_tf = tf.convert_to_tensor(delta, dtype=tf.float64)
    is_weekend_tf = tf.convert_to_tensor(is_weekend, dtype=tf.int32)
    season_sin_tf = tf.convert_to_tensor(season_sin_kt, dtype=tf.float64)
    season_cos_tf = tf.convert_to_tensor(season_cos_kt, dtype=tf.float64)

    num_consumers = int(y.shape[1])
    num_products = int(delta.shape[1])

    peer_adj_m = b2_model.build_peer_adjacency(
        neighbors_m=panel["neighbors_m"],
        n_consumers=num_consumers,
    )
    h_mntj, p_mntj = b2_model.build_deterministic_states(
        y_mit=y_tf,
        n_products=tf.constant(num_products, dtype=tf.int32),
        peer_adj_m=peer_adj_m,
        lookback=tf.constant(int(panel["lookback"]), dtype=tf.int32),
        decay=tf.constant(float(panel["decay"]), dtype=tf.float64),
    )

    p_choice = b2_model.predict_choice_probs_from_theta(
        theta=_theta_np_to_tf(theta),
        delta_mj=delta_tf,
        is_weekend_t=is_weekend_tf,
        season_sin_kt=season_sin_tf,
        season_cos_kt=season_cos_tf,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
    )
    return p_choice.numpy()


def run_phase1(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the Phase 1 baseline choice model."""
    print("=== Phase 1: Baseline choice model (Zhang feature-based) ===")

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

    dgp = out["dgp"]
    delta_hat = np.asarray(out["delta_hat"], dtype=np.float64)

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
    cfg: dict[str, Any],
    delta_mj: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Run the Bonus2 DGP."""
    print(
        "=== Bonus2 DGP: Generating y_mit under habit + peer + weekday/weekend + market seasonality ==="
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

    print("=== Bonus2 DGP complete ===")
    return {
        "panel": out["panel"],
        "theta_true": out["theta_true"],
        "season_period": int(cfg["season_period"]),
        "K": int(cfg["K"]),
    }


def run_bonus2_estimation(cfg: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    """Run the Bonus2 estimator and normalize its outputs."""
    print("=== Bonus2 estimation ===")

    posterior_config = _build_posterior_config(cfg)
    sampler_config = _build_sampler_config(cfg)
    init_config = _build_init_config(cfg)

    y_mit = tf.convert_to_tensor(
        np.asarray(panel["y_mit"], dtype=np.int32), dtype=tf.int32
    )
    delta_mj = tf.convert_to_tensor(
        np.asarray(panel["delta_mj"], dtype=np.float64),
        dtype=tf.float64,
    )
    is_weekend_t = tf.convert_to_tensor(
        np.asarray(panel["is_weekend_t"], dtype=np.int32),
        dtype=tf.int32,
    )
    season_sin_kt = tf.convert_to_tensor(
        np.asarray(panel["season_sin_kt"], dtype=np.float64),
        dtype=tf.float64,
    )
    season_cos_kt = tf.convert_to_tensor(
        np.asarray(panel["season_cos_kt"], dtype=np.float64),
        dtype=tf.float64,
    )

    samples, chunk_summaries = run_chain(
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        neighbors_m=panel["neighbors_m"],
        lookback=int(panel["lookback"]),
        decay=float(panel["decay"]),
        posterior_config=posterior_config,
        sampler_config=sampler_config,
        init_config=init_config,
    )

    theta_hat = _to_numpy_dict(summarize_samples(samples))
    n_saved = int(samples.z_beta_intercept_j.shape[0])

    print("=== Bonus2 estimation complete ===")
    return {
        "samples": samples,
        "chunk_summaries": chunk_summaries,
        "theta_hat": theta_hat,
        "n_saved": n_saved,
    }


def main() -> None:
    """Run the full Bonus Q2 pipeline."""
    out_phase1 = run_phase1(CFG_PHASE1)
    delta_hat = out_phase1["delta_hat"]

    _validate_phase1_delta_hat(
        delta_hat=delta_hat,
        num_products=int(CFG_PHASE1["num_products"]),
    )

    num_markets = int(CFG_PHASE1["num_markets"])
    delta_mj = _tile_delta_mj(delta_hat=delta_hat, num_markets=num_markets)

    dgp_seed = int(CFG_PHASE1["seed"]) + 999
    dgp_out = run_bonus2_dgp(CFG_BONUS2, delta_mj=delta_mj, seed=dgp_seed)

    panel = dgp_out["panel"]
    theta_true = dgp_out["theta_true"]

    init_config = _build_init_config(CFG_BONUS2)
    summarize_bonus2_panel(
        panel=panel,
        theta_true=theta_true,
        init_config=init_config,
        season_period=int(dgp_out["season_period"]),
        num_harmonics=int(dgp_out["K"]),
    )

    est_out = run_bonus2_estimation(CFG_BONUS2, panel=panel)
    theta_hat = est_out["theta_hat"]

    p_choice_hat = _predict_choice_probs(panel=panel, theta=theta_hat)
    p_choice_oracle = _predict_choice_probs(panel=panel, theta=theta_true)

    evaluation = evaluate_bonus2(
        y_mit=np.asarray(panel["y_mit"], dtype=np.int64),
        delta_mj=np.asarray(panel["delta_mj"], dtype=np.float64),
        p_choice_hat_mntc=p_choice_hat,
        p_choice_oracle_mntc=p_choice_oracle,
        theta_hat=theta_hat,
        theta_true=theta_true,
        chunk_summaries=est_out["chunk_summaries"],
        n_saved=int(est_out["n_saved"]),
        eps=float(CFG_BONUS2["eps"]),
    )
    print(format_evaluation_summary(evaluation))


if __name__ == "__main__":
    main()
