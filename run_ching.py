"""
End-to-end orchestration for:

- Phase 1: feature-based baseline choice model (delta_hat)
- Phase 2: Lu-style shrinkage on market-product shocks (E_bar_hat, njt_hat, alpha_hat)
- Phase 3 (stockpiling): generate seller-observed purchases under a forward-looking
  inventory model for all products, with market×product-specific price dynamics and
  market×product-specific price levels, then estimate stockpiling parameters treating
  Phase 1–2 utilities as fixed.

Phase-3 target model (high level):
  - Observed: a_mnjt (M,N,J,T), p_state_mjt (M,J,T), and known (P_price_mj, price_vals_mj)
  - Fixed utilities from Phase 1–2: u_mj = delta_j + E_bar_m + n_mj
  - Parameters:
      per market-product:  beta_mj, alpha_mj, v_mj, fc_mj
      per market:          u_scale_m
      per market-consumer: lambda_mn
"""

from __future__ import annotations

import os

import numpy as np

# Keep TF logs quiet for downstream modules that may import TF internally.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from datasets.ching_dgp import generate_dgp  # noqa: E402
from run_zhang_with_lu import (  # noqa: E402
    print_choice_model_diagnostics,
    print_market_shock_diagnostics,
    run_choice_model,
    run_market_shock_estimator,
)
from ching.stockpiling_estimator import StockpilingEstimator  # noqa: E402
from ching.stockpiling_evaluate import (  # noqa: E402
    evaluate_stockpiling,
    format_evaluation_summary,
)
from ching.stockpiling_model import (  # noqa: E402
    build_inventory_maps,
    predict_p_buy_mnjt_from_theta,
)

# =============================================================================
# Configuration
# =============================================================================

# Phase 1: baseline choice model
CFG_PHASE1 = {
    "seed": 123,
    "num_products": 15,
    "num_groups": 5,
    "num_markets": 5,
    "N_base": 2_000,
    "N_shock": 1_000,
    "x_sd": 1.0,
    "coef_sd": 1.0,
    "p_g_active": 0.2,
    "g_sd": None,
    "sd_E": 0.5,
    "p_active": 0.25,
    "sd_u": 0.5,
    "depth": 10,
    "width": 64,
    "heads": 8,
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "shuffle_buffer": 10_000,
    "eval_include_outside": True,
    "eval_against_empirical": True,
}

# Phase 2: market shock estimator
CFG_PHASE2 = {
    "seed": 0,
    "n_iter": 10,
    "pilot_length": 20,
    "max_rounds": 50,
    "target_low": 0.3,
    "target_high": 0.5,
    "factor_rw": 1.2,
    "factor_tmh": 1.2,
    "ridge": 1e-6,
}

# Phase 3 (stockpiling): DGP + estimation
CFG_PHASE3 = {
    # panel size
    "N": 50,
    "T": 1000,
    # inventory state
    "I_max": 10,
    # price state space
    "S": 2,
    # DGP/likelihood config
    "waste_cost": 1.0,
    "dp_tol": 1e-10,
    "dp_max_iter": 50_000,
    "eps": 1e-12,
    # exogenous price process construction
    "price_seed": 777,
    "p_stay": 0.85,
    "P_noise_sd": 0.05,
    "P_min_prob": 1e-6,
    "price_base_low": 0.7,
    "price_base_high": 1.3,
    "discount_low": 0.10,
    "discount_high": 0.35,
    "price_noise_sd": 0.02,
    # MCMC config
    "mcmc_seed": 0,
    "mcmc_n_iter": 3,
    "k": {
        "beta": 1,
        "alpha": 0.2,
        "v": 0.3,
        "fc": 0.15,
        "lambda": 0.3,
        "u_scale": 0.02,
    },
    "sigmas": {
        "z_beta": 2.0,
        "z_alpha": 2.0,
        "z_v": 2.0,
        "z_fc": 2.0,
        "z_lambda": 2.0,
        "z_u_scale": 2.0,
    },
}


# =============================================================================
# Utilities
# =============================================================================


def _uniform_pi_I0(I_max: int) -> np.ndarray:
    """
    Uniform initial-inventory belief over {0, ..., I_max}.

    Returns:
      ndarray: shape (I_max + 1,), float64.
    """
    return np.full((I_max + 1,), 1.0 / (I_max + 1), dtype=np.float64)


def _build_base_transition(S: int, p_stay: float) -> np.ndarray:
    """
    Build a simple, ergodic SxS transition matrix with local moves.

    - Interior states split (1 - p_stay) equally to neighbors.
    - Endpoints allocate all (1 - p_stay) to their single neighbor.
    """
    if S < 2:
        raise ValueError("S must be >= 2")
    if not (0.0 < p_stay < 1.0):
        raise ValueError("p_stay must be in (0,1)")

    P = np.zeros((S, S), dtype=np.float64)
    for s in range(S):
        P[s, s] = p_stay
        rem = 1.0 - p_stay
        if s == 0:
            P[s, 1] = rem
        elif s == S - 1:
            P[s, S - 2] = rem
        else:
            P[s, s - 1] = 0.5 * rem
            P[s, s + 1] = 0.5 * rem
    return P


def _build_price_transitions(
    rng: np.random.Generator,
    M: int,
    J: int,
    S: int,
    p_stay: float,
    noise_sd: float,
    min_prob: float,
) -> np.ndarray:
    """
    Construct market×product-specific price-state transitions.

    Returns:
      ndarray: P_price_mj with shape (M, J, S, S), row-stochastic float64.
    """
    P_base = _build_base_transition(S=S, p_stay=p_stay)
    P_price_mj = np.empty((M, J, S, S), dtype=np.float64)

    for m in range(M):
        for j in range(J):
            P = P_base.copy()
            P = P + noise_sd * rng.normal(size=(S, S))
            P = np.clip(P, min_prob, None)
            P = P / P.sum(axis=1, keepdims=True)
            P_price_mj[m, j] = P

    return P_price_mj


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
    """
    Construct market×product-specific price levels per price state.

    Convention:
      - state 0 is the highest price level
      - price levels are monotone decreasing across states

    Returns:
      ndarray: price_vals_mj with shape (M, J, S), float64.
    """
    price_vals_mj = np.empty((M, J, S), dtype=np.float64)

    for m in range(M):
        for j in range(J):
            base = rng.uniform(base_low, base_high)
            span = rng.uniform(discount_low, discount_high)

            if S == 2:
                levels = np.array([base, base * (1.0 - span)], dtype=np.float64)
            else:
                frac = np.linspace(0.0, 1.0, S, dtype=np.float64)
                levels = base * (1.0 - span * frac)

            # small multiplicative noise but preserve ordering
            levels = levels * np.exp(noise_sd * rng.normal(size=S))
            levels = np.sort(levels)[::-1]
            price_vals_mj[m, j] = levels

    return price_vals_mj


def build_price_processes(
    M: int,
    J: int,
    S: int,
    seed_price: int,
    *,
    p_stay: float,
    P_noise_sd: float,
    P_min_prob: float,
    price_base_low: float,
    price_base_high: float,
    discount_low: float,
    discount_high: float,
    price_noise_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct market×product-specific price dynamics and price levels.

    Returns:
      P_price_mj:   (M, J, S, S) row-stochastic transitions
      price_vals_mj:(M, J, S)    price levels per state
    """
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


def summarize_stockpiling_panel(panel: dict[str, object]) -> None:
    """Lightweight prints for Phase-3 data objects."""
    a_mnjt = panel["a_mnjt"]  # (M,N,J,T)
    p_state_mjt = panel["p_state_mjt"]  # (M,J,T)
    u_mj = panel["u_mj"]  # (M,J)

    buy_rate = float(np.mean(a_mnjt))
    print("=== Stockpiling data generated ===")
    print(
        "shapes: "
        f"a_mnjt={a_mnjt.shape} | p_state_mjt={p_state_mjt.shape} | u_mj={u_mj.shape}"
    )
    print(f"overall buy rate: {buy_rate:.4f}")


# =============================================================================
# Phase runners
# =============================================================================


def run_phase1(cfg: dict[str, object]) -> dict[str, object]:
    """Run Phase 1 baseline choice model and print diagnostics."""
    print("=== Phase 1: Baseline choice model ===")

    out1 = run_choice_model(
        seed=int(cfg["seed"]),
        num_products=int(cfg["num_products"]),
        num_groups=int(cfg["num_groups"]),
        num_markets=int(cfg["num_markets"]),
        N_base=int(cfg["N_base"]),
        N_shock=int(cfg["N_shock"]),
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


def run_phase2(
    cfg: dict[str, object],
    *,
    dgp: dict[str, object],
    delta_hat: np.ndarray,
    eval_include_outside: bool,
) -> dict[str, object]:
    """Run Phase 2 market shock estimator and print diagnostics."""
    print("=== Phase 2: Market shock estimator ===")

    res2 = run_market_shock_estimator(
        delta_hat=delta_hat,
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
    print("=== Market shock estimator fitted ===")

    print_market_shock_diagnostics(
        delta_hat=delta_hat,
        dgp=dgp,
        res=res2,
        eval_include_outside=eval_include_outside,
    )
    print("=== Phase 2 complete ===")

    return res2


def build_phase3_inputs(
    *,
    delta_hat: np.ndarray,
    res2: dict[str, object],
) -> dict[str, np.ndarray]:
    """
    Build Phase-3 utility objects from Phase-2 outputs.

    Returns:
      dict with:
        - delta_used: (J,) float64
        - E_bar_used: (M,) float64
        - njt_used:   (M,J) float64
        - alpha_hat:  scalar float64 (returned for debugging)
    """
    alpha_hat = float(res2["alpha_hat"])
    E_bar_used = np.asarray(res2["E_bar_hat"], dtype=np.float64)
    njt_used = np.asarray(res2["njt_hat"], dtype=np.float64)

    # Rescale Phase-1 utilities by the Phase-2 normalization so Phase-3 sees
    # utilities on the intended scale.
    delta_used = alpha_hat * np.asarray(delta_hat, dtype=np.float64)

    return {
        "alpha_hat": np.asarray(alpha_hat, dtype=np.float64),
        "delta_used": delta_used,
        "E_bar_used": E_bar_used,
        "njt_used": njt_used,
    }


def run_phase3_dgp(
    cfg: dict[str, object],
    *,
    delta_used: np.ndarray,
    E_bar_used: np.ndarray,
    njt_used: np.ndarray,
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    seed_dgp: int,
) -> dict[str, object]:
    """
    Generate seller-observed stockpiling panel data for all products.

    Returns:
      dict with:
        - a_mnjt      (M,N,J,T) int64 purchases
        - p_state_mjt (M,J,T)   int64 price states
        - u_mj        (M,J)     float64 intercepts (unscaled)
        - theta_true  dict of true parameters
    """
    a_mnjt, p_state_mjt, u_mj, theta_true = generate_dgp(
        seed=int(seed_dgp),
        delta_true=np.asarray(delta_used, dtype=np.float64),
        E_bar_true=np.asarray(E_bar_used, dtype=np.float64),
        njt_true=np.asarray(njt_used, dtype=np.float64),
        N=int(cfg["N"]),
        T=int(cfg["T"]),
        I_max=int(cfg["I_max"]),
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        waste_cost=float(cfg["waste_cost"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
    )
    return {
        "a_mnjt": a_mnjt,
        "p_state_mjt": p_state_mjt,
        "u_mj": u_mj,
        "theta_true": theta_true,
    }


def run_phase3_estimation(
    cfg: dict[str, object],
    *,
    panel: dict[str, object],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
) -> dict[str, object]:
    """
    Fit stockpiling parameters from seller-observed multi-product data.

    Returns:
      dict including:
        - "theta_hat"
        - "n_saved"
        - "accept"
    """
    est = StockpilingEstimator(
        a_mnjt=panel["a_mnjt"],
        p_state_mjt=panel["p_state_mjt"],
        u_mj=panel["u_mj"],
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        I_max=int(cfg["I_max"]),
        pi_I0=_uniform_pi_I0(int(cfg["I_max"])),
        waste_cost=float(cfg["waste_cost"]),
        eps=float(cfg["eps"]),
        tol=float(cfg["dp_tol"]),
        max_iter=int(cfg["dp_max_iter"]),
        sigmas=cfg["sigmas"],
        seed=int(cfg["mcmc_seed"]),
    )
    print("=== Stockpiling Estimator built ===")

    k = cfg["k"]
    est.fit(
        n_iter=int(cfg["mcmc_n_iter"]),
        k={
            "beta": float(k["beta"]),
            "alpha": float(k["alpha"]),
            "v": float(k["v"]),
            "fc": float(k["fc"]),
            "lambda": float(k["lambda"]),
            "u_scale": float(k["u_scale"]),
        },
    )

    print("=== Stockpiling Estimator fitted ===")

    return est.get_results()


def run_phase3_evaluation(
    cfg: dict[str, object],
    *,
    panel: dict[str, object],
    P_price_mj: np.ndarray,
    price_vals_mj: np.ndarray,
    theta_hat: dict[str, np.ndarray],
    theta_true: dict[str, np.ndarray],
    mcmc_diag: dict[str, object],
) -> dict[str, object]:
    """Compute predictive probabilities and run the stockpiling evaluation."""
    I_max = int(cfg["I_max"])

    maps = build_inventory_maps(tf.constant(I_max, dtype=tf.int32))

    # Fixed TF inputs (shared across oracle and fitted predictions)
    a_tf = tf.convert_to_tensor(panel["a_mnjt"])  # (M,N,J,T), int64
    p_state_tf = tf.convert_to_tensor(panel["p_state_mjt"], dtype=tf.int32)  # (M,J,T)
    u_mj_tf = tf.convert_to_tensor(panel["u_mj"], dtype=tf.float64)  # (M,J)

    price_vals_tf = tf.convert_to_tensor(price_vals_mj, dtype=tf.float64)  # (M,J,S)
    P_price_tf = tf.convert_to_tensor(P_price_mj, dtype=tf.float64)  # (M,J,S,S)

    pi_I0_tf = tf.convert_to_tensor(_uniform_pi_I0(I_max), dtype=tf.float64)  # (I,)
    waste_cost_tf = tf.constant(float(cfg["waste_cost"]), dtype=tf.float64)
    eps_tf = tf.constant(float(cfg["eps"]), dtype=tf.float64)
    tol_tf = tf.constant(float(cfg["dp_tol"]), dtype=tf.float64)
    max_iter_tf = tf.constant(int(cfg["dp_max_iter"]), dtype=tf.int32)

    def theta_to_tf(theta_np: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
        return {
            k: tf.convert_to_tensor(v, dtype=tf.float64) for k, v in theta_np.items()
        }

    def predict(theta_np: dict[str, np.ndarray]) -> np.ndarray:
        theta_tf = theta_to_tf(theta_np)
        return predict_p_buy_mnjt_from_theta(
            theta=theta_tf,
            a_mnjt=a_tf,
            p_state_mjt=p_state_tf,
            u_mj=u_mj_tf,
            price_vals_mj=price_vals_tf,
            P_price_mj=P_price_tf,
            pi_I0=pi_I0_tf,
            waste_cost=waste_cost_tf,
            eps=eps_tf,
            tol=tol_tf,
            max_iter=max_iter_tf,
            maps=maps,
        ).numpy()

    p_buy_hat_mnjt = predict(theta_hat)
    p_buy_oracle_mnjt = predict(theta_true)

    return evaluate_stockpiling(
        a_mnjt=panel["a_mnjt"],
        p_buy_hat_mnjt=p_buy_hat_mnjt,
        p_state_mjt=panel["p_state_mjt"],
        theta_hat=theta_hat,
        theta_true=theta_true,
        p_buy_oracle_mnjt=p_buy_oracle_mnjt,
        mcmc=mcmc_diag,
        eps=float(cfg["eps"]),
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run Phase 1–2 utilities, then Phase-3 DGP, estimation, and evaluation."""
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

    # Phase 3: build market×product-specific price dynamics and levels.
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

    # Phase 3: DGP
    print("=== Stockpiling DGP: Generating seller-observed data ===")
    panel = run_phase3_dgp(
        CFG_PHASE3,
        delta_used=delta_used,
        E_bar_used=E_bar_used,
        njt_used=njt_used,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        seed_dgp=int(CFG_PHASE1["seed"]) + 999,
    )
    summarize_stockpiling_panel(panel)

    # Phase 3: estimation
    print("=== Stockpiling estimation ===")
    res3 = run_phase3_estimation(
        CFG_PHASE3,
        panel=panel,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
    )

    # Phase 3: predictive probabilities + evaluation
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
