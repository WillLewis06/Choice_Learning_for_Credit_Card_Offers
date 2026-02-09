# run_cl_with_shocks.py
"""
Phase 1–2 orchestration for:
- Feature-based choice model (baseline utilities)
- Market-product shock recovery via Lu-style shrinkage

This file is Phase-3 agnostic.
"""

from __future__ import annotations

import os
import math
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datasets.cl_with_shocks_dgp import generate_choice_learn_market_shocks_dgp
from models.featurebased import BaseFeatureBasedDeepHalo
from market_shock_estimators.choice_learn.choice_learn_shrinkage import (
    ChoiceLearnShrinkageEstimator,
)

# -----------------------------
# Helpers (shared)
# -----------------------------


def build_items_tensor(xj: np.ndarray) -> tf.Tensor:
    x = tf.convert_to_tensor(np.asarray(xj, dtype=np.float32))
    items = tf.stack([x, tf.square(x)], axis=-1)
    return items[None, :, :]


def build_choice_index_tensor(qj_base: np.ndarray) -> tf.Tensor:
    q = np.asarray(qj_base, dtype=np.int64)
    idx = np.repeat(np.arange(q.shape[0], dtype=np.int64), q)
    return tf.convert_to_tensor(idx, dtype=tf.int32)


def make_training_dataset(
    items_one: tf.Tensor,
    choices: tf.Tensor,
    *,
    batch_size: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(choices)
    ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    def to_xy(y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        b = tf.shape(y_batch)[0]
        x_batch = tf.tile(items_one, multiples=[b, 1, 1])
        return x_batch, y_batch

    ds = ds.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def probs_with_outside(u: np.ndarray) -> tuple[np.ndarray, float]:
    u = np.asarray(u, dtype=np.float64)
    m = max(0.0, float(np.max(u)))
    exp_inside = np.exp(u - m)
    exp_out = math.exp(-m)
    denom = exp_out + float(np.sum(exp_inside))
    return exp_inside / denom, float(exp_out / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


# -----------------------------
# Phase 1: choice model
# -----------------------------


def run_choice_model(
    *,
    seed: int,
    num_products: int,
    num_groups: int,
    num_markets: int,
    N_base: int,
    N_shock: int,
    x_sd: float,
    coef_sd: float,
    p_g_active: float,
    g_sd: float | None,
    sd_E: float,
    p_active: float,
    sd_u: float,
    depth: int,
    width: int,
    heads: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shuffle_buffer: int,
) -> dict[str, object]:
    dgp = generate_choice_learn_market_shocks_dgp(
        seed=seed,
        num_markets=num_markets,
        num_products=num_products,
        num_groups=num_groups,
        N_base=N_base,
        N_shock=N_shock,
        x_sd=x_sd,
        coef_sd=coef_sd,
        p_g_active=p_g_active,
        g_sd=g_sd,
        sd_E=sd_E,
        p_active=p_active,
        sd_u=sd_u,
    )

    xj = dgp["xj"]
    qj_base = dgp["qj_base"]

    items_one = build_items_tensor(xj)
    choices = build_choice_index_tensor(qj_base)

    ds_train = make_training_dataset(
        items_one=items_one,
        choices=choices,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
    )

    model = BaseFeatureBasedDeepHalo(
        num_items=num_products,
        depth=depth,
        width=width,
        heads=heads,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(ds_train, epochs=epochs, verbose=2)
    delta_hat = model(items_one, training=False).numpy()[0]

    return {
        "dgp": dgp,
        "delta_hat": delta_hat,
    }


def print_choice_model_diagnostics(
    *,
    delta_hat: np.ndarray,
    delta_true: np.ndarray,
    qj_base: np.ndarray,
    q0_base: int,
    p_base: np.ndarray,
    p0_base: float,
    N_base: int,
    eval_include_outside: bool,
    eval_against_empirical: bool,
) -> None:
    p_hat, p0_hat = probs_with_outside(delta_hat)

    print("")
    print("============================================================")
    print("PHASE 1: BASELINE CHOICE MODEL")
    print("============================================================")

    print(f"N_base: {N_base} | N_inside: {int(qj_base.sum())} | N_outside: {q0_base}")
    print(f"rmse_prob_inside_vs_true: {rmse(p_hat, p_base):.6f}")

    if eval_include_outside:
        print(
            f"rmse_prob_all_vs_true:    {rmse(np.r_[p0_hat, p_hat], np.r_[p0_base, p_base]):.6f}"
        )

    if eval_against_empirical:
        s_hat = qj_base / float(N_base)
        s0_hat = q0_base / float(N_base)
        print(f"rmse_prob_inside_vs_share:{rmse(p_hat, s_hat):.6f}")
        if eval_include_outside:
            print(
                f"rmse_prob_all_vs_share:   {rmse(np.r_[p0_hat, p_hat], np.r_[s0_hat, s_hat]):.6f}"
            )

    print(
        f"utility recovery (centered): corr={corr(delta_hat - delta_hat.mean(), delta_true - delta_true.mean()):.4f} "
        f"| rmse={rmse(delta_hat - delta_hat.mean(), delta_true - delta_true.mean()):.4f}"
    )


# -----------------------------
# Phase 2: market shock estimator
# -----------------------------


def run_market_shock_estimator(
    *,
    delta_hat: np.ndarray,
    qjt_shock: np.ndarray,
    q0t_shock: np.ndarray,
    seed: int,
    n_iter: int,
    pilot_length: int,
    max_rounds: int,
    target_low: float,
    target_high: float,
    factor_rw: float,
    factor_tmh: float,
    ridge: float,
) -> dict[str, object]:
    T = qjt_shock.shape[0]
    delta_cl = np.repeat(delta_hat[None, :], T, axis=0)

    shrink = ChoiceLearnShrinkageEstimator(
        delta_cl=delta_cl,
        qjt=qjt_shock,
        q0t=q0t_shock,
        seed=seed,
    )

    shrink.fit(
        n_iter=n_iter,
        pilot_length=pilot_length,
        ridge=ridge,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor_rw=factor_rw,
        factor_tmh=factor_tmh,
    )

    return shrink.get_results()


def print_market_shock_diagnostics(
    *,
    delta_hat: np.ndarray,
    dgp: dict[str, object],
    res: dict[str, object],
    eval_include_outside: bool,
) -> None:
    qjt = np.asarray(dgp["qjt_shock"], dtype=np.float64)
    q0t = np.asarray(dgp["q0t_shock"], dtype=np.float64)

    delta_true = np.asarray(dgp["delta_true"], dtype=np.float64)
    E_bar_true = np.asarray(dgp["E_bar_true"], dtype=np.float64)
    njt_true = np.asarray(dgp["njt_true"], dtype=np.float64)

    alpha_hat = float(res["alpha_hat"])
    E_bar_hat = np.asarray(res["E_bar_hat"], dtype=np.float64)
    njt_hat = np.asarray(res["njt_hat"], dtype=np.float64)
    gamma_hat = np.asarray(res["gamma_hat"], dtype=np.float64)
    phi_hat = np.asarray(res["phi_hat"], dtype=np.float64)

    T = qjt.shape[0]

    nll_base = nll_post = nll_oracle = 0.0

    for t in range(T):
        N_t = float(qjt[t].sum() + q0t[t])

        pb_j, pb_0 = probs_with_outside(delta_hat)
        pp_j, pp_0 = probs_with_outside(
            alpha_hat * delta_hat + E_bar_hat[t] + njt_hat[t]
        )
        po_j, po_0 = probs_with_outside(delta_true + E_bar_true[t] + njt_true[t])

        nll_base -= np.sum(qjt[t] * np.log(pb_j)) + q0t[t] * math.log(pb_0)
        nll_post -= np.sum(qjt[t] * np.log(pp_j)) + q0t[t] * math.log(pp_0)
        nll_oracle -= np.sum(qjt[t] * np.log(po_j)) + q0t[t] * math.log(po_0)

    print("")
    print("============================================================")
    print("PHASE 2: MARKET SHOCK ESTIMATOR")
    print("============================================================")
    print(f"alpha_hat: {alpha_hat:.4f}")
    print(
        f"phi_hat: mean={float(phi_hat.mean()):.4f} | "
        f"min={float(phi_hat.min()):.4f} | max={float(phi_hat.max()):.4f}"
    )
    print("")
    print("Average NLL per market:")
    print(f"  baseline-only:     {nll_base / T:.3f}")
    print(f"  posterior-mean:   {nll_post / T:.3f}")
    print(f"  oracle (truth):   {nll_oracle / T:.3f}")


# -----------------------------
# Main (thin wrapper)
# -----------------------------


def main() -> None:
    # --- config (unchanged defaults) ---
    seed = 123
    num_products = 15
    num_groups = 5
    num_markets = 10

    N_base = 2_000
    N_shock = 1_000

    x_sd = 1.0
    coef_sd = 1.0
    p_g_active = 0.2
    g_sd = None

    sd_E = 0.5
    p_active = 0.25
    sd_u = 0.5

    depth = 10
    width = 64
    heads = 8

    epochs = 5
    batch_size = 64
    learning_rate = 1e-3
    shuffle_buffer = 1_000

    shrink_seed = 0
    shrink_n_iter = 10
    shrink_pilot_length = 20
    shrink_max_rounds = 50
    shrink_target_low = 0.3
    shrink_target_high = 0.5
    shrink_factor_rw = 1.2
    shrink_factor_tmh = 1.2
    shrink_ridge = 1e-6

    eval_include_outside = True
    eval_against_empirical = True

    # --- Phase 1 ---
    out1 = run_choice_model(
        seed=seed,
        num_products=num_products,
        num_groups=num_groups,
        num_markets=num_markets,
        N_base=N_base,
        N_shock=N_shock,
        x_sd=x_sd,
        coef_sd=coef_sd,
        p_g_active=p_g_active,
        g_sd=g_sd,
        sd_E=sd_E,
        p_active=p_active,
        sd_u=sd_u,
        depth=depth,
        width=width,
        heads=heads,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shuffle_buffer=shuffle_buffer,
    )

    dgp = out1["dgp"]
    delta_hat = out1["delta_hat"]

    print_choice_model_diagnostics(
        delta_hat=delta_hat,
        delta_true=dgp["delta_true"],
        qj_base=dgp["qj_base"],
        q0_base=int(dgp["q0_base"]),
        p_base=dgp["p_base"],
        p0_base=float(dgp["p0_base"]),
        N_base=N_base,
        eval_include_outside=eval_include_outside,
        eval_against_empirical=eval_against_empirical,
    )

    # --- Phase 2 ---
    res2 = run_market_shock_estimator(
        delta_hat=delta_hat,
        qjt_shock=dgp["qjt_shock"],
        q0t_shock=dgp["q0t_shock"],
        seed=shrink_seed,
        n_iter=shrink_n_iter,
        pilot_length=shrink_pilot_length,
        max_rounds=shrink_max_rounds,
        target_low=shrink_target_low,
        target_high=shrink_target_high,
        factor_rw=shrink_factor_rw,
        factor_tmh=shrink_factor_tmh,
        ridge=shrink_ridge,
    )

    print_market_shock_diagnostics(
        delta_hat=delta_hat,
        dgp=dgp,
        res=res2,
        eval_include_outside=eval_include_outside,
    )


if __name__ == "__main__":
    main()
