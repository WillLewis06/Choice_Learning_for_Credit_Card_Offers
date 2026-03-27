"""
Phase 1-2 orchestration for:
- feature-based Zhang choice model (baseline utilities)
- choice-learn market-shock recovery with the refactored shrinkage runner

This file is Phase-3 agnostic.
"""

from __future__ import annotations

import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from datasets.zhang_with_lu_dgp import generate_choice_learn_market_shocks_dgp
from lu.choice_learn.cl_posterior import ChoiceLearnPosteriorConfig
from lu.choice_learn.cl_shrinkage import (
    ChoiceLearnShrinkageConfig,
    run_chain,
    summarize_samples,
)
from zhang.featurebased import BaseFeatureBasedDeepHalo


# -----------------------------
# Helpers (shared)
# -----------------------------


def build_items_tensor(xj: np.ndarray) -> tf.Tensor:
    """
    Convert product features xj into a single-items tensor for the choice model.

    Expected:
      xj: (J, d_x) or (J,)

    Returns:
      items_one: (1, J, d_x)
    """
    x = np.asarray(xj, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"xj must be 1D or 2D. Got shape {x.shape}.")

    items = tf.convert_to_tensor(x, dtype=tf.float32)
    return items[None, :, :]


def build_choice_index_tensor(qj_base: np.ndarray) -> tf.Tensor:
    """
    Convert base-market inside-good counts into a vector of chosen item indices.

    Expected:
      qj_base: (J,) integer counts

    Returns:
      choices: (sum_j qj_base[j],) int32 indices in [0, J)
    """
    q = np.asarray(qj_base, dtype=np.int64)
    idx = np.repeat(np.arange(q.shape[0], dtype=np.int64), q)
    return tf.convert_to_tensor(idx, dtype=tf.int32)


def make_training_dataset(
    items_one: tf.Tensor,
    choices: tf.Tensor,
    batch_size: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    """Build a tf.data dataset of (items_batch, choice_index_batch) pairs."""
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
    """
    Convert inside-good utilities into logit probabilities with an outside option.

    Outside utility is normalized to 0.
    """
    u = np.asarray(u, dtype=np.float64)
    m = max(0.0, float(np.max(u)))
    exp_inside = np.exp(u - m)
    exp_out = math.exp(-m)
    denom = exp_out + float(np.sum(exp_inside))
    return exp_inside / denom, float(exp_out / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation for two arrays after flattening."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0.0 else float("nan")


def _to_numpy_results(results: dict[str, object]) -> dict[str, object]:
    """Convert tensor-valued result entries into NumPy-backed output."""
    out: dict[str, object] = {}
    for key, value in results.items():
        if isinstance(value, tf.Tensor):
            out[key] = value.numpy()
        else:
            out[key] = value
    return out


# -----------------------------
# Phase 1: choice model
# -----------------------------


def run_choice_model(
    seed: int,
    num_products: int,
    num_groups: int,
    num_markets: int,
    N_base: int,
    N_shock: int,
    num_features: int,
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
    """
    Phase 1: simulate a market and train a feature-based choice model.

    Returns:
      - dgp: dict of simulated objects
      - delta_hat: (J,) estimated utilities from the trained model
    """
    dgp = generate_choice_learn_market_shocks_dgp(
        seed=seed,
        num_markets=num_markets,
        num_products=num_products,
        num_groups=num_groups,
        N_base=N_base,
        N_shock=N_shock,
        num_features=num_features,
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

    return {"dgp": dgp, "delta_hat": delta_hat}


def print_choice_model_diagnostics(
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
    """Print Phase-1 diagnostics comparing baseline model probabilities to truth."""
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
        "utility recovery (centered): "
        f"corr={corr(delta_hat - delta_hat.mean(), delta_true - delta_true.mean()):.4f} "
        f"| rmse={rmse(delta_hat - delta_hat.mean(), delta_true - delta_true.mean()):.4f}"
    )


# -----------------------------
# Phase 2: market shock estimator
# -----------------------------


def run_market_shock_estimator(
    delta_cl: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config: ChoiceLearnPosteriorConfig,
    shrinkage_config: ChoiceLearnShrinkageConfig,
    seed: tf.Tensor,
) -> dict[str, object]:
    """
    Phase 2: recover market shocks with the refactored choice-learn shrinkage sampler.

    Inputs must be float64 tensors with static shapes:
      - delta_cl: (T, J)
      - qjt:      (T, J)
      - q0t:      (T,)
    """
    samples = run_chain(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=seed,
    )
    return _to_numpy_results(summarize_samples(samples))


def print_market_shock_diagnostics(
    delta_hat: np.ndarray,
    dgp: dict[str, object],
    res: dict[str, object],
) -> None:
    """Print Phase-2 diagnostics comparing baseline vs posterior-mean vs oracle."""
    qjt = np.asarray(dgp["qjt_shock"], dtype=np.float64)
    q0t = np.asarray(dgp["q0t_shock"], dtype=np.float64)

    delta_true = np.asarray(dgp["delta_true"], dtype=np.float64)
    E_bar_true = np.asarray(dgp["E_bar_true"], dtype=np.float64)
    njt_true = np.asarray(dgp["njt_true"], dtype=np.float64)
    E_true = E_bar_true[:, None] + njt_true

    alpha_hat = float(np.asarray(res["alpha_hat"], dtype=np.float64))
    E_bar_hat = np.asarray(res["E_bar_hat"], dtype=np.float64)
    njt_hat = np.asarray(res["njt_hat"], dtype=np.float64)
    gamma_hat = np.asarray(res["gamma_hat"], dtype=np.float64)
    E_hat = np.asarray(res["E_hat"], dtype=np.float64)

    T = qjt.shape[0]

    nll_base = 0.0
    nll_post = 0.0
    nll_oracle = 0.0

    p_base_j, p_base_0 = probs_with_outside(delta_hat)

    for t in range(T):
        p_post_j, p_post_0 = probs_with_outside(
            alpha_hat * delta_hat + E_bar_hat[t] + njt_hat[t]
        )
        p_orac_j, p_orac_0 = probs_with_outside(
            delta_true + E_bar_true[t] + njt_true[t]
        )

        nll_base -= np.sum(qjt[t] * np.log(p_base_j)) + q0t[t] * math.log(p_base_0)
        nll_post -= np.sum(qjt[t] * np.log(p_post_j)) + q0t[t] * math.log(p_post_0)
        nll_oracle -= np.sum(qjt[t] * np.log(p_orac_j)) + q0t[t] * math.log(p_orac_0)

    print("")
    print("============================================================")
    print("PHASE 2: MARKET SHOCK ESTIMATOR")
    print("============================================================")
    print(f"alpha_hat: {alpha_hat:.4f}")
    print(
        f"gamma_hat share: mean={float(gamma_hat.mean()):.4f} | "
        f"min={float(gamma_hat.min()):.4f} | max={float(gamma_hat.max()):.4f}"
    )
    print("")
    print(
        "E_bar recovery: "
        f"corr={corr(E_bar_hat, E_bar_true):.4f} | "
        f"rmse={rmse(E_bar_hat, E_bar_true):.4f}"
    )
    print(
        "njt recovery:   "
        f"corr={corr(njt_hat, njt_true):.4f} | "
        f"rmse={rmse(njt_hat, njt_true):.4f}"
    )
    print(
        "E recovery:     "
        f"corr={corr(E_hat, E_true):.4f} | "
        f"rmse={rmse(E_hat, E_true):.4f}"
    )
    print("")
    print("Average NLL per market:")
    print(f"  baseline-only:   {nll_base / T:.3f}")
    print(f"  posterior-mean:  {nll_post / T:.3f}")
    print(f"  oracle (truth):  {nll_oracle / T:.3f}")


# -----------------------------
# Main (thin wrapper)
# -----------------------------


def main() -> None:
    # --- DGP + phase-1 model config ---
    seed = 123
    num_products = 15
    num_groups = 5
    num_markets = 10

    N_base = 2_000
    N_shock = 1_000

    num_features = 4
    x_sd = 1.0
    coef_sd = 1.0
    p_g_active = 0.2
    g_sd = None

    sd_E = 0.5
    p_active = 0.25
    sd_u = 0.5

    depth = 5
    width = 64
    heads = 8

    epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    shuffle_buffer = 1_000

    eval_include_outside = True
    eval_against_empirical = True

    # --- shrinkage (phase-2) config ---
    shrink_seed = 0

    posterior_config = ChoiceLearnPosteriorConfig(
        eps=1e-15,
        alpha_mean=1.0,
        alpha_var=1.0,
        E_bar_mean=0.0,
        E_bar_var=1.0,
        T0_sq=0.01,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
    )

    shrinkage_config = ChoiceLearnShrinkageConfig(
        num_results=500,
        num_burnin_steps=0,
        chunk_size=100,
        k_alpha=1.0,
        k_E_bar=1.0,
        k_njt=1.0,
        pilot_length=20,
        target_low=0.3,
        target_high=0.5,
        max_rounds=50,
        factor=1.2,
    )

    chain_seed = tf.constant([shrink_seed, 0], dtype=tf.int32)

    # --- Phase 1 ---
    out1 = run_choice_model(
        seed=seed,
        num_products=num_products,
        num_groups=num_groups,
        num_markets=num_markets,
        N_base=N_base,
        N_shock=N_shock,
        num_features=num_features,
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
    qjt_shock = np.asarray(dgp["qjt_shock"], dtype=np.float64)
    q0t_shock = np.asarray(dgp["q0t_shock"], dtype=np.float64)

    T = int(qjt_shock.shape[0])

    delta_hat_tf = tf.constant(
        np.asarray(delta_hat, dtype=np.float64), dtype=tf.float64
    )
    delta_cl_tf = tf.tile(delta_hat_tf[None, :], [T, 1])

    qjt_tf = tf.constant(qjt_shock, dtype=tf.float64)
    q0t_tf = tf.constant(q0t_shock, dtype=tf.float64)

    res2 = run_market_shock_estimator(
        delta_cl=delta_cl_tf,
        qjt=qjt_tf,
        q0t=q0t_tf,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=chain_seed,
    )

    print_market_shock_diagnostics(
        delta_hat=delta_hat,
        dgp=dgp,
        res=res2,
    )


if __name__ == "__main__":
    main()
