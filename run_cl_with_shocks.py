# run_choice_learn_market_shocks.py
"""
Orchestration (phase 1 baseline only, TF-native):

- Generate updated DGP (fixed set, product-sparse cross-product interactions).
- Train BaseFeatureBasedDeepHalo using a tf.data.Dataset that streams labels and
  tiles a single fixed choice set tensor for each batch.
- Evaluate using probability RMSE against true probabilities (and optionally
  against empirical shares).

Phase 2 additionally runs the ChoiceLearnShrinkageEstimator on shocked counts using
frozen baseline logits from the featurebased model.
"""

from __future__ import annotations

import os

# Reduce TensorFlow logging noise in terminal output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import numpy as np
import tensorflow as tf

from datasets.cl_with_shocks_dgp import (
    generate_choice_learn_market_shocks_dgp,
)
from models.featurebased import BaseFeatureBasedDeepHalo
from market_shock_estimators.choice_learn.choice_learn_shrinkage import (
    ChoiceLearnShrinkageEstimator,
)


# -----------------------------
# Hyperparameters (lowercase)
# -----------------------------

# DGP
seed = 123
num_products = 15
num_groups = 5
num_markets = 10

N_base = 20_000  # baseline phase: single multinomial draw
N_shock = 10_000  # shock phase: per-market multinomial draws (generated but unused in phase 1)

x_sd = 1.0
coef_sd = 1.0

# product-sparse interaction loadings
p_g_active = 0.2
g_sd = None  # if None, uses coef_sd

# shocks (generated but unused in phase 1)
sd_E = 0.5
p_active = 0.25
sd_u = 0.5

# Baseline model (Zhang)
depth = 10
width = 100
heads = 8

# Training
epochs = 20
batch_size = 64
learning_rate = 1e-3
shuffle_buffer = 10_000

# Evaluation
# Shrinkage (Phase 2)
# MCMC
shrink_seed = 0
shrink_n_iter = 100

# Tuning
shrink_pilot_length = 20
shrink_max_rounds = 50
shrink_target_low = 0.3
shrink_target_high = 0.5
shrink_factor_rw = 1.2
shrink_factor_tmh = 1.2
shrink_ridge = 1e-6

eval_include_outside = True
eval_against_empirical = True


# -----------------------------
# Helpers
# -----------------------------


def build_items_tensor(xj: np.ndarray) -> tf.Tensor:
    """
    Build the single fixed choice set tensor of shape (1, J, dx_items).

    dx_items=2 with [x, x^2] per item.
    """
    x = tf.convert_to_tensor(np.asarray(xj, dtype=np.float32))  # (J,)
    items = tf.stack([x, tf.square(x)], axis=-1)  # (J, 2)
    return items[None, :, :]  # (1, J, 2)


def build_choice_index_tensor(qj_base: np.ndarray) -> tf.Tensor:
    """
    Expand inside-good counts into individual choice indices (0..J-1).

    Returns choices tensor shape (N_inside,) int32.
    """
    q = np.asarray(qj_base, dtype=np.int64)
    idx = np.repeat(np.arange(q.shape[0], dtype=np.int64), q)
    return tf.convert_to_tensor(idx, dtype=tf.int32)


def make_training_dataset(
    items_one: tf.Tensor,
    choices: tf.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Dataset yields (x_batch, y_batch) where:
      y_batch: (B,)
      x_batch: (B, J, dx) obtained by tiling items_one (1, J, dx).
    """
    ds = tf.data.Dataset.from_tensor_slices(choices)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    def to_xy(y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        b = tf.shape(y_batch)[0]
        x_batch = tf.tile(items_one, multiples=[b, 1, 1])  # (B, J, dx)
        return x_batch, y_batch

    ds = ds.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def probs_with_outside_from_logits(delta_hat: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Outside option utility fixed at 0.
    Returns (p_inside (J,), p0).
    """
    u = np.asarray(delta_hat, dtype=np.float64)
    m = max(0.0, float(np.max(u)))
    exp_inside = np.exp(u - m)
    exp_out = math.exp(-m)
    denom = exp_out + float(np.sum(exp_inside))
    return exp_inside / denom, float(exp_out / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


# -----------------------------
# Main
# -----------------------------


def main() -> None:
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
    q0_base = int(dgp["q0_base"])
    p_base = np.asarray(dgp["p_base"], dtype=np.float64)
    p0_base = float(dgp["p0_base"])

    items_one = build_items_tensor(xj)  # (1, J, 2)
    choices = build_choice_index_tensor(qj_base)  # (N_inside,)

    n_inside = int(choices.shape[0])
    if n_inside == 0:
        raise ValueError(
            "No inside purchases in baseline draw; increase N_base or adjust coefficients."
        )

    ds_train = make_training_dataset(
        items_one=items_one,
        choices=choices,
        batch_size=batch_size,
        shuffle=True,
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
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.fit(ds_train, epochs=epochs, verbose=1)

    # Predict logits for the fixed set once
    delta_hat = model(items_one, training=False).numpy()[0]  # (J,)

    # Probability RMSE against truth
    p_hat, p0_hat = probs_with_outside_from_logits(delta_hat)

    rmse_inside_true = rmse(p_hat, p_base)
    if eval_include_outside:
        rmse_all_true = rmse(np.r_[p0_hat, p_hat], np.r_[p0_base, p_base])
    else:
        rmse_all_true = float("nan")

    # Optional: probability RMSE against empirical shares from the single multinomial draw
    if eval_against_empirical:
        s_hat = np.asarray(qj_base, dtype=np.float64) / float(N_base)
        s0_hat = float(q0_base) / float(N_base)

        rmse_inside_share = rmse(p_hat, s_hat)
        if eval_include_outside:
            rmse_all_share = rmse(np.r_[p0_hat, p_hat], np.r_[s0_hat, s_hat])
        else:
            rmse_all_share = float("nan")
    else:
        rmse_inside_share = float("nan")
        rmse_all_share = float("nan")

    print("baseline-only evaluation (tf.data)")
    print(f"num_products: {num_products}")
    print(f"N_base: {N_base} | N_inside: {n_inside} | N_outside: {q0_base}")
    print(f"rmse_prob_inside_vs_true: {rmse_inside_true:.6f}")
    if eval_include_outside:
        print(f"rmse_prob_all_vs_true:    {rmse_all_true:.6f}")
    if eval_against_empirical:
        print(f"rmse_prob_inside_vs_share:{rmse_inside_share:.6f}")
        if eval_include_outside:
            print(f"rmse_prob_all_vs_share:   {rmse_all_share:.6f}")

    # -----------------------------
    # Phase 2: run shrinkage estimator (wiring only)
    # -----------------------------

    qjt_shock = np.asarray(dgp["qjt_shock"], dtype=np.float64)  # (T, J)
    q0t_shock = np.asarray(dgp["q0t_shock"], dtype=np.float64)  # (T,)

    T = int(qjt_shock.shape[0])
    J = int(qjt_shock.shape[1])

    # Replicate frozen baseline logits across markets to match (T, J)
    delta_cl = np.repeat(np.asarray(delta_hat, dtype=np.float64)[None, :], T, axis=0)

    shrink = ChoiceLearnShrinkageEstimator(
        delta_cl=delta_cl,
        qjt=qjt_shock,
        q0t=q0t_shock,
        seed=shrink_seed,
    )

    shrink.fit(
        n_iter=shrink_n_iter,
        pilot_length=shrink_pilot_length,
        ridge=shrink_ridge,
        target_low=shrink_target_low,
        target_high=shrink_target_high,
        max_rounds=shrink_max_rounds,
        factor_rw=shrink_factor_rw,
        factor_tmh=shrink_factor_tmh,
    )

    res = shrink.get_results()

    # -----------------------------
    # Phase 3: results + diagnosis
    # -----------------------------

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 0.0:
            return float("nan")
        return float(np.sum(a * b) / denom)

    def probs_with_outside_from_utility(u: np.ndarray) -> tuple[np.ndarray, float]:
        u = np.asarray(u, dtype=np.float64)
        m = max(0.0, float(np.max(u)))
        exp_inside = np.exp(u - m)
        exp_out = math.exp(-m)
        denom = exp_out + float(np.sum(exp_inside))
        return exp_inside / denom, float(exp_out / denom)

    def nll_from_counts(qj: np.ndarray, q0: float, pj: np.ndarray, p0: float) -> float:
        # Multinomial constant omitted (does not affect model comparisons).
        eps = 1e-15
        pj = np.clip(np.asarray(pj, dtype=np.float64), eps, 1.0)
        p0 = float(np.clip(p0, eps, 1.0))
        return float(
            -(
                np.sum(np.asarray(qj, dtype=np.float64) * np.log(pj))
                + float(q0) * math.log(p0)
            )
        )

    def center_market(
        E_bar: np.ndarray, njt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identification convention for reporting:
        Move the within-market mean of n[t,:] into E_bar[t], so n[t,:] is mean-zero across products.
        This preserves E_bar[t] + n[t,j] exactly.
        """
        njt = np.asarray(njt, dtype=np.float64)
        E_bar = np.asarray(E_bar, dtype=np.float64)
        m = njt.mean(axis=1)  # (T,)
        E_c = E_bar + m
        n_c = njt - m[:, None]
        return E_c, n_c

    # ---- Pull truth (for scoring) ----
    delta_true = np.asarray(dgp["delta_true"], dtype=np.float64)  # (J,)
    E_bar_true = np.asarray(dgp["E_bar_true"], dtype=np.float64)  # (T,)
    njt_true = np.asarray(dgp["njt_true"], dtype=np.float64)  # (T,J)

    # ---- Pull posterior means (estimates) ----
    alpha_hat = float(res["alpha_hat"])
    E_bar_hat = np.asarray(res["E_bar_hat"], dtype=np.float64)  # (T,)
    njt_hat = np.asarray(res["njt_hat"], dtype=np.float64)  # (T,J)
    gamma_hat = np.asarray(res["gamma_hat"], dtype=np.float64)  # (T,J)
    phi_hat = np.asarray(res["phi_hat"], dtype=np.float64)  # (T,)
    n_saved = int(res["n_saved"])

    # ---- Phase 1 (baseline) diagnostics: utility recovery up to additive constant ----
    delta_hat_c = np.asarray(delta_hat, dtype=np.float64) - float(np.mean(delta_hat))
    delta_true_c = delta_true - float(np.mean(delta_true))
    baseline_logit_corr = corr(delta_hat_c, delta_true_c)
    baseline_logit_rmse = rmse(delta_hat_c, delta_true_c)

    # ---- Phase 2 predictive diagnostics ----
    # Baseline-only: uses frozen delta_cl (no online correction).
    # Shock-adjusted: uses posterior mean utilities alpha*delta_cl + E_bar + n.
    # Oracle: uses ground-truth utilities (reference upper bound in simulation).
    nll_base = 0.0
    nll_post = 0.0
    nll_oracle = 0.0

    rmse_share_inside_base = 0.0
    rmse_share_inside_post = 0.0
    rmse_share_inside_oracle = 0.0

    rmse_prob_all_post_vs_oracle = 0.0

    for t in range(T):
        N_t = float(q0t_shock[t] + float(np.sum(qjt_shock[t])))

        # Empirical phase-2 shares
        s_emp_j = np.asarray(qjt_shock[t], dtype=np.float64) / N_t

        # Baseline-only prediction
        pb_j, pb_0 = probs_with_outside_from_utility(delta_cl[t])

        # Posterior-mean corrected prediction
        u_post_t = alpha_hat * delta_cl[t] + E_bar_hat[t] + njt_hat[t]
        pp_j, pp_0 = probs_with_outside_from_utility(u_post_t)

        # Oracle prediction (truth utilities)
        u_true_t = delta_true + E_bar_true[t] + njt_true[t]
        po_j, po_0 = probs_with_outside_from_utility(u_true_t)

        # NLL on observed counts
        nll_base += nll_from_counts(qjt_shock[t], q0t_shock[t], pb_j, pb_0)
        nll_post += nll_from_counts(qjt_shock[t], q0t_shock[t], pp_j, pp_0)
        nll_oracle += nll_from_counts(qjt_shock[t], q0t_shock[t], po_j, po_0)

        # RMSE vs empirical inside shares
        rmse_share_inside_base += rmse(pb_j, s_emp_j)
        rmse_share_inside_post += rmse(pp_j, s_emp_j)
        rmse_share_inside_oracle += rmse(po_j, s_emp_j)

        # Probability distance to oracle probabilities (outside+inside)
        if eval_include_outside:
            rmse_prob_all_post_vs_oracle += rmse(np.r_[pp_0, pp_j], np.r_[po_0, po_j])

    nll_base /= float(T)
    nll_post /= float(T)
    nll_oracle /= float(T)

    rmse_share_inside_base /= float(T)
    rmse_share_inside_post /= float(T)
    rmse_share_inside_oracle /= float(T)

    if eval_include_outside:
        rmse_prob_all_post_vs_oracle /= float(T)
    else:
        rmse_prob_all_post_vs_oracle = float("nan")

    # ---- Phase 2 recovery diagnostics ----
    # Primary recovery target: combined shock surface s[t,j] = E_bar[t] + n[t,j]
    s_true = E_bar_true[:, None] + njt_true
    s_hat = E_bar_hat[:, None] + njt_hat
    s_rmse = rmse(s_hat, s_true)
    s_corr = corr(s_hat, s_true)

    # Decomposed components reported under a centering convention
    E_true_c, n_true_c = center_market(E_bar_true, njt_true)
    E_hat_c, n_hat_c = center_market(E_bar_hat, njt_hat)
    E_rmse = rmse(E_hat_c, E_true_c)
    E_corr = corr(E_hat_c, E_true_c)
    n_rmse = rmse(n_hat_c, n_true_c)
    n_corr = corr(n_hat_c, n_true_c)

    # ---- Sparsity / inclusion diagnostics ----
    # Ground truth "active" defined by non-trivial injected product shock.
    active_true = np.abs(njt_true) > 1e-12
    active_pred = gamma_hat > 0.5

    tp = int(np.sum(active_true & active_pred))
    fp = int(np.sum((~active_true) & active_pred))
    fn = int(np.sum(active_true & (~active_pred)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )

    # Separation of posterior inclusion probabilities
    gamma_active_mean = (
        float(np.mean(gamma_hat[active_true])) if np.any(active_true) else float("nan")
    )
    gamma_inactive_mean = (
        float(np.mean(gamma_hat[~active_true]))
        if np.any(~active_true)
        else float("nan")
    )

    # ---- Ranked summaries for interpretability ----
    top_k = 5
    idx_top_E = np.argsort(np.abs(E_hat_c))[::-1][
        :top_k
    ]  # markets with largest centered market-wide shift

    mean_abs_n_by_j = np.mean(
        np.abs(n_hat_c), axis=0
    )  # product-level average abs deviation
    idx_top_prod = np.argsort(mean_abs_n_by_j)[::-1][: min(top_k, J)]
    active_rate_hat = np.mean((gamma_hat > 0.5), axis=0)

    # ---- Print report ----
    print("")
    print("============================================================")
    print("RESULTS SUMMARY")
    print("============================================================")

    print("")
    print("Phase 1 (baseline logits): recovery of centered utilities")
    print(f"  corr(delta_hat, delta_true): {baseline_logit_corr:.4f}")
    print(f"  rmse(delta_hat, delta_true): {baseline_logit_rmse:.4f}")

    print("")
    print("Phase 2 (posterior means): key estimated quantities")
    print(f"  n_saved (MCMC draws used for posterior means): {n_saved}")
    print(f"  alpha_hat (global rescaling of baseline logits): {alpha_hat:.4f}")
    print(
        f"  phi_hat (market-level sparsity rate): mean={float(np.mean(phi_hat)):.4f} | min={float(np.min(phi_hat)):.4f} | max={float(np.max(phi_hat)):.4f}"
    )

    print("")
    print("Phase 2 predictive performance on observed phase-2 counts")
    print("  Average per-market NLL (lower is better):")
    print(f"    baseline-only (frozen logits):           {nll_base:.3f}")
    print(f"    online-corrected (posterior mean):       {nll_post:.3f}")
    print(f"    oracle (truth utilities; reference):     {nll_oracle:.3f}")
    print("  Average RMSE vs empirical inside shares (lower is better):")
    print(f"    baseline-only:                           {rmse_share_inside_base:.4f}")
    print(f"    online-corrected:                        {rmse_share_inside_post:.4f}")
    print(
        f"    oracle (reference):                      {rmse_share_inside_oracle:.4f}"
    )
    if eval_include_outside:
        print(
            f"  RMSE vs oracle probabilities (outside+inside): {rmse_prob_all_post_vs_oracle:.4f}"
        )

    print("")
    print("Phase 2 recovery vs ground truth (simulation scoring)")
    print("  Primary target: combined shock surface s[t,j] = E_bar[t] + n[t,j]")
    print(f"    rmse(s_hat, s_true): {s_rmse:.4f} | corr: {s_corr:.4f}")
    print("  Decomposition reported under market-centering convention:")
    print(
        "    (E_bar absorbs mean of n within each market; n is mean-zero across products)"
    )
    print(f"    market-wide component E_bar: rmse={E_rmse:.4f} | corr={E_corr:.4f}")
    print(f"    product deviations n:        rmse={n_rmse:.4f} | corr={n_corr:.4f}")

    print("")
    print("Sparsity / inclusion diagnostics (gamma_hat thresholded at 0.5)")
    print(f"  precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f}")
    print(f"  mean gamma_hat on true-active entries:   {gamma_active_mean:.4f}")
    print(f"  mean gamma_hat on true-inactive entries: {gamma_inactive_mean:.4f}")

    print("")
    print("Largest estimated market-wide shifts (after centering)")
    for r, t in enumerate(idx_top_E, start=1):
        print(
            f"  {r}) market {int(t)}: E_hat_centered={E_hat_c[t]:+.4f} | E_true_centered={E_true_c[t]:+.4f}"
        )

    print("")
    print(
        "Most affected products overall (mean abs centered product deviation across markets)"
    )
    for r, j in enumerate(idx_top_prod, start=1):
        print(
            f"  {r}) product {int(j)}: mean|n_hat_centered|={mean_abs_n_by_j[j]:.4f} "
            f"| Pr(gamma>0.5) across markets={active_rate_hat[j]:.2f}"
        )

    print("============================================================")


if __name__ == "__main__":
    main()
