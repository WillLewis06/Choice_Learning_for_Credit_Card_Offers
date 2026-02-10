# ching/stockpiling_evaluate.py
#
# Simple evaluation utilities for the stockpiling model.
#
# What it reports:
# 1) Parameter recovery (true vs fitted): RMSE/MAE and means.
# 2) Predictive fit on observed actions: NLL per decision and Brier/RMSE on predicted buy probabilities.
#
# Extra (low-baggage) additions:
# - Sample size (M, N, T, n_obs)
# - Constant-rate baseline metrics (using empirical buy rate)
# - Buy rate by price state (empirical vs predicted) + RMSE across states
# - Optional formatting helper for clean milestone prints

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from ching.stockpiling_posterior import (
    action_likelihood_by_inventory,
    bayes_update_belief_mn,
    build_inventory_maps,
    initial_inventory_belief,
    select_pi_by_state,
    solve_ccp_buy,
    transition_belief,
)


# =============================================================================
# Parameter recovery
# =============================================================================


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Compare true vs fitted constrained parameters.

    For each key in theta_hat (beta, alpha, v, fc, lambda_c, u_scale):
      rmse, mae, mean_true, mean_hat

    If theta_true lacks u_scale, it is treated as ones_like(theta_hat["u_scale"]).
    """
    out: dict[str, dict[str, float]] = {}

    def _rmse_mae(t: np.ndarray, h: np.ndarray) -> tuple[float, float]:
        d = h - t
        rmse = float(np.sqrt(np.mean(d * d)))
        mae = float(np.mean(np.abs(d)))
        return rmse, mae

    for k, hat in theta_hat.items():
        if k == "u_scale" and k not in theta_true:
            true = np.ones_like(np.asarray(hat, dtype=np.float64), dtype=np.float64)
        else:
            if k not in theta_true:
                continue
            true = np.asarray(theta_true[k], dtype=np.float64)

        hat_arr = np.asarray(hat, dtype=np.float64)
        rmse, mae = _rmse_mae(true, hat_arr)

        out[k] = {
            "rmse": rmse,
            "mae": mae,
            "mean_true": float(np.mean(true)),
            "mean_hat": float(np.mean(hat_arr)),
        }

    return out


# =============================================================================
# Predictive fit (NLL + Brier/RMSE)
# =============================================================================


def predictive_metrics(
    *,
    a_imt: np.ndarray,  # (M,N,T) 0/1
    p_state_mt: np.ndarray,  # (M,T) state indices in {0,...,S-1}
    u_m: np.ndarray,  # (M,)
    price_vals: np.ndarray,  # (S,)
    P_price: np.ndarray,  # (S,S)
    I_max: int,
    pi_I0: np.ndarray,  # (I_max+1,)
    waste_cost: float,
    eps: float,
    tol: float,
    max_iter: int,
    theta: dict[
        str, np.ndarray
    ],  # constrained: beta,alpha,v,fc,lambda_c (M,N), u_scale (M,)
) -> dict[str, Any]:
    """
    Compute predictive fit metrics implied by theta.

    Returns a dict containing:
      - shape: {M, N, T, n_obs}
      - fitted metrics: nll_per_obs, brier, rmse_prob, buy_rate_emp, buy_rate_pred
      - by-state: buy_rate_by_state_emp/pred, rmse_buy_rate_by_state
      - baseline: constant-rate baseline using empirical buy rate (p0)
    """
    # --- Convert fixed inputs once ---
    a_tf = tf.convert_to_tensor(a_imt)
    s_tf = tf.convert_to_tensor(p_state_mt)

    u_tf = tf.convert_to_tensor(np.asarray(u_m, dtype=np.float64), dtype=tf.float64)
    price_tf = tf.convert_to_tensor(
        np.asarray(price_vals, dtype=np.float64), dtype=tf.float64
    )
    P_tf = tf.convert_to_tensor(np.asarray(P_price, dtype=np.float64), dtype=tf.float64)

    I_max_tf = tf.convert_to_tensor(int(I_max), dtype=tf.int32)
    pi_tf = tf.convert_to_tensor(np.asarray(pi_I0, dtype=np.float64), dtype=tf.float64)

    waste_tf = tf.convert_to_tensor(float(waste_cost), dtype=tf.float64)
    eps_tf = tf.convert_to_tensor(float(eps), dtype=tf.float64)
    tol_tf = tf.convert_to_tensor(float(tol), dtype=tf.float64)
    max_iter_tf = tf.convert_to_tensor(int(max_iter), dtype=tf.int32)

    # --- Convert theta dict to tf.float64, and ensure u_scale exists ---
    theta_np = dict(theta)
    if "u_scale" not in theta_np:
        theta_np["u_scale"] = np.ones_like(
            np.asarray(u_m, dtype=np.float64), dtype=np.float64
        )

    theta_tf: dict[str, tf.Tensor] = {
        k: tf.convert_to_tensor(np.asarray(v, dtype=np.float64), dtype=tf.float64)
        for k, v in theta_np.items()
    }

    # --- Precompute maps + solve DP once ---
    maps = build_inventory_maps(I_max_tf)  # (D_down, D_up, stockout_mask, at_cap_mask)
    D_down, D_up, _, _ = maps

    ccp_buy = solve_ccp_buy(
        u_m=u_tf,
        price_vals=price_tf,
        P_price=P_tf,
        theta=theta_tf,
        waste_cost=waste_tf,
        tol=tol_tf,
        max_iter=max_iter_tf,
        maps=maps,
    )

    # --- Forward filter with streaming accumulators ---
    M = tf.shape(a_tf)[0]
    N = tf.shape(a_tf)[1]
    T = tf.shape(a_tf)[2]
    S = tf.shape(price_tf)[0]  # number of price states

    b = initial_inventory_belief(pi_tf, M, N, eps_tf)

    ll = tf.constant(0.0, dtype=tf.float64)
    sum_p = tf.constant(0.0, dtype=tf.float64)
    sum_p2 = tf.constant(0.0, dtype=tf.float64)
    sum_pa = tf.constant(0.0, dtype=tf.float64)
    sum_a = tf.constant(0.0, dtype=tf.float64)

    # By-state sums (over all m,n,t):
    sum_p_state = tf.zeros([S], dtype=tf.float64)
    sum_a_state = tf.zeros([S], dtype=tf.float64)
    cnt_state = tf.zeros([S], dtype=tf.float64)  # counts in "observations" (includes N)

    def cond(t: tf.Tensor, *_):
        return t < T

    def body(
        t: tf.Tensor,
        b_curr: tf.Tensor,
        ll_curr: tf.Tensor,
        sum_p_curr: tf.Tensor,
        sum_p2_curr: tf.Tensor,
        sum_pa_curr: tf.Tensor,
        sum_a_curr: tf.Tensor,
        sum_p_state_curr: tf.Tensor,
        sum_a_state_curr: tf.Tensor,
        cnt_state_curr: tf.Tensor,
    ):
        s_mt = s_tf[:, t]  # (M,)
        a_mn = a_tf[:, :, t]  # (M,N)

        pi_mnI = select_pi_by_state(ccp_buy, s_mt)  # (M,N,I)
        p_buy = tf.reduce_sum(b_curr * pi_mnI, axis=2)  # (M,N)
        a_f = tf.cast(a_mn, tf.float64)

        # Global streaming sums.
        sum_p_curr += tf.reduce_sum(p_buy)
        sum_p2_curr += tf.reduce_sum(tf.square(p_buy))
        sum_pa_curr += tf.reduce_sum(p_buy * a_f)
        sum_a_curr += tf.reduce_sum(a_f)

        # By-state sums (Option A): one-hot aggregation over markets at time t.
        # s_mt is market-level; we treat each market-time as contributing N observations.
        p_sum_m = tf.reduce_sum(p_buy, axis=1)  # (M,)
        a_sum_m = tf.reduce_sum(a_f, axis=1)  # (M,)
        n_f = tf.cast(N, tf.float64)

        one_hot = tf.one_hot(s_mt, depth=S, dtype=tf.float64)  # (M,S)
        one_hot_T = tf.transpose(one_hot)  # (S,M)

        sum_p_state_curr += tf.linalg.matvec(one_hot_T, p_sum_m)  # (S,)
        sum_a_state_curr += tf.linalg.matvec(one_hot_T, a_sum_m)  # (S,)
        cnt_state_curr += tf.reduce_sum(one_hot, axis=0) * n_f  # (S,)

        # Likelihood update.
        lik_I = action_likelihood_by_inventory(pi_mnI, a_mn)  # (M,N,I)
        w_post, ll_step_mn = bayes_update_belief_mn(b_curr, lik_I, eps_tf)
        ll_step = tf.reduce_sum(ll_step_mn)

        b_next = transition_belief(
            w_post, a_mn, theta_tf["lambda_c"], D_down, D_up, eps_tf
        )

        return (
            t + 1,
            b_next,
            ll_curr + ll_step,
            sum_p_curr,
            sum_p2_curr,
            sum_pa_curr,
            sum_a_curr,
            sum_p_state_curr,
            sum_a_state_curr,
            cnt_state_curr,
        )

    t0 = tf.constant(0, dtype=tf.int32)
    _, _, ll, sum_p, sum_p2, sum_pa, sum_a, sum_p_state, sum_a_state, cnt_state = (
        tf.while_loop(
            cond,
            body,
            loop_vars=[
                t0,
                b,
                ll,
                sum_p,
                sum_p2,
                sum_pa,
                sum_a,
                sum_p_state,
                sum_a_state,
                cnt_state,
            ],
            shape_invariants=[
                t0.get_shape(),
                tf.TensorShape([None, None, None]),
                ll.get_shape(),
                ll.get_shape(),
                ll.get_shape(),
                ll.get_shape(),
                ll.get_shape(),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
            ],
        )
    )

    count = tf.cast(
        tf.cast(M, tf.int64) * tf.cast(N, tf.int64) * tf.cast(T, tf.int64),
        tf.float64,
    )

    # Core fitted metrics.
    nll_per_obs = -ll / tf.maximum(count, eps_tf)

    brier = (sum_p2 - 2.0 * sum_pa + sum_a) / tf.maximum(count, eps_tf)  # mean((p-a)^2)
    rmse_prob = tf.sqrt(tf.maximum(brier, 0.0))

    buy_rate_emp = sum_a / tf.maximum(count, eps_tf)
    buy_rate_pred = sum_p / tf.maximum(count, eps_tf)

    # By-state buy rates and RMSE across states.
    emp_state = tf.where(
        cnt_state > 0.0, sum_a_state / cnt_state, tf.constant(np.nan, tf.float64)
    )
    pred_state = tf.where(
        cnt_state > 0.0, sum_p_state / cnt_state, tf.constant(np.nan, tf.float64)
    )

    mask_valid = tf.cast(
        tf.math.is_finite(emp_state) & tf.math.is_finite(pred_state), tf.float64
    )
    diffs = (pred_state - emp_state) * mask_valid
    denom = tf.maximum(tf.reduce_sum(mask_valid), 1.0)
    rmse_by_state = tf.sqrt(tf.reduce_sum(tf.square(diffs)) / denom)

    # Constant-rate baseline using empirical buy rate p0.
    p0 = tf.clip_by_value(buy_rate_emp, eps_tf, 1.0 - eps_tf)
    baseline_nll = -(p0 * tf.math.log(p0) + (1.0 - p0) * tf.math.log(1.0 - p0))
    baseline_brier = p0 * (1.0 - p0)
    baseline_rmse = tf.sqrt(baseline_brier)

    # Materialize to Python/NumPy.
    M_i = int(M.numpy())
    N_i = int(N.numpy())
    T_i = int(T.numpy())

    emp_state_np = emp_state.numpy()
    pred_state_np = pred_state.numpy()
    by_state_emp = {
        int(i): float(emp_state_np[i])
        for i in range(emp_state_np.shape[0])
        if np.isfinite(emp_state_np[i])
    }
    by_state_pred = {
        int(i): float(pred_state_np[i])
        for i in range(pred_state_np.shape[0])
        if np.isfinite(pred_state_np[i])
    }

    return {
        "shape": {"M": M_i, "N": N_i, "T": T_i, "n_obs": int(M_i * N_i * T_i)},
        "nll_per_obs": float(nll_per_obs.numpy()),
        "brier": float(brier.numpy()),
        "rmse_prob": float(rmse_prob.numpy()),
        "buy_rate_emp": float(buy_rate_emp.numpy()),
        "buy_rate_pred": float(buy_rate_pred.numpy()),
        "buy_rate_by_state_emp": by_state_emp,
        "buy_rate_by_state_pred": by_state_pred,
        "rmse_buy_rate_by_state": float(rmse_by_state.numpy()),
        "baseline": {
            "p0": float(p0.numpy()),
            "nll_per_obs": float(baseline_nll.numpy()),
            "brier": float(baseline_brier.numpy()),
            "rmse_prob": float(baseline_rmse.numpy()),
        },
    }


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_stockpiling(
    *,
    a_imt: np.ndarray,
    p_state_mt: np.ndarray,
    u_m: np.ndarray,
    price_vals: np.ndarray,
    P_price: np.ndarray,
    I_max: int,
    pi_I0: np.ndarray,
    waste_cost: float,
    eps: float,
    tol: float,
    max_iter: int,
    theta_hat: dict[str, np.ndarray],
    theta_true: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Any]:
    """
    Simple evaluation wrapper.

    Returns:
      {
        "fit": {...},
        "oracle": {...},   # only if theta_true provided
        "param": {...},    # only if theta_true provided
      }
    """
    out: dict[str, Any] = {}

    out["fit"] = predictive_metrics(
        a_imt=a_imt,
        p_state_mt=p_state_mt,
        u_m=u_m,
        price_vals=price_vals,
        P_price=P_price,
        I_max=I_max,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        theta=theta_hat,
    )

    if theta_true is not None:
        out["param"] = parameter_metrics(theta_true, theta_hat)

        theta_oracle = dict(theta_true)
        if "u_scale" in theta_hat and "u_scale" not in theta_oracle:
            theta_oracle["u_scale"] = np.ones_like(
                np.asarray(theta_hat["u_scale"]), dtype=np.float64
            )

        out["oracle"] = predictive_metrics(
            a_imt=a_imt,
            p_state_mt=p_state_mt,
            u_m=u_m,
            price_vals=price_vals,
            P_price=P_price,
            I_max=I_max,
            pi_I0=pi_I0,
            waste_cost=waste_cost,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            theta=theta_oracle,
        )

    return out


# =============================================================================
# Formatting helper (optional)
# =============================================================================


def format_evaluation_summary(
    eval_out: dict[str, Any],
    *,
    param_order: Optional[list[str]] = None,
) -> str:
    fit = eval_out["fit"]
    oracle = eval_out.get("oracle")
    params = eval_out.get("param")

    shp = fit.get("shape", {})
    M = shp.get("M")
    N = shp.get("N")
    T = shp.get("T")
    n_obs = shp.get("n_obs")

    base = fit["baseline"]

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    # Header
    if M is not None and N is not None and T is not None:
        lines.append(f"data: M={M} N={N} T={T} | n_obs={n_obs}")
    lines.append("")

    # Main metrics table
    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
        f"{'brier':>10} "
        f"{'rmse':>10} "
        f"{'buy_emp':>8} "
        f"{'buy_pred':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(tag: str, d: dict[str, Any]) -> str:
        return (
            f"{tag:<10}"
            f"{f6(d['nll_per_obs'])} "
            f"{f6(d['brier'])} "
            f"{f6(d['rmse_prob'])} "
            f"{f4(d['buy_rate_emp'])} "
            f"{f4(d['buy_rate_pred'])}"
        )

    # baseline has no buy_emp/buy_pred in dict; print p0 instead.
    baseline_row = (
        f"{'baseline':<10}"
        f"{f6(base['nll_per_obs'])} "
        f"{f6(base['brier'])} "
        f"{f6(base['rmse_prob'])} "
        f"{f4(base['p0'])} "
        f"{'':>8}"
    )
    lines.append(baseline_row)
    lines.append(row("fitted", fit))
    if oracle is not None:
        lines.append(row("oracle", oracle))

    # Deltas
    lines.append("")
    nll_gain = base["nll_per_obs"] - fit["nll_per_obs"]
    brier_gain = base["brier"] - fit["brier"]
    rmse_gain = base["rmse_prob"] - fit["rmse_prob"]
    lines.append(
        f"gain vs baseline: Δnll={nll_gain:.6f} | Δbrier={brier_gain:.6f} | Δrmse={rmse_gain:.6f}"
    )
    if oracle is not None:
        lines.append(
            f"fitted - oracle: Δnll={(fit['nll_per_obs'] - oracle['nll_per_obs']):.6f} | "
            f"Δbrier={(fit['brier'] - oracle['brier']):.6f} | "
            f"Δrmse={(fit['rmse_prob'] - oracle['rmse_prob']):.6f}"
        )

    # By-state table
    emp_s = fit.get("buy_rate_by_state_emp", {})
    pred_s = fit.get("buy_rate_by_state_pred", {})
    if emp_s:
        lines.append("")
        lines.append("buy rate by price state")
        st_header = f"{'state':<8}{'emp':>10} {'pred':>10} {'diff':>10}"
        lines.append(st_header)
        lines.append("-" * len(st_header))
        for s in sorted(emp_s.keys()):
            emp = emp_s[s]
            pred = pred_s.get(s, float("nan"))
            diff = pred - emp
            lines.append(f"{str(s):<8}{f6(emp)} {f6(pred)} {f6(diff)}")
        lines.append(f"rmse across states: {fit['rmse_buy_rate_by_state']:.6f}")

    # Parameter recovery table
    if params is not None and isinstance(params, dict) and params:
        if param_order is None:
            param_order = ["beta", "alpha", "v", "fc", "lambda_c", "u_scale"]

        present = [k for k in param_order if k in params]
        worst = sorted(present, key=lambda k: params[k]["rmse"], reverse=True)

        lines.append("")
        lines.append("parameter recovery (sorted by rmse)")
        p_header = (
            f"{'param':<10}"
            f"{'rmse':>10} "
            f"{'mae':>10} "
            f"{'mean_true':>10} "
            f"{'mean_hat':>10} "
            f"{'bias':>10}"
        )
        lines.append(p_header)
        lines.append("-" * len(p_header))
        for k in worst:
            pk = params[k]
            bias = pk["mean_hat"] - pk["mean_true"]
            lines.append(
                f"{k:<10}"
                f"{f6(pk['rmse'])} "
                f"{f6(pk['mae'])} "
                f"{f6(pk['mean_true'])} "
                f"{f6(pk['mean_hat'])} "
                f"{f6(bias)}"
            )

    return "\n".join(lines)
