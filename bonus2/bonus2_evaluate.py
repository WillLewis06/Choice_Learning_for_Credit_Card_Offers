"""
bonus2/bonus2_evaluate.py

Evaluation utilities for Bonus Q2.

Scope:
- Pure NumPy predictive metrics from realized choices and predicted probabilities.
- Baseline is the delta-only model (outside utility 0, inside utility = delta_mj).

Non-goals:
- No TensorFlow calls.
- No routing/fallback logic to "compute probabilities if missing".
- No input validation (assumed handled upstream).

Choice encoding:
  y_mit (M,N,T) with values:
    0 = outside option
    c = j+1 for inside product j in {1..J}

Predicted probabilities:
  p_choice_mntc (M,N,T,J+1) with c=0 outside, c=j+1 inside j.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Predictive metrics
# =============================================================================


def choice_metrics_from_probs(
    y_mit: np.ndarray,
    p_choice_mntc: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """
    Compute predictive metrics from per-observation probabilities.

    Returns:
      {
        "nll": average negative log-likelihood per (m,i,t),
        "acc": argmax accuracy,
        "p_true": mean predicted probability of realized class,
        "out_emp": empirical outside share,
        "out_pred": predicted outside share
      }
    """
    y = np.asarray(y_mit, dtype=np.int64)
    p = np.asarray(p_choice_mntc, dtype=np.float64)

    # Predicted probability assigned to the realized class.
    p_true = np.take_along_axis(p, y[..., None], axis=3)[..., 0]

    # NLL uses clipped probabilities to avoid log(0).
    p_true_clip = np.clip(p_true, eps, 1.0)
    nll = float(-np.mean(np.log(p_true_clip)))

    # Accuracy uses argmax class prediction.
    acc = float(np.mean(np.argmax(p, axis=3) == y))

    # Outside shares.
    out_emp = float(np.mean(y == 0))
    out_pred = float(np.mean(p[..., 0]))

    return {
        "nll": nll,
        "acc": acc,
        "p_true": float(np.mean(p_true)),
        "out_emp": out_emp,
        "out_pred": out_pred,
    }


def choice_metrics_from_market_probs(
    y_mit: np.ndarray,
    p_mjc: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """
    Compute metrics when probabilities are market-only (M,C) and constant in (i,t).

    This is useful for the delta-only baseline where probabilities do not vary by
    consumer or time.
    """
    y = np.asarray(y_mit, dtype=np.int64)
    p = np.asarray(p_mjc, dtype=np.float64)

    # Broadcast market probabilities to observation grid (M,N,T,C).
    p4 = p[:, None, None, :]

    # Realized-class probability and NLL.
    p_true = np.take_along_axis(p4, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)
    nll = float(-np.mean(np.log(p_true_clip)))

    # Market-level argmax prediction (constant within market).
    c_hat_m = np.argmax(p, axis=1)  # (M,)
    acc = float(np.mean(y == c_hat_m[:, None, None]))

    # Outside shares.
    out_emp = float(np.mean(y == 0))
    out_pred = float(np.mean(p[:, 0]))

    return {
        "nll": nll,
        "acc": acc,
        "p_true": float(np.mean(p_true)),
        "out_emp": out_emp,
        "out_pred": out_pred,
    }


def delta_only_baseline_probs(delta_mj: np.ndarray) -> np.ndarray:
    """
    Compute delta-only baseline probabilities per market.

    Utility:
      v_out = 0
      v_in(j) = delta_mj[m,j]

    Returns:
      p_mjc: (M, J+1) with c=0 outside and c=j+1 for inside product j.
    """
    delta = np.asarray(delta_mj, dtype=np.float64)

    # Stabilize exponentials by shifting by max(max(delta), outside=0).
    max_u = np.maximum(0.0, np.max(delta, axis=1))  # (M,)

    exp_in = np.exp(delta - max_u[:, None])  # (M,J)
    exp_out = np.exp(-max_u)  # (M,)
    den = exp_out + np.sum(exp_in, axis=1)  # (M,)

    p_out = (exp_out / den)[:, None]  # (M,1)
    p_in = exp_in / den[:, None]  # (M,J)

    return np.concatenate([p_out, p_in], axis=1)  # (M,J+1)


# =============================================================================
# Parameter recovery
# =============================================================================


def _rmse(true: np.ndarray, hat: np.ndarray) -> float:
    """Root mean squared error."""
    d = np.asarray(hat, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def parameter_recovery_mean_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Mean-level recovery diagnostics for the current Bonus2 parameterization.

    Keys reported:
      beta_intercept_j, beta_habit_j, beta_peer_j, weekend_lift_j, a_m, b_m

    weekend_lift_j is derived as:
      beta_weekend_jw[j,1] - beta_weekend_jw[j,0]
    """
    bt_int_t = np.asarray(theta_true["beta_intercept_j"], dtype=np.float64)
    bt_int_h = np.asarray(theta_hat["beta_intercept_j"], dtype=np.float64)

    bh_t = np.asarray(theta_true["beta_habit_j"], dtype=np.float64)
    bh_h = np.asarray(theta_hat["beta_habit_j"], dtype=np.float64)

    bp_t = np.asarray(theta_true["beta_peer_j"], dtype=np.float64)
    bp_h = np.asarray(theta_hat["beta_peer_j"], dtype=np.float64)

    bw_t = np.asarray(theta_true["beta_weekend_jw"], dtype=np.float64)
    bw_h = np.asarray(theta_hat["beta_weekend_jw"], dtype=np.float64)
    wl_t = bw_t[:, 1] - bw_t[:, 0]
    wl_h = bw_h[:, 1] - bw_h[:, 0]

    am_t = np.asarray(theta_true["a_m"], dtype=np.float64)
    am_h = np.asarray(theta_hat["a_m"], dtype=np.float64)

    bm_t = np.asarray(theta_true["b_m"], dtype=np.float64)
    bm_h = np.asarray(theta_hat["b_m"], dtype=np.float64)

    def pack(t: np.ndarray, h: np.ndarray) -> dict[str, float]:
        mt = float(np.mean(t))
        mh = float(np.mean(h))
        return {"rmse": _rmse(t, h), "mean_true": mt, "mean_hat": mh, "bias": mh - mt}

    return {
        "beta_intercept_j": pack(bt_int_t, bt_int_h),
        "beta_habit_j": pack(bh_t, bh_h),
        "beta_peer_j": pack(bp_t, bp_h),
        "weekend_lift_j": pack(wl_t, wl_h),
        "a_m": pack(am_t, am_h),
        "b_m": pack(bm_t, bm_h),
    }


def parameter_recovery_dispersion_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """
    Dispersion recovery diagnostics.

    Reports dispersion on:
      weekend_lift_j, a_m, b_m
    """
    bw_t = np.asarray(theta_true["beta_weekend_jw"], dtype=np.float64)
    bw_h = np.asarray(theta_hat["beta_weekend_jw"], dtype=np.float64)
    wl_t = bw_t[:, 1] - bw_t[:, 0]
    wl_h = bw_h[:, 1] - bw_h[:, 0]

    am_t = np.asarray(theta_true["a_m"], dtype=np.float64)
    am_h = np.asarray(theta_hat["a_m"], dtype=np.float64)

    bm_t = np.asarray(theta_true["b_m"], dtype=np.float64)
    bm_h = np.asarray(theta_hat["b_m"], dtype=np.float64)

    def pack(t: np.ndarray, h: np.ndarray) -> dict[str, float]:
        st = float(np.std(t))
        sh = float(np.std(h))
        return {"rmse": _rmse(t, h), "std_true": st, "std_hat": sh, "bias_std": sh - st}

    return {
        "weekend_lift_j": pack(wl_t, wl_h),
        "a_m": pack(am_t, am_h),
        "b_m": pack(bm_t, bm_h),
    }


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_bonus2(
    y_mit: np.ndarray,
    delta_mj: np.ndarray,
    p_choice_hat_mntc: np.ndarray,
    p_choice_oracle_mntc: np.ndarray | None,
    theta_hat: dict[str, Any] | None,
    theta_true: dict[str, Any] | None,
    mcmc: dict[str, Any] | None,
    eps: float,
) -> dict[str, Any]:
    """
    Evaluate fitted probabilities against the delta-only baseline (and optionally oracle).

    This function expects probabilities to be provided explicitly. It does not
    compute probabilities from theta.

    Returns:
      {
        "shape": {...},
        "models": {...},
        "deltas": {...},
        "param": {...} (optional),
        "mcmc": {...} (optional),
      }
    """
    y = np.asarray(y_mit, dtype=np.int64)
    delta = np.asarray(delta_mj, dtype=np.float64)
    p_fit = np.asarray(p_choice_hat_mntc, dtype=np.float64)

    M, N, T = (int(x) for x in y.shape)
    J = int(delta.shape[1])
    n_obs = int(M * N * T)

    # Baseline (delta-only, market-constant).
    p_base_mjc = delta_only_baseline_probs(delta)
    base_metrics = choice_metrics_from_market_probs(y, p_base_mjc, eps)

    # Fitted (per observation).
    fit_metrics = choice_metrics_from_probs(y, p_fit, eps)

    models: dict[str, dict[str, float]] = {
        "baseline": base_metrics,
        "fitted": fit_metrics,
    }

    # Oracle (optional).
    if p_choice_oracle_mntc is not None:
        p_or = np.asarray(p_choice_oracle_mntc, dtype=np.float64)
        models["oracle"] = choice_metrics_from_probs(y, p_or, eps)

    # Deltas are defined as fitted - comparator.
    deltas: dict[str, dict[str, float]] = {}
    deltas["fitted_minus_baseline"] = {
        "delta_nll": float(models["fitted"]["nll"] - models["baseline"]["nll"]),
        "delta_acc": float(models["fitted"]["acc"] - models["baseline"]["acc"]),
    }
    if "oracle" in models:
        deltas["fitted_minus_oracle"] = {
            "delta_nll": float(models["fitted"]["nll"] - models["oracle"]["nll"]),
            "delta_acc": float(models["fitted"]["acc"] - models["oracle"]["acc"]),
        }

    out: dict[str, Any] = {
        "shape": {"M": M, "N": N, "T": T, "J": J, "n_obs": n_obs},
        "models": models,
        "deltas": deltas,
    }

    # Parameter recovery (optional).
    if theta_true is not None and theta_hat is not None:
        out["param"] = {
            "mean_stats": parameter_recovery_mean_stats(theta_true, theta_hat),
            "dispersion_stats": parameter_recovery_dispersion_stats(
                theta_true, theta_hat
            ),
        }

    # MCMC acceptance rates / draws (optional).
    if mcmc is not None:
        out["mcmc"] = {
            "n_saved": mcmc.get("n_saved", None),
            "accept_rates": mcmc.get("accept", {}),
        }

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format the evaluate_bonus2 output into a compact, examiner-readable summary string."""
    shp = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    params = eval_out.get("param", None)
    mcmc = eval_out.get("mcmc", None)

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    if shp:
        lines.append(
            f"data: M={shp.get('M')} N={shp.get('N')} T={shp.get('T')} J={shp.get('J')} | n_obs={shp.get('n_obs')}"
        )
        lines.append("")

    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
        f"{'acc':>10} "
        f"{'p_true':>10} "
        f"{'out_emp':>8} "
        f"{'out_pred':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(tag: str, d: dict[str, Any]) -> str:
        return (
            f"{tag:<10}"
            f"{f6(float(d['nll']))} "
            f"{f6(float(d['acc']))} "
            f"{f6(float(d['p_true']))} "
            f"{f4(float(d['out_emp']))} "
            f"{f4(float(d['out_pred']))}"
        )

    if "baseline" in models:
        lines.append(row("baseline", models["baseline"]))
    if "fitted" in models:
        lines.append(row("fitted", models["fitted"]))
    if "oracle" in models:
        lines.append(row("oracle", models["oracle"]))

    dvb = deltas.get("fitted_minus_baseline", None)
    if dvb is not None:
        lines.append("")
        lines.append(
            f"fitted - baseline: Δnll={float(dvb['delta_nll']):.6f} | Δacc={float(dvb['delta_acc']):.6f}"
        )

    dvo = deltas.get("fitted_minus_oracle", None)
    if dvo is not None:
        lines.append(
            f"fitted - oracle:   Δnll={float(dvo['delta_nll']):.6f} | Δacc={float(dvo['delta_acc']):.6f}"
        )

    if isinstance(params, dict):
        mean_stats = params.get("mean_stats", {})
        disp_stats = params.get("dispersion_stats", {})

        if mean_stats:
            lines.append("")
            lines.append("parameter recovery (mean-level)")
            lines.append(
                f"{'param':<18}{'rmse':>10} {'mean_true':>10} {'mean_hat':>10} {'bias':>10}"
            )
            lines.append("-" * 60)
            for k, d in mean_stats.items():
                lines.append(
                    f"{k:<18}{d['rmse']:>10.6f} {d['mean_true']:>10.6f} {d['mean_hat']:>10.6f} {d['bias']:>10.6f}"
                )

        if disp_stats:
            lines.append("")
            lines.append("parameter recovery (dispersion)")
            lines.append(
                f"{'param':<18}{'rmse':>10} {'std_true':>10} {'std_hat':>10} {'bias_std':>10}"
            )
            lines.append("-" * 60)
            for k, d in disp_stats.items():
                lines.append(
                    f"{k:<18}{d['rmse']:>10.6f} {d['std_true']:>10.6f} {d['std_hat']:>10.6f} {d['bias_std']:>10.6f}"
                )

    if isinstance(mcmc, dict):
        rates = mcmc.get("accept_rates", {})
        n_saved = mcmc.get("n_saved", None)
        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("mcmc acceptance (block rates)")
            if n_saved is not None:
                lines.append(f"n_saved: {int(n_saved)}")
            lines.append(f"{'block':<16}{'rate':>10}")
            lines.append("-" * 26)
            for k in sorted(rates.keys()):
                lines.append(f"{k:<16}{float(rates[k]):>10.4f}")

    return "\n".join(lines)
