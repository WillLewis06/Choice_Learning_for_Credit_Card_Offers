"""
bonus2/bonus2_evaluate.py

Evaluation utilities for Bonus Q2.

Scope:
- Pure NumPy predictive metrics from realized choices and predicted probabilities.
- Baseline is the delta-only model (outside utility 0, inside utility = delta_mj).
- Optional parameter-recovery summaries on the structural parameter scale.
- Optional chain summaries derived from chunk-level diagnostics produced by the
  refactored TFP sampler.

Non-goals:
- No TensorFlow calls.
- No routing/fallback logic to compute probabilities from theta.
- No input validation (assumed handled upstream).

Choice encoding:
  y_mit (M,N,T) with values:
    0 = outside option
    c = j+1 for inside product j in {1..J}

Predicted probabilities:
  p_choice_mntc (M,N,T,J+1) with c=0 outside, c=j+1 inside j.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    p_true = np.take_along_axis(p, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)

    nll = float(-np.mean(np.log(p_true_clip)))
    acc = float(np.mean(np.argmax(p, axis=3) == y))
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

    p4 = p[:, None, None, :]
    p_true = np.take_along_axis(p4, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)

    nll = float(-np.mean(np.log(p_true_clip)))
    c_hat_m = np.argmax(p, axis=1)
    acc = float(np.mean(y == c_hat_m[:, None, None]))
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

    max_u = np.maximum(0.0, np.max(delta, axis=1))
    exp_in = np.exp(delta - max_u[:, None])
    exp_out = np.exp(-max_u)
    den = exp_out + np.sum(exp_in, axis=1)

    p_out = (exp_out / den)[:, None]
    p_in = exp_in / den[:, None]
    return np.concatenate([p_out, p_in], axis=1)


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
# Chain summaries
# =============================================================================


def _get_field(summary: Any, name: str) -> float:
    """Read a scalar field from either an object with attributes or a dict."""
    if isinstance(summary, dict):
        value = summary[name]
    else:
        value = getattr(summary, name)
    return float(value)


def chain_summary_from_chunk_summaries(
    chunk_summaries: Sequence[Any],
    n_saved: int | None,
) -> dict[str, Any]:
    """
    Build a compact chain summary from chunk-level diagnostics.

    Expected fields on each chunk summary:
      beta_intercept_accept_mean
      beta_habit_accept_mean
      beta_peer_accept_mean
      beta_weekend_accept_mean
      a_accept_mean
      b_accept_mean
      joint_logpost_last
    """
    if len(chunk_summaries) == 0:
        return {
            "n_saved": n_saved,
            "num_chunks": 0,
            "accept_rates": {},
            "final_joint_logpost": None,
        }

    accept_rates = {
        "beta_intercept": float(
            np.mean(
                [_get_field(s, "beta_intercept_accept_mean") for s in chunk_summaries]
            )
        ),
        "beta_habit": float(
            np.mean([_get_field(s, "beta_habit_accept_mean") for s in chunk_summaries])
        ),
        "beta_peer": float(
            np.mean([_get_field(s, "beta_peer_accept_mean") for s in chunk_summaries])
        ),
        "beta_weekend": float(
            np.mean(
                [_get_field(s, "beta_weekend_accept_mean") for s in chunk_summaries]
            )
        ),
        "a_m": float(
            np.mean([_get_field(s, "a_accept_mean") for s in chunk_summaries])
        ),
        "b_m": float(
            np.mean([_get_field(s, "b_accept_mean") for s in chunk_summaries])
        ),
    }

    final_joint_logpost = _get_field(chunk_summaries[-1], "joint_logpost_last")

    return {
        "n_saved": n_saved,
        "num_chunks": int(len(chunk_summaries)),
        "accept_rates": accept_rates,
        "final_joint_logpost": final_joint_logpost,
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
    chunk_summaries: Sequence[Any] | None,
    n_saved: int | None,
    eps: float,
) -> dict[str, Any]:
    """
    Evaluate fitted probabilities against the delta-only baseline and optionally oracle.

    This function expects probabilities to be provided explicitly. It does not
    compute probabilities from theta.

    Returns:
      {
        "shape": {...},
        "models": {...},
        "deltas": {...},
        "param": {...} (optional),
        "chain": {...} (optional),
      }
    """
    y = np.asarray(y_mit, dtype=np.int64)
    delta = np.asarray(delta_mj, dtype=np.float64)
    p_fit = np.asarray(p_choice_hat_mntc, dtype=np.float64)

    M, N, T = (int(x) for x in y.shape)
    J = int(delta.shape[1])
    n_obs = int(M * N * T)

    p_base_mjc = delta_only_baseline_probs(delta)
    base_metrics = choice_metrics_from_market_probs(y, p_base_mjc, eps)
    fit_metrics = choice_metrics_from_probs(y, p_fit, eps)

    models: dict[str, dict[str, float]] = {
        "baseline": base_metrics,
        "fitted": fit_metrics,
    }

    if p_choice_oracle_mntc is not None:
        p_or = np.asarray(p_choice_oracle_mntc, dtype=np.float64)
        models["oracle"] = choice_metrics_from_probs(y, p_or, eps)

    deltas: dict[str, dict[str, float]] = {
        "fitted_minus_baseline": {
            "delta_nll": float(models["fitted"]["nll"] - models["baseline"]["nll"]),
            "delta_acc": float(models["fitted"]["acc"] - models["baseline"]["acc"]),
        }
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

    if theta_true is not None and theta_hat is not None:
        out["param"] = {
            "mean_stats": parameter_recovery_mean_stats(theta_true, theta_hat),
            "dispersion_stats": parameter_recovery_dispersion_stats(
                theta_true, theta_hat
            ),
        }

    if chunk_summaries is not None:
        out["chain"] = chain_summary_from_chunk_summaries(
            chunk_summaries=chunk_summaries,
            n_saved=n_saved,
        )

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format evaluate_bonus2 output into a compact summary string."""
    shp = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    params = eval_out.get("param", None)
    chain = eval_out.get("chain", None)

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

    if isinstance(chain, dict):
        rates = chain.get("accept_rates", {})
        n_saved = chain.get("n_saved", None)
        num_chunks = chain.get("num_chunks", None)
        final_joint_logpost = chain.get("final_joint_logpost", None)

        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("chain summary")
            if n_saved is not None:
                lines.append(f"n_saved: {int(n_saved)}")
            if num_chunks is not None:
                lines.append(f"num_chunks: {int(num_chunks)}")
            if final_joint_logpost is not None:
                lines.append(f"final_joint_logpost: {float(final_joint_logpost):.6f}")
            lines.append("")
            lines.append("acceptance (mean block rates)")
            lines.append(f"{'block':<16}{'rate':>10}")
            lines.append("-" * 26)
            for k in sorted(rates.keys()):
                lines.append(f"{k:<16}{float(rates[k]):>10.4f}")

    return "\n".join(lines)
