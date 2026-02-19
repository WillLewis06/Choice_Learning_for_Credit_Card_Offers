"""
bonus2/bonus2_evaluate.py

Evaluation utilities for Bonus Q2 (updated spec).

Design goals:
- Metrics are pure NumPy.
- Baseline is the δ-only model (NOT empirical-share baseline).
- Supports (a) passing predicted choice probabilities directly, or
  (b) computing fitted/oracle probabilities from theta via bonus2_model.

Assumption:
- This module is called only with inputs that already satisfy the Bonus2
  contracts (validated upstream in bonus2_input_validation.py). Therefore,
  this file does not perform input-contract validation (shapes/ranges/sums).

Choice encoding:
  y_mit (M,N,T) with values:
    0 = outside option
    c = j+1 for inside product j in {1..J}

Predicted probabilities:
  p_choice_mntc (M,N,T,J+1) with c=0 outside, c=j+1 inside j.

Time features:
  w_t (T,) int in {0,1} where 1=weekend, 0=weekday.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Metrics
# =============================================================================


def choice_metrics_from_probs(
    y_mit: np.ndarray,
    p_choice_mntc: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Multinomial predictive metrics from per-observation probabilities.

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


def delta_only_baseline_probs(delta_mj: np.ndarray) -> np.ndarray:
    """
    δ-only baseline probabilities per market.

    Utility:
      v_out = 0
      v_in(j) = delta_mj[m,j]

    Returns:
      p_mjc: (M, J+1) with c=0 outside and c=j+1 for inside product j.
    """
    delta = np.asarray(delta_mj, dtype=np.float64)
    J = int(delta.shape[1])

    max_u = np.maximum(0.0, np.max(delta, axis=1))  # (M,)
    exp_in = np.exp(delta - max_u[:, None])  # (M,J)
    exp_out = np.exp(-max_u)  # (M,)
    den = exp_out + np.sum(exp_in, axis=1)  # (M,)

    p_out = (exp_out / den)[:, None]  # (M,1)
    p_in = exp_in / den[:, None]  # (M,J)
    return np.concatenate([p_out, p_in], axis=1)  # (M,J+1)


def choice_metrics_from_market_probs(
    y_mit: np.ndarray,
    p_mjc: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Multinomial metrics when probabilities are market-only (M,C), constant in (i,t).
    """
    y = np.asarray(y_mit, dtype=np.int64)
    p = np.asarray(p_mjc, dtype=np.float64)

    # Broadcast to (M,N,T,C) then pick realized class
    p4 = p[:, None, None, :]
    p_true = np.take_along_axis(p4, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)

    nll = float(-np.mean(np.log(p_true_clip)))
    c_hat = np.argmax(p, axis=1)  # (M,)
    acc = float(np.mean(y == c_hat[:, None, None]))
    out_emp = float(np.mean(y == 0))
    out_pred = float(np.mean(p[:, 0]))

    return {
        "nll": nll,
        "acc": acc,
        "p_true": float(np.mean(p_true)),
        "out_emp": out_emp,
        "out_pred": out_pred,
    }


# =============================================================================
# Parameter recovery
# =============================================================================


def _rmse(true: np.ndarray, hat: np.ndarray) -> float:
    d = np.asarray(hat, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def parameter_recovery_mean_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
    *,
    order: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    RMSE and mean statistics for updated-spec parameters.

    Default keys:
      beta_market_j, beta_habit_j, beta_peer_j, a_m, b_m
    """
    if order is None:
        order = ["beta_market_j", "beta_habit_j", "beta_peer_j", "a_m", "b_m"]

    out: dict[str, dict[str, float]] = {}
    for k in order:
        if k not in theta_true or k not in theta_hat:
            continue
        t = np.asarray(theta_true[k], dtype=np.float64)
        h = np.asarray(theta_hat[k], dtype=np.float64)
        mt = float(np.mean(t))
        mh = float(np.mean(h))
        out[k] = {"rmse": _rmse(t, h), "mean_true": mt, "mean_hat": mh, "bias": mh - mt}
    return out


def parameter_recovery_std_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
    *,
    order: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    RMSE and std statistics for dispersion-type parameters.

    Default keys:
      beta_dow_j
    """
    if order is None:
        order = ["beta_dow_j"]

    out: dict[str, dict[str, float]] = {}
    for k in order:
        if k not in theta_true or k not in theta_hat:
            continue
        t = np.asarray(theta_true[k], dtype=np.float64)
        h = np.asarray(theta_hat[k], dtype=np.float64)
        st = float(np.std(t))
        sh = float(np.std(h))
        out[k] = {
            "rmse": _rmse(t, h),
            "std_true": st,
            "std_hat": sh,
            "bias_std": sh - st,
        }
    return out


# =============================================================================
# Optional probability computation via TF model
# =============================================================================


def _predict_probs_via_model(
    theta: dict[str, Any],
    y_mit: np.ndarray,
    delta_mj: np.ndarray,
    w_t: np.ndarray,
    season_sin_kt: np.ndarray,
    season_cos_kt: np.ndarray,
    peer_adj_m: Any,
    L: int,
    decay: float,
) -> np.ndarray:
    import tensorflow as tf
    from bonus2 import bonus2_model as model

    y_tf = tf.convert_to_tensor(y_mit, dtype=tf.int32)
    delta_tf = tf.convert_to_tensor(delta_mj, dtype=tf.float64)
    w_tf = tf.convert_to_tensor(w_t, dtype=tf.int32)
    sin_tf = tf.convert_to_tensor(season_sin_kt, dtype=tf.float64)
    cos_tf = tf.convert_to_tensor(season_cos_kt, dtype=tf.float64)
    L_tf = tf.convert_to_tensor(int(L), dtype=tf.int32)
    decay_tf = tf.convert_to_tensor(float(decay), dtype=tf.float64)

    # Call positionally to avoid coupling to argument names.
    p_tf = model.predict_choice_probs_from_theta(
        theta,
        y_tf,
        delta_tf,
        w_tf,
        sin_tf,
        cos_tf,
        peer_adj_m,
        L_tf,
        decay_tf,
    )
    return p_tf.numpy()


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_bonus2(
    y_mit: np.ndarray,
    *,
    delta_mj: np.ndarray,
    p_choice_hat_mntc: np.ndarray | None = None,
    theta_hat: dict[str, Any] | None = None,
    theta_true: dict[str, Any] | None = None,
    p_choice_oracle_mntc: np.ndarray | None = None,
    p_choice_baseline_mntc: np.ndarray | None = None,
    w_t: np.ndarray | None = None,
    season_sin_kt: np.ndarray | None = None,
    season_cos_kt: np.ndarray | None = None,
    peer_adj_m: Any | None = None,
    L: int | None = None,
    decay: float | None = None,
    mcmc: dict[str, Any] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Evaluate Bonus2 predictions and (optionally) parameter recovery and MCMC diagnostics.

    You can provide predicted probabilities directly (p_choice_*), or provide theta_* plus
    the required inputs to compute probabilities via the TF model.
    """
    y = np.asarray(y_mit, dtype=np.int64)
    delta = np.asarray(delta_mj, dtype=np.float64)

    M, N, T = (int(x) for x in y.shape)
    J = int(delta.shape[1])
    n_obs = int(M * N * T)

    # Baseline
    if p_choice_baseline_mntc is None:
        p_base_mjc = delta_only_baseline_probs(delta)
        base_metrics = choice_metrics_from_market_probs(y, p_base_mjc, eps=eps)
    else:
        p_base = np.asarray(p_choice_baseline_mntc, dtype=np.float64)
        base_metrics = choice_metrics_from_probs(y, p_base, eps=eps)

    # Fitted
    if p_choice_hat_mntc is None:
        p_fit = _predict_probs_via_model(
            theta=theta_hat,  # assumed provided
            y_mit=y,
            delta_mj=delta,
            w_t=np.asarray(w_t, dtype=np.int64),
            season_sin_kt=np.asarray(season_sin_kt, dtype=np.float64),
            season_cos_kt=np.asarray(season_cos_kt, dtype=np.float64),
            peer_adj_m=peer_adj_m,
            L=int(L),
            decay=float(decay),
        )
    else:
        p_fit = np.asarray(p_choice_hat_mntc, dtype=np.float64)

    # Oracle (optional)
    p_or = None
    if p_choice_oracle_mntc is not None:
        p_or = np.asarray(p_choice_oracle_mntc, dtype=np.float64)
    elif theta_true is not None:
        p_or = _predict_probs_via_model(
            theta=theta_true,
            y_mit=y,
            delta_mj=delta,
            w_t=np.asarray(w_t, dtype=np.int64),
            season_sin_kt=np.asarray(season_sin_kt, dtype=np.float64),
            season_cos_kt=np.asarray(season_cos_kt, dtype=np.float64),
            peer_adj_m=peer_adj_m,
            L=int(L),
            decay=float(decay),
        )

    # Metrics
    models: dict[str, dict[str, float]] = {
        "baseline": base_metrics,
        "fitted": choice_metrics_from_probs(y, p_fit, eps=eps),
    }
    if p_or is not None:
        models["oracle"] = choice_metrics_from_probs(y, p_or, eps=eps)

    deltas: dict[str, dict[str, float]] = {}
    base = models["baseline"]
    fit = models["fitted"]
    deltas["fitted_minus_baseline"] = {
        "delta_nll": float(fit["nll"] - base["nll"]),
        "delta_acc": float(fit["acc"] - base["acc"]),
    }
    if "oracle" in models:
        ora = models["oracle"]
        deltas["fitted_minus_oracle"] = {
            "delta_nll": float(fit["nll"] - ora["nll"]),
            "delta_acc": float(fit["acc"] - ora["acc"]),
        }

    out: dict[str, Any] = {
        "shape": {"M": M, "N": N, "T": T, "J": J, "n_obs": n_obs},
        "models": models,
        "deltas": deltas,
    }

    if theta_true is not None and theta_hat is not None:
        out["param"] = {
            "mean_stats": parameter_recovery_mean_stats(theta_true, theta_hat),
            "std_stats": parameter_recovery_std_stats(theta_true, theta_hat),
        }

    if mcmc is not None:
        accept_rates = mcmc.get("accept", {})
        n_saved = mcmc.get("n_saved", None)
        out["mcmc"] = {"n_saved": n_saved, "accept_rates": accept_rates}

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    shp = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    params = eval_out.get("param", None)
    mcmc = eval_out.get("mcmc", None)

    M = shp.get("M")
    N = shp.get("N")
    T = shp.get("T")
    J = shp.get("J")
    n_obs = shp.get("n_obs")

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    if M is not None and N is not None and T is not None and J is not None:
        lines.append(f"data: M={M} N={N} T={T} J={J} | n_obs={n_obs}")
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
            f"fitted - baseline: Δnll={dvb['delta_nll']:.6f} | Δacc={dvb['delta_acc']:.6f}"
        )

    dvo = deltas.get("fitted_minus_oracle", None)
    if dvo is not None:
        lines.append(
            f"fitted - oracle:   Δnll={dvo['delta_nll']:.6f} | Δacc={dvo['delta_acc']:.6f}"
        )

    if isinstance(params, dict):
        mean_stats = params.get("mean_stats", {})
        std_stats = params.get("std_stats", {})

        if mean_stats:
            lines.append("")
            lines.append("parameter recovery (hat, mean stats)")
            lines.append(
                f"{'param':<16}{'rmse':>10} {'mean_true':>10} {'mean_hat':>10} {'bias':>10}"
            )
            lines.append("-" * 56)
            for k, d in mean_stats.items():
                lines.append(
                    f"{k:<16}{d['rmse']:>10.6f} {d['mean_true']:>10.6f} {d['mean_hat']:>10.6f} {d['bias']:>10.6f}"
                )

        if std_stats:
            lines.append("")
            lines.append("parameter recovery (hat, std stats)")
            lines.append(
                f"{'param':<16}{'rmse':>10} {'std_true':>10} {'std_hat':>10} {'bias_std':>10}"
            )
            lines.append("-" * 56)
            for k, d in std_stats.items():
                lines.append(
                    f"{k:<16}{d['rmse']:>10.6f} {d['std_true']:>10.6f} {d['std_hat']:>10.6f} {d['bias_std']:>10.6f}"
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
