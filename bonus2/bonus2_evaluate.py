"""
bonus2_evaluate.py

Evaluation utilities for Bonus Q2 (updated spec).

Design goals:
- Metrics are pure NumPy.
- Baseline is the δ-only model (NOT empirical-share baseline).
- Supports (a) passing predicted choice probabilities directly, or
  (b) computing fitted/oracle probabilities from theta via bonus2_model.

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
# Validation
# =============================================================================


def _validate_choice_panel(y_mit: np.ndarray) -> tuple[int, int, int]:
    y = np.asarray(y_mit)
    if y.ndim != 3:
        raise ValueError(f"y_mit must have shape (M,N,T); got {y.shape}")
    return (int(y.shape[0]), int(y.shape[1]), int(y.shape[2]))


def _validate_probs(
    y_mit: np.ndarray,
    p_choice_mntc: np.ndarray,
    *,
    tol: float,
) -> tuple[int, int, int, int]:
    y = np.asarray(y_mit)
    p = np.asarray(p_choice_mntc, dtype=np.float64)

    if p.ndim != 4:
        raise ValueError(f"p_choice_mntc must have shape (M,N,T,C); got {p.shape}")

    M, N, T = _validate_choice_panel(y)
    if p.shape[0] != M or p.shape[1] != N or p.shape[2] != T:
        raise ValueError(
            f"p_choice_mntc first 3 dims must match y_mit (M,N,T)=({M},{N},{T}); got {p.shape}"
        )

    C = int(p.shape[3])
    y_min = int(np.min(y))
    y_max = int(np.max(y))
    if y_min < 0 or y_max >= C:
        raise ValueError(
            f"y_mit values must be in [0,{C-1}]; got min={y_min}, max={y_max}"
        )

    if not np.isfinite(p).all():
        raise ValueError("p_choice_mntc contains NaN/Inf")

    p_min = float(np.min(p))
    p_max = float(np.max(p))
    if p_min < -tol or p_max > 1.0 + tol:
        raise ValueError(
            f"p_choice_mntc must be in [0,1] (tol={tol}); got min={p_min:.6g}, max={p_max:.6g}"
        )

    rs = np.sum(p, axis=3)
    max_dev = float(np.max(np.abs(rs - 1.0)))
    if max_dev > tol:
        raise ValueError(
            f"p_choice_mntc rows must sum to 1 (tol={tol}); max |sum-1|={max_dev:.6g}"
        )

    return M, N, T, C


# =============================================================================
# Metrics
# =============================================================================


def choice_metrics_from_probs(
    y_mit: np.ndarray,
    p_choice_mntc: np.ndarray,
    *,
    eps: float = 1e-12,
    validate: bool = True,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Multinomial predictive metrics.

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

    if validate:
        _validate_probs(y, p, tol=tol)

    # p_true per observation
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
    if delta.ndim != 2:
        raise ValueError(f"delta_mj must have shape (M,J); got {delta.shape}")

    J = int(delta.shape[1])
    # Stable softmax with outside utility 0
    max_u = np.maximum(0.0, np.max(delta, axis=1))  # (M,)
    exp_in = np.exp(delta - max_u[:, None])  # (M,J)
    exp_out = np.exp(-max_u)  # (M,)
    den = exp_out + np.sum(exp_in, axis=1)  # (M,)

    p_out = (exp_out / den)[:, None]  # (M,1)
    p_in = exp_in / den[:, None]  # (M,J)
    p_mjc = np.concatenate([p_out, p_in], axis=1)  # (M,J+1)

    return p_mjc


def choice_metrics_from_market_probs(
    y_mit: np.ndarray,
    p_mjc: np.ndarray,
    *,
    eps: float = 1e-12,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Multinomial metrics when probabilities are market-only (M,C), constant in (i,t).
    """
    y = np.asarray(y_mit, dtype=np.int64)
    M, N, T = _validate_choice_panel(y)
    p = np.asarray(p_mjc, dtype=np.float64)

    if p.ndim != 2:
        raise ValueError(f"p_mjc must have shape (M,C); got {p.shape}")
    if int(p.shape[0]) != M:
        raise ValueError(f"p_mjc first axis must be M={M}; got {p.shape}")
    C = int(p.shape[1])

    y_min = int(np.min(y))
    y_max = int(np.max(y))
    if y_min < 0 or y_max >= C:
        raise ValueError(
            f"y_mit values must be in [0,{C-1}]; got min={y_min}, max={y_max}"
        )

    if not np.isfinite(p).all():
        raise ValueError("p_mjc contains NaN/Inf")
    p_min = float(np.min(p))
    p_max = float(np.max(p))
    if p_min < -tol or p_max > 1.0 + tol:
        raise ValueError(
            f"p_mjc must be in [0,1] (tol={tol}); got min={p_min:.6g}, max={p_max:.6g}"
        )
    rs = np.sum(p, axis=1)
    max_dev = float(np.max(np.abs(rs - 1.0)))
    if max_dev > tol:
        raise ValueError(
            f"p_mjc rows must sum to 1 (tol={tol}); max |sum-1|={max_dev:.6g}"
        )

    n_obs = float(M * N * T)
    out_emp = float(np.mean(y == 0))

    sum_logp = 0.0
    sum_ptrue = 0.0
    sum_acc = 0.0
    sum_outpred = 0.0

    for m in range(M):
        y_flat = y[m].ravel()
        p_row = p[m]

        p_true = p_row[y_flat]
        p_true_clip = np.clip(p_true, eps, 1.0)

        sum_ptrue += float(np.sum(p_true))
        sum_logp += float(np.sum(np.log(p_true_clip)))

        c_hat = int(np.argmax(p_row))
        sum_acc += float(np.sum(y_flat == c_hat))

        # same for every (i,t) within market m
        sum_outpred += float(p_row[0]) * float(N * T)

    nll = float(-sum_logp / n_obs)
    acc = float(sum_acc / n_obs)
    out_pred = float(sum_outpred / n_obs)

    return {
        "nll": nll,
        "acc": acc,
        "p_true": float(sum_ptrue / n_obs),
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
    validate_probs: bool = True,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Evaluate Bonus2 predictions and (optionally) parameter recovery and MCMC diagnostics.

    You can provide predicted probabilities directly (p_choice_*), or provide theta_* plus
    the required inputs to compute probabilities via the TF model.

    Output schema:
      {
        "shape": {...},
        "models": {"baseline":..., "fitted":..., "oracle":... (optional)},
        "deltas": {"fitted_minus_baseline":..., "fitted_minus_oracle":... (optional)},
        "param": {"mean_stats":..., "std_stats":...} (optional),
        "mcmc": {"n_saved": int|None, "accept_rates": dict[str,float]} (optional)
      }
    """
    y = np.asarray(y_mit, dtype=np.int64)
    M, N, T = _validate_choice_panel(y)
    delta = np.asarray(delta_mj, dtype=np.float64)
    if delta.shape[0] != M:
        raise ValueError(f"delta_mj first axis must be M={M}; got {delta.shape}")
    J = int(delta.shape[1])
    n_obs = int(M * N * T)

    # Baseline
    if p_choice_baseline_mntc is None:
        p_base_mjc = delta_only_baseline_probs(delta)
        base_metrics = choice_metrics_from_market_probs(y, p_base_mjc, eps=eps, tol=tol)
    else:
        p_base = np.asarray(p_choice_baseline_mntc, dtype=np.float64)
        base_metrics = choice_metrics_from_probs(
            y, p_base, eps=eps, validate=validate_probs, tol=tol
        )

    # Fitted
    if p_choice_hat_mntc is None:
        if theta_hat is None:
            raise ValueError(
                "Provide either p_choice_hat_mntc or theta_hat (+ model inputs)"
            )

        need = (w_t, season_sin_kt, season_cos_kt, peer_adj_m, L, decay)
        if any(x is None for x in need):
            raise ValueError(
                "theta_hat provided but missing model inputs to compute probabilities"
            )

        p_fit = _predict_probs_via_model(
            theta=theta_hat,
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
        need = (w_t, season_sin_kt, season_cos_kt, peer_adj_m, L, decay)
        if any(x is None for x in need):
            raise ValueError(
                "theta_true provided but missing model inputs to compute oracle probabilities"
            )
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
        "fitted": choice_metrics_from_probs(
            y, p_fit, eps=eps, validate=validate_probs, tol=tol
        ),
    }
    if p_or is not None:
        models["oracle"] = choice_metrics_from_probs(
            y, p_or, eps=eps, validate=validate_probs, tol=tol
        )

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
