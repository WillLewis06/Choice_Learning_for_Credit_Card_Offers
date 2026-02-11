# ching/stockpiling_evaluate.py
#
# Lightweight evaluation utilities for the stockpiling model.
#
# Design goals:
# - Pure NumPy (no TensorFlow).
# - No DP / posterior / filtering mechanics inside evaluation.
# - Evaluate from predicted buy probabilities and (optionally) parameter truth.
#
# Outputs:
# - Predictive fit: NLL per obs, RMSE of probability predictions vs actions, buy rates,
#   plus optional by-price-state summaries.
# - Parameter recovery (optional): RMSE + means for each parameter block.
# - MCMC diagnostics (optional): acceptance summaries, passed through from the estimator.

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# =============================================================================
# Helpers
# =============================================================================


def _as_float(x: Any) -> Optional[float]:
    """Convert a scalar-like value to float; return None if conversion fails."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return None


def _as_int(x: Any) -> Optional[int]:
    """Convert a scalar-like value to int; return None if conversion fails."""
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(np.asarray(x).item())
        except Exception:
            return None


# =============================================================================
# Parameter recovery
# =============================================================================


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Compare true vs fitted constrained parameters.

    For each key present in both dictionaries:
      - rmse
      - mean_true
      - mean_hat

    Notes:
      - Arrays may be (M,N) or (M,) depending on the parameter block.
      - Keys are not enforced; this function compares only intersecting keys.
    """
    out: dict[str, dict[str, float]] = {}

    def _rmse(t: np.ndarray, h: np.ndarray) -> float:
        d = (h - t).astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(d * d)))

    keys = [k for k in theta_hat.keys() if k in theta_true]
    for k in keys:
        true = np.asarray(theta_true[k], dtype=np.float64)
        hat = np.asarray(theta_hat[k], dtype=np.float64)
        out[k] = {
            "rmse": _rmse(true, hat),
            "mean_true": float(np.mean(true)),
            "mean_hat": float(np.mean(hat)),
        }

    return out


# =============================================================================
# Predictive fit (NLL + RMSE) from probabilities
# =============================================================================


def predictive_metrics_from_probs(
    a_imt: np.ndarray,
    p_buy_imt: np.ndarray,
    p_state_mt: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute predictive metrics given predicted buy probabilities.

    Inputs:
      a_imt: (M,N,T) actions in {0,1}
      p_buy_imt: (M,N,T) predicted probabilities in [0,1]
      p_state_mt: optional (M,T) discrete price state indices

    Metrics:
      - nll_per_obs: mean negative log likelihood
      - rmse_prob: sqrt(mean((p-a)^2))
      - buy_rate_emp: mean(a)
      - buy_rate_pred: mean(p)

    By-state summaries (if p_state_mt provided):
      - buy_rate_by_state_emp[s] = mean(a | state=s)
      - buy_rate_by_state_pred[s] = mean(p | state=s)
      - rmse_buy_rate_by_state = sqrt(mean_s (pred(s)-emp(s))^2), with equal weight per state
    """
    a = np.asarray(a_imt, dtype=np.float64)
    p = np.asarray(p_buy_imt, dtype=np.float64)

    if a.shape != p.shape:
        raise ValueError(f"a_imt shape {a.shape} must match p_buy_imt shape {p.shape}")

    M, N, T = a.shape
    n_obs = int(M * N * T)

    p = np.clip(p, eps, 1.0 - eps)

    nll = -np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p))
    rmse_prob = float(np.sqrt(np.mean((p - a) ** 2)))

    buy_rate_emp = float(np.mean(a))
    buy_rate_pred = float(np.mean(p))

    p0 = float(np.clip(buy_rate_emp, eps, 1.0 - eps))
    baseline_nll = -np.mean(a * np.log(p0) + (1.0 - a) * np.log(1.0 - p0))
    baseline_rmse = float(np.sqrt(np.mean((p0 - a) ** 2)))

    by_state_emp: dict[int, float] = {}
    by_state_pred: dict[int, float] = {}
    rmse_by_state = float("nan")

    if p_state_mt is not None:
        st = np.asarray(p_state_mt)
        if st.shape != (M, T):
            raise ValueError(
                f"p_state_mt must have shape (M,T) = {(M, T)}, got {st.shape}"
            )

        st_vals = np.unique(st.astype(np.int64, copy=False))
        diffs: list[float] = []

        for s in st_vals.tolist():
            mask_mt = st == s  # (M,T)
            count_mt = int(mask_mt.sum())
            if count_mt == 0:
                continue

            # Broadcast mask over N via multiplication (avoid boolean indexing pitfalls).
            mask_mnt = mask_mt[:, None, :]  # (M,1,T)
            den = float(count_mt * N)

            emp = float(np.sum(a * mask_mnt) / den)
            pred = float(np.sum(p * mask_mnt) / den)

            by_state_emp[int(s)] = emp
            by_state_pred[int(s)] = pred
            diffs.append((pred - emp) ** 2)

        if diffs:
            rmse_by_state = float(np.sqrt(np.mean(diffs)))

    return {
        "shape": {"M": int(M), "N": int(N), "T": int(T), "n_obs": n_obs},
        "nll_per_obs": float(nll),
        "rmse_prob": float(rmse_prob),
        "buy_rate_emp": float(buy_rate_emp),
        "buy_rate_pred": float(buy_rate_pred),
        "buy_rate_by_state_emp": by_state_emp,
        "buy_rate_by_state_pred": by_state_pred,
        "rmse_buy_rate_by_state": float(rmse_by_state),
        "baseline": {
            "p0": float(p0),
            "nll_per_obs": float(baseline_nll),
            "rmse_prob": float(baseline_rmse),
            "buy_rate_emp": float(buy_rate_emp),
            "buy_rate_pred": float(p0),
        },
    }


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_stockpiling(
    a_imt: np.ndarray,
    p_buy_hat_imt: np.ndarray,
    p_state_mt: Optional[np.ndarray] = None,
    theta_hat: Optional[dict[str, np.ndarray]] = None,
    theta_true: Optional[dict[str, np.ndarray]] = None,
    p_buy_oracle_imt: Optional[np.ndarray] = None,
    mcmc: Optional[dict[str, Any]] = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Evaluate predictive fit and (optionally) parameter recovery and MCMC diagnostics.

    Required:
      - a_imt: (M,N,T)
      - p_buy_hat_imt: (M,N,T)

    Optional:
      - p_state_mt: (M,T) for by-state summaries
      - theta_true + theta_hat for parameter recovery
      - p_buy_oracle_imt: (M,N,T) for oracle predictive metrics
      - mcmc: dict passed through from estimator.get_results()
    """
    out: dict[str, Any] = {}

    out["fit"] = predictive_metrics_from_probs(
        a_imt=a_imt,
        p_buy_imt=p_buy_hat_imt,
        p_state_mt=p_state_mt,
        eps=eps,
    )

    if p_buy_oracle_imt is not None:
        out["oracle"] = predictive_metrics_from_probs(
            a_imt=a_imt,
            p_buy_imt=p_buy_oracle_imt,
            p_state_mt=p_state_mt,
            eps=eps,
        )

    if theta_true is not None:
        if theta_hat is None:
            raise ValueError("theta_true provided but theta_hat is None")
        out["param"] = parameter_metrics(theta_true, theta_hat)

    if mcmc is not None:
        out["mcmc"] = mcmc

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(
    eval_out: dict[str, Any],
    param_order: Optional[list[str]] = None,
) -> str:
    """
    Format evaluation output into a reviewer-friendly text report.

    Expects eval_out from evaluate_stockpiling(...), with at least:
      - eval_out["fit"]
    """
    fit = eval_out["fit"]
    oracle = eval_out.get("oracle")
    params = eval_out.get("param")
    mcmc = eval_out.get("mcmc")

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

    if M is not None and N is not None and T is not None:
        lines.append(f"data: M={M} N={N} T={T} | n_obs={n_obs}")
    lines.append("")

    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
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
            f"{f6(d['rmse_prob'])} "
            f"{f4(d['buy_rate_emp'])} "
            f"{f4(d['buy_rate_pred'])}"
        )

    lines.append(row("baseline", base))
    lines.append(row("fitted", fit))
    if oracle is not None:
        lines.append(row("oracle", oracle))

    lines.append("")
    nll_gain = base["nll_per_obs"] - fit["nll_per_obs"]
    rmse_gain = base["rmse_prob"] - fit["rmse_prob"]
    lines.append(f"gain vs baseline: Δnll={nll_gain:.6f} | Δrmse={rmse_gain:.6f}")
    if oracle is not None:
        lines.append(
            f"fitted - oracle: Δnll={(fit['nll_per_obs'] - oracle['nll_per_obs']):.6f} | "
            f"Δrmse={(fit['rmse_prob'] - oracle['rmse_prob']):.6f}"
        )

    emp_s = fit.get("buy_rate_by_state_emp", {})
    pred_s = fit.get("buy_rate_by_state_pred", {})
    if emp_s:
        lines.append("")
        lines.append("buy rate by price state")
        st_header = f"{'state':<8}{'emp':>10} {'pred':>10} {'diff':>10}"
        lines.append(st_header)
        lines.append("-" * len(st_header))
        for s in sorted(emp_s.keys()):
            emp = float(emp_s[s])
            pred = float(pred_s.get(s, float("nan")))
            diff = pred - emp
            lines.append(f"{str(s):<8}{f6(emp)} {f6(pred)} {f6(diff)}")
        lines.append(f"rmse across states: {fit['rmse_buy_rate_by_state']:.6f}")

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
                f"{f6(pk['mean_true'])} "
                f"{f6(pk['mean_hat'])} "
                f"{f6(bias)}"
            )

    if isinstance(mcmc, dict):
        accept = mcmc.get("accept", {})
        rates = accept.get("rates", {})
        counts = accept.get("counts", {})
        n_saved = _as_int(mcmc.get("n_saved", None))

        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("mcmc acceptance (elementwise)")
            if n_saved is not None:
                lines.append(f"n_saved: {n_saved}")

            a_header = f"{'block':<10}{'rate':>10}{'accepted':>12}{'proposed':>12}"
            lines.append(a_header)
            lines.append("-" * len(a_header))

            order = ["beta", "alpha", "v", "fc", "lambda_c", "u_scale"]

            total_acc = 0
            total_prop = 0

            for k in order:
                if k not in rates:
                    continue

                r = _as_float(rates.get(k))
                c = _as_int(counts.get(k))

                proposed = None
                if n_saved is not None and M is not None and N is not None:
                    if k == "u_scale":
                        proposed = n_saved * int(M)
                    else:
                        proposed = n_saved * int(M) * int(N)

                if c is not None:
                    total_acc += c
                if proposed is not None:
                    total_prop += proposed

                r_str = f"{r:>10.4f}" if r is not None else f"{'':>10}"
                c_str = f"{c:>12d}" if c is not None else f"{'':>12}"
                p_str = f"{proposed:>12d}" if proposed is not None else f"{'':>12}"
                lines.append(f"{k:<10}{r_str}{c_str}{p_str}")

            if total_prop > 0:
                overall = total_acc / max(1, total_prop)
                lines.append(f"overall: {overall:.4f} ({total_acc}/{total_prop})")

    return "\n".join(lines)
