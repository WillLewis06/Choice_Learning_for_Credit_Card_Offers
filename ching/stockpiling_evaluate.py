"""
ching/stockpiling_evaluate.py

NumPy evaluation utilities for the Phase-3 stockpiling model.

This module assumes inputs have already been validated/normalized upstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PARAM_KEYS = ("beta", "alpha", "v", "fc", "u_scale")


def predictive_metrics_from_probs(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute predictive metrics from predicted buy probabilities.

    Args:
      a_mnjt: (M,N,J,T) actions in {0,1}
      p_buy_mnjt: (M,N,J,T) predicted probabilities in [0,1]
      eps: numerical clipping constant for log-loss

    Returns:
      dict with keys: nll_per_obs, brier, rmse_prob, buy_rate_emp, buy_rate_pred
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    p = np.asarray(p_buy_mnjt, dtype=np.float64)

    p = np.clip(p, eps, 1.0 - eps)

    nll = -np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p))
    brier = float(np.mean((p - a) ** 2))
    rmse = float(np.sqrt(brier))

    return {
        "nll_per_obs": float(nll),
        "brier": float(brier),
        "rmse_prob": float(rmse),
        "buy_rate_emp": float(np.mean(a)),
        "buy_rate_pred": float(np.mean(p)),
    }


def baseline_metrics_from_actions(
    a_mnjt: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Baseline model: constant probability p0 equal to empirical buy rate.

    Args:
      a_mnjt: (M,N,J,T) actions in {0,1}
      eps: numerical clipping constant for log-loss

    Returns:
      dict with keys: p0, nll_per_obs, brier, rmse_prob, buy_rate_emp, buy_rate_pred
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    buy_rate_emp = float(np.mean(a))

    p0 = float(np.clip(buy_rate_emp, eps, 1.0 - eps))
    nll = -np.mean(a * np.log(p0) + (1.0 - a) * np.log(1.0 - p0))
    brier = float(np.mean((p0 - a) ** 2))
    rmse = float(np.sqrt(brier))

    return {
        "p0": float(p0),
        "nll_per_obs": float(nll),
        "brier": float(brier),
        "rmse_prob": float(rmse),
        "buy_rate_emp": float(buy_rate_emp),
        "buy_rate_pred": float(p0),
    }


def by_price_state_summary(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    s_mjt: np.ndarray,
) -> dict[str, Any]:
    """By-price-state buy rates.

    Aggregation:
      - Sum over consumers first -> (M,J,T) totals
      - Bin by observed price state s in s_mjt
      - Report state-level empirical and predicted buy rates, and RMSE across states

    Returns:
      {
        "emp": {state: rate, ...},
        "pred": {state: rate, ...},
        "rmse": float
      }
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    p = np.asarray(p_buy_mnjt, dtype=np.float64)
    st = np.asarray(s_mjt, dtype=np.int64)

    M, N, J, T = a.shape

    # Aggregate over consumers: (M,J,T)
    a_sum_mjt = a.sum(axis=1)
    p_sum_mjt = p.sum(axis=1)

    st_flat = st.reshape(-1)
    a_sum_flat = a_sum_mjt.reshape(-1)
    p_sum_flat = p_sum_mjt.reshape(-1)

    if st_flat.size == 0:
        return {"emp": {}, "pred": {}, "rmse": float("nan")}

    S_obs = int(st_flat.max()) + 1
    counts = np.bincount(st_flat, minlength=S_obs).astype(
        np.float64
    )  # #(m,j,t) in state
    den = counts * float(N)

    emp_num = np.bincount(st_flat, weights=a_sum_flat, minlength=S_obs).astype(
        np.float64
    )
    pred_num = np.bincount(st_flat, weights=p_sum_flat, minlength=S_obs).astype(
        np.float64
    )

    emp = np.full((S_obs,), np.nan, dtype=np.float64)
    pred = np.full((S_obs,), np.nan, dtype=np.float64)

    mask = den > 0.0
    emp[mask] = emp_num[mask] / den[mask]
    pred[mask] = pred_num[mask] / den[mask]

    emp_dict: dict[int, float] = {
        int(s): float(emp[s]) for s in range(S_obs) if mask[s]
    }
    pred_dict: dict[int, float] = {
        int(s): float(pred[s]) for s in range(S_obs) if mask[s]
    }

    diffs = (pred[mask] - emp[mask]) ** 2
    rmse = float(np.sqrt(np.mean(diffs))) if diffs.size else float("nan")

    return {"emp": emp_dict, "pred": pred_dict, "rmse": rmse}


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare true vs fitted constrained parameters for standard Phase-3 keys."""
    out: dict[str, dict[str, float]] = {}

    for k in PARAM_KEYS:
        if k not in theta_true or k not in theta_hat:
            raise ValueError(
                f"parameter_metrics: missing key '{k}' in theta_true/theta_hat"
            )
        true = np.asarray(theta_true[k], dtype=np.float64)
        hat = np.asarray(theta_hat[k], dtype=np.float64)

        d = (hat - true).astype(np.float64, copy=False)
        rmse = float(np.sqrt(np.mean(d * d)))

        mt = float(np.mean(true))
        mh = float(np.mean(hat))
        out[k] = {
            "rmse": rmse,
            "mean_true": mt,
            "mean_hat": mh,
            "bias": mh - mt,
        }

    return out


def evaluate_stockpiling(
    a_mnjt: np.ndarray,
    p_buy_hat_mnjt: np.ndarray,
    s_mjt: np.ndarray | None,
    theta_hat: dict[str, np.ndarray] | None,
    theta_true: dict[str, np.ndarray] | None,
    p_buy_oracle_mnjt: np.ndarray | None,
    mcmc: dict[str, Any] | None,
    eps: float,
) -> dict[str, Any]:
    """Evaluate predictive fit and optional parameter recovery / MCMC diagnostics.

    Args:
      a_mnjt: (M,N,J,T) observed purchases
      p_buy_hat_mnjt: (M,N,J,T) fitted predictive probabilities
      s_mjt: (M,J,T) price states, or None to skip by-state summary
      theta_hat: fitted parameters dict, or None
      theta_true: true parameters dict, or None
      p_buy_oracle_mnjt: oracle predictive probabilities, or None
      mcmc: optional MCMC diagnostics dict with required keys {"accept","n_saved"}
      eps: numerical clipping constant for log-loss

    Returns:
      dict with keys:
        - shape
        - models
        - deltas
        - by_price_state (optional)
        - param (optional)
        - mcmc (optional)
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    p_hat = np.asarray(p_buy_hat_mnjt, dtype=np.float64)

    M, N, J, T = (int(x) for x in a.shape)
    n_obs = int(M * N * J * T)

    models: dict[str, dict[str, float]] = {
        "baseline": baseline_metrics_from_actions(a, eps),
        "fitted": predictive_metrics_from_probs(a, p_hat, eps),
    }

    if p_buy_oracle_mnjt is not None:
        p_or = np.asarray(p_buy_oracle_mnjt, dtype=np.float64)
        models["oracle"] = predictive_metrics_from_probs(a, p_or, eps)

    base = models["baseline"]
    fit = models["fitted"]

    deltas: dict[str, dict[str, float]] = {
        "fitted_vs_baseline": {
            "delta_nll": float(base["nll_per_obs"] - fit["nll_per_obs"]),
            "delta_rmse": float(base["rmse_prob"] - fit["rmse_prob"]),
            "delta_brier": float(base["brier"] - fit["brier"]),
        }
    }
    if "oracle" in models:
        ora = models["oracle"]
        deltas["fitted_vs_oracle"] = {
            "delta_nll": float(fit["nll_per_obs"] - ora["nll_per_obs"]),
            "delta_rmse": float(fit["rmse_prob"] - ora["rmse_prob"]),
            "delta_brier": float(fit["brier"] - ora["brier"]),
        }

    out: dict[str, Any] = {
        "shape": {"M": M, "N": N, "J": J, "T": T, "n_obs": n_obs},
        "models": models,
        "deltas": deltas,
    }

    if s_mjt is not None:
        out["by_price_state"] = by_price_state_summary(a, p_hat, s_mjt)

    if theta_true is not None:
        if theta_hat is None:
            raise ValueError(
                "evaluate_stockpiling: theta_true provided but theta_hat is None"
            )
        out["param"] = parameter_metrics(theta_true, theta_hat)

    if mcmc is not None:
        if "accept" not in mcmc or "n_saved" not in mcmc:
            raise ValueError(
                "evaluate_stockpiling: mcmc must contain keys {'accept','n_saved'}"
            )
        accept_rates = mcmc["accept"]
        n_saved = mcmc["n_saved"]
        out["mcmc"] = {"n_saved": n_saved, "accept_rates": accept_rates}

    return out


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format evaluation output into a compact text report."""
    shp = eval_out["shape"]
    models = eval_out["models"]
    deltas = eval_out["deltas"]
    by_state = eval_out.get("by_price_state", None)
    params = eval_out.get("param", None)
    mcmc = eval_out.get("mcmc", None)

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    lines.append(
        f"data: M={shp['M']} N={shp['N']} J={shp['J']} T={shp['T']} | n_obs={shp['n_obs']}"
    )
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
            f"{f6(float(d['nll_per_obs']))} "
            f"{f6(float(d['rmse_prob']))} "
            f"{f4(float(d['buy_rate_emp']))} "
            f"{f4(float(d['buy_rate_pred']))}"
        )

    lines.append(row("baseline", models["baseline"]))
    lines.append(row("fitted", models["fitted"]))
    if "oracle" in models:
        lines.append(row("oracle", models["oracle"]))

    dvb = deltas["fitted_vs_baseline"]
    lines.append("")
    lines.append(
        f"gain vs baseline: Δnll={dvb['delta_nll']:.6f} | Δrmse={dvb['delta_rmse']:.6f}"
    )

    if "fitted_vs_oracle" in deltas:
        dvo = deltas["fitted_vs_oracle"]
        lines.append(
            f"fitted - oracle: Δnll={dvo['delta_nll']:.6f} | Δrmse={dvo['delta_rmse']:.6f}"
        )

    if by_state is not None:
        lines.append("")
        lines.append(f"by-price-state RMSE: {by_state['rmse']:.6f}")

    if params is not None:
        lines.append("")
        lines.append("parameter recovery:")
        for k in PARAM_KEYS:
            d = params[k]
            lines.append(
                f"  {k:<8} rmse={d['rmse']:.6f} | mean_true={d['mean_true']:.6f} | mean_hat={d['mean_hat']:.6f}"
            )

    if mcmc is not None:
        lines.append("")
        lines.append(f"mcmc: n_saved={mcmc['n_saved']}")
        acc = mcmc["accept_rates"]
        for k in sorted(acc.keys()):
            lines.append(f"  accept[{k}]={float(acc[k]):.4f}")

    return "\n".join(lines)
