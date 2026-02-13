# ching/stockpiling_evaluate.py
#
# Lightweight evaluation utilities for the stockpiling model (multi-product).
#
# Design goals:
# - Pure NumPy (no TensorFlow).
# - No DP / posterior / filtering mechanics inside evaluation.
# - Standardized output schema: eval_out["models"] holds comparable metrics for
#   baseline / fitted / (optional) oracle; eval_out contains precomputed deltas,
#   optional by-price-state summaries, optional parameter recovery, and optional
#   MCMC acceptance rates (schema-driven, no inferred proposed counts).

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Validation helpers (local)
# =============================================================================


def _validate_panel_shapes(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    p_state_mjt: np.ndarray | None,
) -> tuple[int, int, int, int, int]:
    a = np.asarray(a_mnjt)
    p = np.asarray(p_buy_mnjt)

    if a.ndim != 4:
        raise ValueError(f"a_mnjt must have shape (M,N,J,T); got {a.shape}")
    if p.ndim != 4:
        raise ValueError(f"p_buy_mnjt must have shape (M,N,J,T); got {p.shape}")
    if a.shape != p.shape:
        raise ValueError(
            f"a_mnjt shape {a.shape} must match p_buy_mnjt shape {p.shape}"
        )

    M, N, J, T = (int(x) for x in a.shape)
    n_obs = int(M * N * J * T)

    if p_state_mjt is not None:
        st = np.asarray(p_state_mjt)
        if st.shape != (M, J, T):
            raise ValueError(
                f"p_state_mjt must have shape (M,J,T)={(M, J, T)}, got {st.shape}"
            )

    return M, N, J, T, n_obs


# =============================================================================
# Core predictive metrics
# =============================================================================


def predictive_metrics_from_probs(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Compute predictive metrics given predicted buy probabilities.

    Inputs:
      a_mnjt: (M,N,J,T) actions in {0,1}
      p_buy_mnjt: (M,N,J,T) predicted probabilities in [0,1]

    Returns:
      dict with:
        - nll_per_obs
        - brier
        - rmse_prob
        - buy_rate_emp
        - buy_rate_pred
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    p = np.asarray(p_buy_mnjt, dtype=np.float64)

    # Shape validation done by evaluate_stockpiling; keep this function minimal.

    p = np.clip(p, eps, 1.0 - eps)

    nll = -np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p))
    brier = float(np.mean((p - a) ** 2))
    rmse = float(np.sqrt(brier))

    buy_rate_emp = float(np.mean(a))
    buy_rate_pred = float(np.mean(p))

    return {
        "nll_per_obs": float(nll),
        "brier": float(brier),
        "rmse_prob": float(rmse),
        "buy_rate_emp": float(buy_rate_emp),
        "buy_rate_pred": float(buy_rate_pred),
    }


def baseline_metrics_from_actions(
    a_mnjt: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Baseline model: constant probability p0 = empirical buy rate.

    Returns same metric keys as predictive_metrics_from_probs, plus "p0".
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
    p_state_mjt: np.ndarray,
) -> dict[str, Any]:
    """
    By-price-state buy rates, aggregated over (m,j,t) and averaged over consumers.

    Returns:
      {
        "emp": {state: rate, ...},
        "pred": {state: rate, ...},
        "rmse": float   # equal weight per state among states observed
      }
    """
    a = np.asarray(a_mnjt, dtype=np.float64)
    p = np.asarray(p_buy_mnjt, dtype=np.float64)
    st = np.asarray(p_state_mjt, dtype=np.int64)

    M, N, J, T = a.shape

    # Aggregate over consumers first: (M,J,T)
    a_sum_mjt = a.sum(axis=1)  # (M,J,T)
    p_sum_mjt = p.sum(axis=1)  # (M,J,T)

    st_flat = st.reshape(-1)
    a_sum_flat = a_sum_mjt.reshape(-1)
    p_sum_flat = p_sum_mjt.reshape(-1)

    if st_flat.size == 0:
        return {"emp": {}, "pred": {}, "rmse": float("nan")}

    S_obs = int(st_flat.max()) + 1
    counts = np.bincount(st_flat, minlength=S_obs).astype(
        np.float64
    )  # (#(m,j,t) in state s)
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


# =============================================================================
# Parameter recovery
# =============================================================================


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
    *,
    order: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compare true vs fitted constrained parameters.

    For each key present in both dictionaries:
      - rmse
      - mean_true
      - mean_hat
      - bias = mean_hat - mean_true

    Keys are reported in a deterministic order (default: beta, alpha, v, fc, lambda, u_scale),
    filtered to those present.
    """
    if order is None:
        order = ["beta", "alpha", "v", "fc", "lambda", "u_scale"]

    out: dict[str, dict[str, float]] = {}

    def _rmse(t: np.ndarray, h: np.ndarray) -> float:
        d = (h - t).astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(d * d)))

    for k in order:
        if k not in theta_true or k not in theta_hat:
            continue
        true = np.asarray(theta_true[k], dtype=np.float64)
        hat = np.asarray(theta_hat[k], dtype=np.float64)
        mt = float(np.mean(true))
        mh = float(np.mean(hat))
        out[k] = {
            "rmse": _rmse(true, hat),
            "mean_true": mt,
            "mean_hat": mh,
            "bias": mh - mt,
        }

    return out


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_stockpiling(
    a_mnjt: np.ndarray,
    p_buy_hat_mnjt: np.ndarray,
    p_state_mjt: np.ndarray | None = None,
    theta_hat: dict[str, np.ndarray] | None = None,
    theta_true: dict[str, np.ndarray] | None = None,
    p_buy_oracle_mnjt: np.ndarray | None = None,
    mcmc: dict[str, Any] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Evaluate predictive fit and (optionally) parameter recovery and MCMC diagnostics.

    Output schema:
      {
        "shape": {...},
        "models": {"baseline":..., "fitted":..., "oracle":... (optional)},
        "deltas": {"fitted_vs_baseline":..., "fitted_vs_oracle":... (optional)},
        "by_price_state": {...} (optional),
        "param": {...} (optional),
        "mcmc": {"n_saved": int|None, "accept_rates": dict[str,float]} (optional)
      }
    """
    M, N, J, T, n_obs = _validate_panel_shapes(a_mnjt, p_buy_hat_mnjt, p_state_mjt)
    a = np.asarray(a_mnjt, dtype=np.float64)

    models: dict[str, dict[str, float]] = {}
    models["baseline"] = baseline_metrics_from_actions(a, eps=eps)
    models["fitted"] = predictive_metrics_from_probs(a, p_buy_hat_mnjt, eps=eps)

    if p_buy_oracle_mnjt is not None:
        p_or = np.asarray(p_buy_oracle_mnjt)
        if p_or.shape != a.shape:
            raise ValueError(
                f"p_buy_oracle_mnjt shape {p_or.shape} must match a_mnjt shape {a.shape}"
            )
        models["oracle"] = predictive_metrics_from_probs(a, p_or, eps=eps)

    deltas: dict[str, dict[str, float]] = {}
    base = models["baseline"]
    fit = models["fitted"]
    deltas["fitted_vs_baseline"] = {
        "delta_nll": float(base["nll_per_obs"] - fit["nll_per_obs"]),
        "delta_rmse": float(base["rmse_prob"] - fit["rmse_prob"]),
        "delta_brier": float(base["brier"] - fit["brier"]),
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

    if p_state_mjt is not None:
        out["by_price_state"] = by_price_state_summary(a, p_buy_hat_mnjt, p_state_mjt)

    if theta_true is not None:
        if theta_hat is None:
            raise ValueError("theta_true provided but theta_hat is None")
        out["param"] = parameter_metrics(theta_true, theta_hat)

    if mcmc is not None:
        # Estimator now provides accept as a flat dict of rates; keep schema simple.
        accept_rates = mcmc.get("accept", {})
        n_saved = mcmc.get("n_saved", None)
        out["mcmc"] = {"n_saved": n_saved, "accept_rates": accept_rates}

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(
    eval_out: dict[str, Any],
    param_order: list[str] | None = None,
) -> str:
    """
    Format evaluation output into a reviewer-friendly text report.

    Expects eval_out from evaluate_stockpiling(...).
    """
    shp = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    by_state = eval_out.get("by_price_state", None)
    params = eval_out.get("param", None)
    mcmc = eval_out.get("mcmc", None)

    M = shp.get("M")
    N = shp.get("N")
    J = shp.get("J")
    T = shp.get("T")
    n_obs = shp.get("n_obs")

    baseline = models.get("baseline", {})
    fitted = models.get("fitted", {})
    oracle = models.get("oracle", None)

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    if M is not None and N is not None and J is not None and T is not None:
        lines.append(f"data: M={M} N={N} J={J} T={T} | n_obs={n_obs}")
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

    if baseline:
        lines.append(row("baseline", baseline))
    if fitted:
        lines.append(row("fitted", fitted))
    if oracle is not None:
        lines.append(row("oracle", oracle))

    # Deltas
    dvb = deltas.get("fitted_vs_baseline", None)
    if dvb is not None:
        lines.append("")
        lines.append(
            f"gain vs baseline: Δnll={dvb['delta_nll']:.6f} | Δrmse={dvb['delta_rmse']:.6f}"
        )

    dvo = deltas.get("fitted_vs_oracle", None)
    if dvo is not None:
        lines.append(
            f"fitted - oracle: Δnll={dvo['delta_nll']:.6f} | Δrmse={dvo['delta_rmse']:.6f}"
        )

    # By price state
    if isinstance(by_state, dict):
        emp = by_state.get("emp", {})
        pred = by_state.get("pred", {})
        if isinstance(emp, dict) and emp:
            lines.append("")
            lines.append("buy rate by price state")
            st_header = f"{'state':<8}{'emp':>10} {'pred':>10} {'diff':>10}"
            lines.append(st_header)
            lines.append("-" * len(st_header))
            for s in sorted(emp.keys()):
                e = float(emp[s])
                p = float(pred.get(s, float("nan")))
                diff = p - e
                lines.append(f"{str(s):<8}{f6(e)} {f6(p)} {f6(diff)}")
            lines.append(
                f"rmse across states: {float(by_state.get('rmse', float('nan'))):.6f}"
            )

    # Parameter recovery
    if params is not None and isinstance(params, dict) and params:
        if param_order is None:
            param_order = ["beta", "alpha", "v", "fc", "lambda", "u_scale"]
        present = [k for k in param_order if k in params]
        worst = sorted(present, key=lambda k: float(params[k]["rmse"]), reverse=True)

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
            lines.append(
                f"{k:<10}"
                f"{f6(float(pk['rmse']))} "
                f"{f6(float(pk['mean_true']))} "
                f"{f6(float(pk['mean_hat']))} "
                f"{f6(float(pk['bias']))}"
            )

    # MCMC acceptance (rates only, schema-driven)
    if isinstance(mcmc, dict):
        rates = mcmc.get("accept_rates", {})
        n_saved = mcmc.get("n_saved", None)
        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("mcmc acceptance (elementwise rates)")
            if n_saved is not None:
                lines.append(f"n_saved: {n_saved}")

            a_header = f"{'block':<10}{'rate':>10}"
            lines.append(a_header)
            lines.append("-" * len(a_header))

            order = ["beta", "alpha", "v", "fc", "lambda", "u_scale"]
            vals: list[float] = []
            for k in order:
                if k not in rates:
                    continue
                r = float(rates[k])
                vals.append(r)
                lines.append(f"{k:<10}{r:>10.4f}")
            if vals:
                lines.append(f"mean rate: {float(np.mean(vals)):.4f}")

    return "\n".join(lines)
