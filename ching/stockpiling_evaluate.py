"""Evaluation helpers for the Ching-style stockpiling simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_PARAM_KEYS = ("beta", "alpha", "v", "fc")
MCMC_ACCEPT_KEYS = (
    "beta_accept",
    "alpha_accept",
    "v_accept",
    "fc_accept",
    "u_scale_accept",
)

__all__ = [
    "predictive_metrics_from_probs",
    "baseline_metrics_from_actions",
    "by_price_state_summary",
    "parameter_metrics",
    "evaluate_stockpiling",
    "format_evaluation_summary",
]


def predictive_metrics_from_probs(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute predictive metrics from predicted buy probabilities."""
    p = np.clip(p_buy_mnjt, eps, 1.0 - eps)

    nll = -np.mean(a_mnjt * np.log(p) + (1.0 - a_mnjt) * np.log(1.0 - p))
    brier = float(np.mean((p - a_mnjt) ** 2))
    rmse = float(np.sqrt(brier))

    return {
        "nll_per_obs": float(nll),
        "brier": float(brier),
        "rmse_prob": rmse,
        "buy_rate_emp": float(np.mean(a_mnjt)),
        "buy_rate_pred": float(np.mean(p)),
    }


def baseline_metrics_from_actions(
    a_mnjt: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute baseline metrics using the empirical mean buy rate."""
    buy_rate_emp = float(np.mean(a_mnjt))
    p0 = float(np.clip(buy_rate_emp, eps, 1.0 - eps))

    nll = -np.mean(a_mnjt * np.log(p0) + (1.0 - a_mnjt) * np.log(1.0 - p0))
    brier = float(np.mean((p0 - a_mnjt) ** 2))
    rmse = float(np.sqrt(brier))

    return {
        "p0": p0,
        "nll_per_obs": float(nll),
        "brier": float(brier),
        "rmse_prob": rmse,
        "buy_rate_emp": buy_rate_emp,
        "buy_rate_pred": p0,
    }


def by_price_state_summary(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    s_mjt: np.ndarray,
) -> dict[str, Any]:
    """Summarize empirical and predicted buy rates by observed price state."""
    _, n_consumers, _, _ = a_mnjt.shape

    a_sum_mjt = a_mnjt.sum(axis=1)
    p_sum_mjt = p_buy_mnjt.sum(axis=1)

    state_flat = s_mjt.reshape(-1)
    a_sum_flat = a_sum_mjt.reshape(-1)
    p_sum_flat = p_sum_mjt.reshape(-1)

    if state_flat.size == 0:
        return {"emp": {}, "pred": {}, "rmse": float("nan")}

    n_states = int(state_flat.max()) + 1
    counts = np.bincount(state_flat, minlength=n_states).astype(np.float64)
    denom = counts * float(n_consumers)

    emp_num = np.bincount(
        state_flat,
        weights=a_sum_flat,
        minlength=n_states,
    ).astype(np.float64)
    pred_num = np.bincount(
        state_flat,
        weights=p_sum_flat,
        minlength=n_states,
    ).astype(np.float64)

    emp = np.full(n_states, np.nan, dtype=np.float64)
    pred = np.full(n_states, np.nan, dtype=np.float64)

    mask = denom > 0.0
    emp[mask] = emp_num[mask] / denom[mask]
    pred[mask] = pred_num[mask] / denom[mask]

    emp_dict = {int(s): float(emp[s]) for s in range(n_states) if mask[s]}
    pred_dict = {int(s): float(pred[s]) for s in range(n_states) if mask[s]}

    diffs = (pred[mask] - emp[mask]) ** 2
    rmse = float(np.sqrt(np.mean(diffs))) if diffs.size else float("nan")

    return {"emp": emp_dict, "pred": pred_dict, "rmse": rmse}


def _parameter_keys(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> list[str]:
    """Return the parameter keys to compare."""
    keys: list[str] = [
        key for key in DEFAULT_PARAM_KEYS if key in theta_true and key in theta_hat
    ]
    extra_keys = sorted(
        key for key in theta_true if key in theta_hat and key not in DEFAULT_PARAM_KEYS
    )
    keys.extend(extra_keys)

    if not keys:
        raise ValueError("parameter_metrics: theta_true and theta_hat share no keys.")
    return keys


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare true and fitted constrained parameters."""
    out: dict[str, dict[str, float]] = {}

    for key in _parameter_keys(theta_true, theta_hat):
        true = np.asarray(theta_true[key], dtype=np.float64)
        hat = np.asarray(theta_hat[key], dtype=np.float64)

        if true.shape != hat.shape:
            raise ValueError(
                f"parameter_metrics: shape mismatch for '{key}': "
                f"true {true.shape} vs hat {hat.shape}."
            )

        diff = hat - true
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mean_true = float(np.mean(true))
        mean_hat = float(np.mean(hat))

        out[key] = {
            "rmse": rmse,
            "mean_true": mean_true,
            "mean_hat": mean_hat,
            "bias": mean_hat - mean_true,
        }

    return out


def _normalize_mcmc_summary(mcmc: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the MCMC summary schema."""
    if "n_saved" not in mcmc:
        raise ValueError("evaluate_stockpiling: mcmc must contain key 'n_saved'.")

    out: dict[str, Any] = {"n_saved": int(mcmc["n_saved"])}
    if out["n_saved"] <= 0:
        raise ValueError(
            f"evaluate_stockpiling: mcmc['n_saved'] must be > 0; got {out['n_saved']}."
        )

    accept_rates: dict[str, float] = {}
    for key in MCMC_ACCEPT_KEYS:
        if key not in mcmc:
            raise ValueError(
                f"evaluate_stockpiling: mcmc must contain acceptance key '{key}'."
            )
        accept_rates[key] = float(mcmc[key])
    out["accept_rates"] = accept_rates

    if "num_chunks" in mcmc:
        out["num_chunks"] = int(mcmc["num_chunks"])
    if "joint_logpost_last" in mcmc:
        out["joint_logpost_last"] = float(mcmc["joint_logpost_last"])

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
    """Evaluate predictive fit and optional parameter recovery."""
    a = np.asarray(a_mnjt, dtype=np.float64)
    p_hat = np.asarray(p_buy_hat_mnjt, dtype=np.float64)

    m_size, n_size, j_size, t_size = (int(x) for x in a.shape)
    n_obs = int(m_size * n_size * j_size * t_size)

    models: dict[str, dict[str, float]] = {
        "baseline": baseline_metrics_from_actions(a, eps),
        "fitted": predictive_metrics_from_probs(a, p_hat, eps),
    }

    if p_buy_oracle_mnjt is not None:
        p_oracle = np.asarray(p_buy_oracle_mnjt, dtype=np.float64)
        models["oracle"] = predictive_metrics_from_probs(a, p_oracle, eps)

    baseline = models["baseline"]
    fitted = models["fitted"]

    deltas: dict[str, dict[str, float]] = {
        "fitted_vs_baseline": {
            "delta_nll": float(baseline["nll_per_obs"] - fitted["nll_per_obs"]),
            "delta_rmse": float(baseline["rmse_prob"] - fitted["rmse_prob"]),
            "delta_brier": float(baseline["brier"] - fitted["brier"]),
        }
    }

    if "oracle" in models:
        oracle = models["oracle"]
        deltas["fitted_vs_oracle"] = {
            "delta_nll": float(fitted["nll_per_obs"] - oracle["nll_per_obs"]),
            "delta_rmse": float(fitted["rmse_prob"] - oracle["rmse_prob"]),
            "delta_brier": float(fitted["brier"] - oracle["brier"]),
        }

    out: dict[str, Any] = {
        "shape": {
            "M": m_size,
            "N": n_size,
            "J": j_size,
            "T": t_size,
            "n_obs": n_obs,
        },
        "models": models,
        "deltas": deltas,
    }

    if s_mjt is not None:
        states = np.asarray(s_mjt, dtype=np.int64)
        out["by_price_state"] = by_price_state_summary(a, p_hat, states)

    if theta_true is not None:
        if theta_hat is None:
            raise ValueError(
                "evaluate_stockpiling: theta_true provided but theta_hat is None."
            )
        param = parameter_metrics(theta_true, theta_hat)
        out["param"] = param
        out["param_keys"] = list(param.keys())

    if mcmc is not None:
        out["mcmc"] = _normalize_mcmc_summary(mcmc)

    return out


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format evaluation output into a compact text report."""
    shape = eval_out["shape"]
    models = eval_out["models"]
    deltas = eval_out["deltas"]
    by_state = eval_out.get("by_price_state")
    params = eval_out.get("param")
    param_keys = eval_out.get("param_keys", [])
    mcmc = eval_out.get("mcmc")

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    def model_row(name: str, metrics: dict[str, Any]) -> str:
        return (
            f"{name:<10}"
            f"{f6(float(metrics['nll_per_obs']))} "
            f"{f6(float(metrics['rmse_prob']))} "
            f"{f4(float(metrics['buy_rate_emp']))} "
            f"{f4(float(metrics['buy_rate_pred']))}"
        )

    lines: list[str] = [
        (
            f"data: M={shape['M']} N={shape['N']} "
            f"J={shape['J']} T={shape['T']} | n_obs={shape['n_obs']}"
        ),
        "",
    ]

    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
        f"{'rmse':>10} "
        f"{'buy_emp':>8} "
        f"{'buy_pred':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(model_row("baseline", models["baseline"]))
    lines.append(model_row("fitted", models["fitted"]))
    if "oracle" in models:
        lines.append(model_row("oracle", models["oracle"]))

    dvb = deltas["fitted_vs_baseline"]
    lines.append("")
    lines.append(
        "gain vs baseline: "
        f"Δnll={dvb['delta_nll']:.6f} | "
        f"Δrmse={dvb['delta_rmse']:.6f} | "
        f"Δbrier={dvb['delta_brier']:.6f}"
    )

    if "fitted_vs_oracle" in deltas:
        dvo = deltas["fitted_vs_oracle"]
        lines.append(
            "fitted - oracle: "
            f"Δnll={dvo['delta_nll']:.6f} | "
            f"Δrmse={dvo['delta_rmse']:.6f} | "
            f"Δbrier={dvo['delta_brier']:.6f}"
        )

    if by_state is not None:
        lines.append("")
        lines.append(f"by-price-state RMSE: {by_state['rmse']:.6f}")

    if params is not None:
        lines.append("")
        lines.append("parameter recovery:")
        for key in param_keys:
            stats = params[key]
            lines.append(
                f"  {key:<8} "
                f"rmse={stats['rmse']:.6f} | "
                f"mean_true={stats['mean_true']:.6f} | "
                f"mean_hat={stats['mean_hat']:.6f} | "
                f"bias={stats['bias']:.6f}"
            )

    if mcmc is not None:
        lines.append("")
        summary_line = f"mcmc: n_saved={mcmc['n_saved']}"
        if "num_chunks" in mcmc:
            summary_line += f" | num_chunks={mcmc['num_chunks']}"
        if "joint_logpost_last" in mcmc:
            summary_line += f" | joint_logpost_last={mcmc['joint_logpost_last']:.6f}"
        lines.append(summary_line)

        for key in MCMC_ACCEPT_KEYS:
            lines.append(f"  {key}={mcmc['accept_rates'][key]:.4f}")

    return "\n".join(lines)
