from __future__ import annotations

from typing import Any

import numpy as np

PARAM_KEYS = ("beta", "alpha", "v", "fc", "u_scale")
MCMC_ACCEPT_KEYS = (
    "beta_accept",
    "alpha_accept",
    "v_accept",
    "fc_accept",
    "u_scale_accept",
)


def predictive_metrics_from_probs(
    a_mnjt: np.ndarray,
    p_buy_mnjt: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute predictive metrics from predicted buy probabilities."""
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
    """Compute baseline metrics using the empirical mean buy rate."""
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
    """Summarize empirical and predicted buy rates by observed price state."""
    a = np.asarray(a_mnjt, dtype=np.float64)
    p = np.asarray(p_buy_mnjt, dtype=np.float64)
    st = np.asarray(s_mjt, dtype=np.int64)

    _, N, _, _ = a.shape

    a_sum_mjt = a.sum(axis=1)
    p_sum_mjt = p.sum(axis=1)

    st_flat = st.reshape(-1)
    a_sum_flat = a_sum_mjt.reshape(-1)
    p_sum_flat = p_sum_mjt.reshape(-1)

    if st_flat.size == 0:
        return {"emp": {}, "pred": {}, "rmse": float("nan")}

    S_obs = int(st_flat.max()) + 1
    counts = np.bincount(st_flat, minlength=S_obs).astype(np.float64)
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

    for key in PARAM_KEYS:
        if key not in theta_true or key not in theta_hat:
            raise ValueError(
                f"parameter_metrics: missing key '{key}' in theta_true/theta_hat"
            )

        true = np.asarray(theta_true[key], dtype=np.float64)
        hat = np.asarray(theta_hat[key], dtype=np.float64)

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
    """Validate and normalize the refactored MCMC summary schema."""
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
                "evaluate_stockpiling: mcmc must contain acceptance key " f"'{key}'."
            )
        accept_rates[key] = float(mcmc[key])

    out["accept_rates"] = accept_rates

    optional_scalar_keys = ("num_chunks", "joint_logpost_last")
    for key in optional_scalar_keys:
        if key in mcmc:
            out[key] = (
                float(mcmc[key]) if key == "joint_logpost_last" else int(mcmc[key])
            )

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
    """Evaluate predictive fit and optional parameter recovery / MCMC diagnostics."""
    a = np.asarray(a_mnjt, dtype=np.float64)
    p_hat = np.asarray(p_buy_hat_mnjt, dtype=np.float64)

    M, N, J, T = (int(x) for x in a.shape)
    n_obs = int(M * N * J * T)

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
        out["mcmc"] = _normalize_mcmc_summary(mcmc)

    return out


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format evaluation output into a compact text report."""
    shape = eval_out["shape"]
    models = eval_out["models"]
    deltas = eval_out["deltas"]
    by_state = eval_out.get("by_price_state")
    params = eval_out.get("param")
    mcmc = eval_out.get("mcmc")

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []
    lines.append(
        f"data: M={shape['M']} N={shape['N']} J={shape['J']} T={shape['T']} | n_obs={shape['n_obs']}"
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

    def row(tag: str, metrics: dict[str, Any]) -> str:
        return (
            f"{tag:<10}"
            f"{f6(float(metrics['nll_per_obs']))} "
            f"{f6(float(metrics['rmse_prob']))} "
            f"{f4(float(metrics['buy_rate_emp']))} "
            f"{f4(float(metrics['buy_rate_pred']))}"
        )

    lines.append(row("baseline", models["baseline"]))
    lines.append(row("fitted", models["fitted"]))
    if "oracle" in models:
        lines.append(row("oracle", models["oracle"]))

    dvb = deltas["fitted_vs_baseline"]
    lines.append("")
    lines.append(
        f"gain vs baseline: Δnll={dvb['delta_nll']:.6f} | Δrmse={dvb['delta_rmse']:.6f} | Δbrier={dvb['delta_brier']:.6f}"
    )

    if "fitted_vs_oracle" in deltas:
        dvo = deltas["fitted_vs_oracle"]
        lines.append(
            f"fitted - oracle: Δnll={dvo['delta_nll']:.6f} | Δrmse={dvo['delta_rmse']:.6f} | Δbrier={dvo['delta_brier']:.6f}"
        )

    if by_state is not None:
        lines.append("")
        lines.append(f"by-price-state RMSE: {by_state['rmse']:.6f}")

    if params is not None:
        lines.append("")
        lines.append("parameter recovery:")
        for key in PARAM_KEYS:
            d = params[key]
            lines.append(
                f"  {key:<8} rmse={d['rmse']:.6f} | mean_true={d['mean_true']:.6f} | mean_hat={d['mean_hat']:.6f} | bias={d['bias']:.6f}"
            )

    if mcmc is not None:
        lines.append("")
        line = f"mcmc: n_saved={mcmc['n_saved']}"
        if "num_chunks" in mcmc:
            line += f" | num_chunks={mcmc['num_chunks']}"
        if "joint_logpost_last" in mcmc:
            line += f" | joint_logpost_last={float(mcmc['joint_logpost_last']):.6f}"
        lines.append(line)
        for key in MCMC_ACCEPT_KEYS:
            lines.append(f"  {key}={float(mcmc['accept_rates'][key]):.4f}")

    return "\n".join(lines)
