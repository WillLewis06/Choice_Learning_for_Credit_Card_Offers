"""Evaluate Bonus Q2 predictions and parameter recovery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _pack_choice_metrics(
    nll: float,
    acc: float,
    p_true: np.ndarray,
    out_emp: float,
    out_pred: float,
) -> dict[str, float]:
    """Pack predictive metrics into a standard dictionary."""
    return {
        "nll": float(nll),
        "acc": float(acc),
        "p_true": float(np.mean(p_true)),
        "out_emp": float(out_emp),
        "out_pred": float(out_pred),
    }


def choice_metrics_from_probs(
    y_mit: np.ndarray,
    p_choice_mntc: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute predictive metrics from per-observation probabilities."""
    y = np.asarray(y_mit, dtype=np.int64)
    p = np.asarray(p_choice_mntc, dtype=np.float64)

    p_true = np.take_along_axis(p, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)

    return _pack_choice_metrics(
        nll=-np.mean(np.log(p_true_clip)),
        acc=np.mean(np.argmax(p, axis=3) == y),
        p_true=p_true,
        out_emp=np.mean(y == 0),
        out_pred=np.mean(p[..., 0]),
    )


def choice_metrics_from_market_probs(
    y_mit: np.ndarray,
    p_mjc: np.ndarray,
    eps: float,
) -> dict[str, float]:
    """Compute predictive metrics when probabilities vary only by market."""
    y = np.asarray(y_mit, dtype=np.int64)
    p = np.asarray(p_mjc, dtype=np.float64)

    p4 = p[:, None, None, :]
    p_true = np.take_along_axis(p4, y[..., None], axis=3)[..., 0]
    p_true_clip = np.clip(p_true, eps, 1.0)
    c_hat_m = np.argmax(p, axis=1)

    return _pack_choice_metrics(
        nll=-np.mean(np.log(p_true_clip)),
        acc=np.mean(y == c_hat_m[:, None, None]),
        p_true=p_true,
        out_emp=np.mean(y == 0),
        out_pred=np.mean(p[:, 0]),
    )


def delta_only_baseline_probs(delta_mj: np.ndarray) -> np.ndarray:
    """Compute delta-only baseline probabilities for each market."""
    delta = np.asarray(delta_mj, dtype=np.float64)

    max_u = np.maximum(0.0, np.max(delta, axis=1))
    exp_in = np.exp(delta - max_u[:, None])
    exp_out = np.exp(-max_u)
    den = exp_out + np.sum(exp_in, axis=1)

    p_out = (exp_out / den)[:, None]
    p_in = exp_in / den[:, None]
    return np.concatenate([p_out, p_in], axis=1)


def _rmse(true: np.ndarray, hat: np.ndarray) -> float:
    """Return the root mean squared error."""
    diff = np.asarray(hat, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _prepare_parameter_arrays(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build aligned true and fitted parameter arrays for recovery analysis."""
    beta_intercept_true = np.asarray(theta_true["beta_intercept_j"], dtype=np.float64)
    beta_intercept_hat = np.asarray(theta_hat["beta_intercept_j"], dtype=np.float64)

    beta_habit_true = np.asarray(theta_true["beta_habit_j"], dtype=np.float64)
    beta_habit_hat = np.asarray(theta_hat["beta_habit_j"], dtype=np.float64)

    beta_peer_true = np.asarray(theta_true["beta_peer_j"], dtype=np.float64)
    beta_peer_hat = np.asarray(theta_hat["beta_peer_j"], dtype=np.float64)

    beta_weekend_true = np.asarray(theta_true["beta_weekend_jw"], dtype=np.float64)
    beta_weekend_hat = np.asarray(theta_hat["beta_weekend_jw"], dtype=np.float64)
    weekend_lift_true = beta_weekend_true[:, 1] - beta_weekend_true[:, 0]
    weekend_lift_hat = beta_weekend_hat[:, 1] - beta_weekend_hat[:, 0]

    a_true = np.asarray(theta_true["a_m"], dtype=np.float64)
    a_hat = np.asarray(theta_hat["a_m"], dtype=np.float64)

    b_true = np.asarray(theta_true["b_m"], dtype=np.float64)
    b_hat = np.asarray(theta_hat["b_m"], dtype=np.float64)

    return {
        "beta_intercept_j": (beta_intercept_true, beta_intercept_hat),
        "beta_habit_j": (beta_habit_true, beta_habit_hat),
        "beta_peer_j": (beta_peer_true, beta_peer_hat),
        "weekend_lift_j": (weekend_lift_true, weekend_lift_hat),
        "a_m": (a_true, a_hat),
        "b_m": (b_true, b_hat),
    }


def parameter_recovery_mean_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute mean-level parameter recovery summaries."""
    arrays = _prepare_parameter_arrays(theta_true=theta_true, theta_hat=theta_hat)

    out: dict[str, dict[str, float]] = {}
    for name, (true_arr, hat_arr) in arrays.items():
        mean_true = float(np.mean(true_arr))
        mean_hat = float(np.mean(hat_arr))
        out[name] = {
            "rmse": _rmse(true_arr, hat_arr),
            "mean_true": mean_true,
            "mean_hat": mean_hat,
            "bias": mean_hat - mean_true,
        }
    return out


def parameter_recovery_dispersion_stats(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute dispersion-level recovery summaries."""
    arrays = _prepare_parameter_arrays(theta_true=theta_true, theta_hat=theta_hat)

    out: dict[str, dict[str, float]] = {}
    for name in ("weekend_lift_j", "a_m", "b_m"):
        true_arr, hat_arr = arrays[name]
        std_true = float(np.std(true_arr))
        std_hat = float(np.std(hat_arr))
        out[name] = {
            "rmse": _rmse(true_arr, hat_arr),
            "std_true": std_true,
            "std_hat": std_hat,
            "bias_std": std_hat - std_true,
        }
    return out


def _metric_delta(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Compute NLL and accuracy differences between two metric dictionaries."""
    return {
        "delta_nll": float(left["nll"] - right["nll"]),
        "delta_acc": float(left["acc"] - right["acc"]),
    }


def chain_summary_from_chunk_summaries(
    chunk_summaries: Sequence[Any],
    n_saved: int | None,
) -> dict[str, Any]:
    """Build a compact chain summary from chunk-level diagnostics."""
    if len(chunk_summaries) == 0:
        return {
            "n_saved": n_saved,
            "num_chunks": 0,
            "accept_rates": {},
            "final_joint_logpost": None,
        }

    field_map = {
        "beta_intercept": "beta_intercept_accept_mean",
        "beta_habit": "beta_habit_accept_mean",
        "beta_peer": "beta_peer_accept_mean",
        "beta_weekend": "beta_weekend_accept_mean",
        "a_m": "a_accept_mean",
        "b_m": "b_accept_mean",
    }

    accept_rates = {
        block: float(np.mean([getattr(summary, field) for summary in chunk_summaries]))
        for block, field in field_map.items()
    }

    return {
        "n_saved": n_saved,
        "num_chunks": int(len(chunk_summaries)),
        "accept_rates": accept_rates,
        "final_joint_logpost": float(chunk_summaries[-1].joint_logpost_last),
    }


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
    """Evaluate fitted Bonus Q2 predictions and optional recovery statistics."""
    y = np.asarray(y_mit, dtype=np.int64)
    delta = np.asarray(delta_mj, dtype=np.float64)
    p_fit = np.asarray(p_choice_hat_mntc, dtype=np.float64)

    num_markets, num_consumers, num_periods = (int(v) for v in y.shape)
    num_products = int(delta.shape[1])

    p_base_mjc = delta_only_baseline_probs(delta)
    baseline_metrics = choice_metrics_from_market_probs(y, p_base_mjc, eps)
    fitted_metrics = choice_metrics_from_probs(y, p_fit, eps)

    models: dict[str, dict[str, float]] = {
        "baseline": baseline_metrics,
        "fitted": fitted_metrics,
    }

    if p_choice_oracle_mntc is not None:
        p_oracle = np.asarray(p_choice_oracle_mntc, dtype=np.float64)
        models["oracle"] = choice_metrics_from_probs(y, p_oracle, eps)

    deltas: dict[str, dict[str, float]] = {
        "fitted_minus_baseline": _metric_delta(
            left=models["fitted"],
            right=models["baseline"],
        )
    }
    if "oracle" in models:
        deltas["fitted_minus_oracle"] = _metric_delta(
            left=models["fitted"],
            right=models["oracle"],
        )

    out: dict[str, Any] = {
        "shape": {
            "M": num_markets,
            "N": num_consumers,
            "T": num_periods,
            "J": num_products,
            "n_obs": int(num_markets * num_consumers * num_periods),
        },
        "models": models,
        "deltas": deltas,
    }

    if theta_true is not None and theta_hat is not None:
        out["param"] = {
            "mean_stats": parameter_recovery_mean_stats(
                theta_true=theta_true,
                theta_hat=theta_hat,
            ),
            "dispersion_stats": parameter_recovery_dispersion_stats(
                theta_true=theta_true,
                theta_hat=theta_hat,
            ),
        }

    if chunk_summaries is not None:
        out["chain"] = chain_summary_from_chunk_summaries(
            chunk_summaries=chunk_summaries,
            n_saved=n_saved,
        )

    return out


def _format_metric_table(models: dict[str, dict[str, float]]) -> list[str]:
    """Format the predictive-metrics table."""

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    def row(name: str, metric: dict[str, float]) -> str:
        return (
            f"{name:<10}"
            f"{f6(metric['nll'])} "
            f"{f6(metric['acc'])} "
            f"{f6(metric['p_true'])} "
            f"{f4(metric['out_emp'])} "
            f"{f4(metric['out_pred'])}"
        )

    lines = [
        f"{'model':<10}{'nll':>10} {'acc':>10} {'p_true':>10} {'out_emp':>8} {'out_pred':>8}",
        "-" * 60,
    ]
    for name in ("baseline", "fitted", "oracle"):
        if name in models:
            lines.append(row(name, models[name]))
    return lines


def _format_delta_section(deltas: dict[str, dict[str, float]]) -> list[str]:
    """Format the metric-delta section."""
    lines: list[str] = []
    if "fitted_minus_baseline" in deltas:
        d = deltas["fitted_minus_baseline"]
        lines.append(
            f"fitted - baseline: Δnll={d['delta_nll']:.6f} | Δacc={d['delta_acc']:.6f}"
        )
    if "fitted_minus_oracle" in deltas:
        d = deltas["fitted_minus_oracle"]
        lines.append(
            f"fitted - oracle:   Δnll={d['delta_nll']:.6f} | Δacc={d['delta_acc']:.6f}"
        )
    return lines


def _format_parameter_section(params: dict[str, Any]) -> list[str]:
    """Format the parameter-recovery section."""
    lines: list[str] = []

    mean_stats = params.get("mean_stats", {})
    if mean_stats:
        lines.extend(
            [
                "parameter recovery (mean-level)",
                f"{'param':<18}{'rmse':>10} {'mean_true':>10} {'mean_hat':>10} {'bias':>10}",
                "-" * 60,
            ]
        )
        for name, stat in mean_stats.items():
            lines.append(
                f"{name:<18}"
                f"{stat['rmse']:>10.6f} "
                f"{stat['mean_true']:>10.6f} "
                f"{stat['mean_hat']:>10.6f} "
                f"{stat['bias']:>10.6f}"
            )

    disp_stats = params.get("dispersion_stats", {})
    if disp_stats:
        if lines:
            lines.append("")
        lines.extend(
            [
                "parameter recovery (dispersion)",
                f"{'param':<18}{'rmse':>10} {'std_true':>10} {'std_hat':>10} {'bias_std':>10}",
                "-" * 60,
            ]
        )
        for name, stat in disp_stats.items():
            lines.append(
                f"{name:<18}"
                f"{stat['rmse']:>10.6f} "
                f"{stat['std_true']:>10.6f} "
                f"{stat['std_hat']:>10.6f} "
                f"{stat['bias_std']:>10.6f}"
            )

    return lines


def _format_chain_section(chain: dict[str, Any]) -> list[str]:
    """Format the compact chain-summary section."""
    rates = chain.get("accept_rates", {})
    if not rates:
        return []

    lines = ["chain summary"]
    if chain.get("n_saved") is not None:
        lines.append(f"n_saved: {int(chain['n_saved'])}")
    if chain.get("num_chunks") is not None:
        lines.append(f"num_chunks: {int(chain['num_chunks'])}")
    if chain.get("final_joint_logpost") is not None:
        lines.append(f"final_joint_logpost: {float(chain['final_joint_logpost']):.6f}")

    lines.append("")
    lines.append("acceptance (mean block rates)")
    lines.append(f"{'block':<16}{'rate':>10}")
    lines.append("-" * 26)
    for name in sorted(rates):
        lines.append(f"{name:<16}{float(rates[name]):>10.4f}")
    return lines


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """Format evaluation output into a compact multi-line summary."""
    shape = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    params = eval_out.get("param")
    chain = eval_out.get("chain")

    lines: list[str] = []

    if shape:
        lines.append(
            f"data: M={shape.get('M')} N={shape.get('N')} "
            f"T={shape.get('T')} J={shape.get('J')} | n_obs={shape.get('n_obs')}"
        )
        lines.append("")

    if models:
        lines.extend(_format_metric_table(models))

    delta_lines = _format_delta_section(deltas)
    if delta_lines:
        lines.append("")
        lines.extend(delta_lines)

    if isinstance(params, dict):
        param_lines = _format_parameter_section(params)
        if param_lines:
            lines.append("")
            lines.extend(param_lines)

    if isinstance(chain, dict):
        chain_lines = _format_chain_section(chain)
        if chain_lines:
            lines.append("")
            lines.extend(chain_lines)

    return "\n".join(lines)
