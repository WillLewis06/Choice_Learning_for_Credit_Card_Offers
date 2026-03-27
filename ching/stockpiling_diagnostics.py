"""Diagnostics helpers for the Ching-style stockpiling sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import tensorflow as tf

__all__ = [
    "StockpilingChunkTrace",
    "StockpilingChunkSummary",
    "format_scalar",
    "summarize_chunk_trace",
    "format_chunk_progress_line",
    "report_chunk_progress",
    "format_run_summary_line",
    "report_run_summary",
    "build_mcmc_summary",
]


@dataclass(frozen=True)
class StockpilingChunkTrace:
    """Store the retained diagnostics trace for one MCMC chunk."""

    beta: tf.Tensor
    mean_alpha: tf.Tensor
    mean_v: tf.Tensor
    mean_fc: tf.Tensor
    mean_u_scale: tf.Tensor
    joint_logpost: tf.Tensor

    beta_accept: tf.Tensor
    alpha_accept: tf.Tensor
    v_accept: tf.Tensor
    fc_accept: tf.Tensor
    u_scale_accept: tf.Tensor


@dataclass(frozen=True)
class StockpilingChunkSummary:
    """Store the scalar reporting summary for one MCMC chunk."""

    chunk_idx: int
    total_chunks: int | None

    beta_last: float
    mean_alpha_last: float
    mean_v_last: float
    mean_fc_last: float
    mean_u_scale_last: float
    joint_logpost_last: float

    beta_accept_mean: float
    alpha_accept_mean: float
    v_accept_mean: float
    fc_accept_mean: float
    u_scale_accept_mean: float


def _last_float(x: tf.Tensor) -> float:
    """Return the last element of a tensor trace as a Python float."""
    return float(tf.reshape(x, (-1,))[-1].numpy())


def _mean_float(x: tf.Tensor) -> float:
    """Return the mean of a tensor trace as a Python float."""
    return float(tf.reduce_mean(x).numpy())


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty float sequence."""
    if len(values) == 0:
        raise ValueError("values must be non-empty.")
    return float(sum(values) / len(values))


def format_scalar(x: float, precision: int = 4) -> str:
    """Format a scalar for one-line diagnostics output."""
    return f"{x:.{precision}f}"


def summarize_chunk_trace(
    trace: StockpilingChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> StockpilingChunkSummary:
    """Convert a retained chunk trace into a scalar chunk summary."""
    return StockpilingChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        beta_last=_last_float(trace.beta),
        mean_alpha_last=_last_float(trace.mean_alpha),
        mean_v_last=_last_float(trace.mean_v),
        mean_fc_last=_last_float(trace.mean_fc),
        mean_u_scale_last=_last_float(trace.mean_u_scale),
        joint_logpost_last=_last_float(trace.joint_logpost),
        beta_accept_mean=_mean_float(trace.beta_accept),
        alpha_accept_mean=_mean_float(trace.alpha_accept),
        v_accept_mean=_mean_float(trace.v_accept),
        fc_accept_mean=_mean_float(trace.fc_accept),
        u_scale_accept_mean=_mean_float(trace.u_scale_accept),
    )


def format_chunk_progress_line(summary: StockpilingChunkSummary) -> str:
    """Format the one-line progress report for one chunk."""
    if summary.total_chunks is None:
        chunk_label = f"chunk={summary.chunk_idx}"
    else:
        chunk_label = f"chunk={summary.chunk_idx}/{summary.total_chunks}"

    return (
        f"[Stockpiling] {chunk_label}"
        f" | beta={format_scalar(summary.beta_last)}"
        f" mean_alpha={format_scalar(summary.mean_alpha_last)}"
        f" mean_v={format_scalar(summary.mean_v_last)}"
        f" mean_fc={format_scalar(summary.mean_fc_last)}"
        f" mean_u_scale={format_scalar(summary.mean_u_scale_last)}"
        f" | logpost={format_scalar(summary.joint_logpost_last)}"
        f" | acc_beta={format_scalar(summary.beta_accept_mean)}"
        f" acc_alpha={format_scalar(summary.alpha_accept_mean)}"
        f" acc_v={format_scalar(summary.v_accept_mean)}"
        f" acc_fc={format_scalar(summary.fc_accept_mean)}"
        f" acc_u_scale={format_scalar(summary.u_scale_accept_mean)}"
    )


def report_chunk_progress(
    trace: StockpilingChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> StockpilingChunkSummary:
    """Print and return the scalar summary for one chunk."""
    summary = summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )
    print(format_chunk_progress_line(summary))
    return summary


def format_run_summary_line(summaries: Sequence[StockpilingChunkSummary]) -> str:
    """Format the final run summary across all chunks."""
    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    mean_acc_beta = _mean([s.beta_accept_mean for s in summaries])
    mean_acc_alpha = _mean([s.alpha_accept_mean for s in summaries])
    mean_acc_v = _mean([s.v_accept_mean for s in summaries])
    mean_acc_fc = _mean([s.fc_accept_mean for s in summaries])
    mean_acc_u_scale = _mean([s.u_scale_accept_mean for s in summaries])

    return (
        f"[Stockpiling] final"
        f" | chunks={len(summaries)}"
        f" | beta={format_scalar(last.beta_last)}"
        f" mean_alpha={format_scalar(last.mean_alpha_last)}"
        f" mean_v={format_scalar(last.mean_v_last)}"
        f" mean_fc={format_scalar(last.mean_fc_last)}"
        f" mean_u_scale={format_scalar(last.mean_u_scale_last)}"
        f" | logpost={format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_beta={format_scalar(mean_acc_beta)}"
        f" mean_acc_alpha={format_scalar(mean_acc_alpha)}"
        f" mean_acc_v={format_scalar(mean_acc_v)}"
        f" mean_acc_fc={format_scalar(mean_acc_fc)}"
        f" mean_acc_u_scale={format_scalar(mean_acc_u_scale)}"
    )


def report_run_summary(summaries: Sequence[StockpilingChunkSummary]) -> None:
    """Print the final run summary line."""
    print(format_run_summary_line(summaries))


def build_mcmc_summary(
    summaries: Sequence[StockpilingChunkSummary],
    n_saved: int,
) -> dict[str, Any]:
    """Build the evaluation-facing MCMC summary from chunk summaries."""
    if n_saved <= 0:
        raise ValueError("n_saved must be > 0.")
    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    return {
        "n_saved": int(n_saved),
        "beta_accept": _mean([s.beta_accept_mean for s in summaries]),
        "alpha_accept": _mean([s.alpha_accept_mean for s in summaries]),
        "v_accept": _mean([s.v_accept_mean for s in summaries]),
        "fc_accept": _mean([s.fc_accept_mean for s in summaries]),
        "u_scale_accept": _mean([s.u_scale_accept_mean for s in summaries]),
        "num_chunks": int(len(summaries)),
        "joint_logpost_last": float(summaries[-1].joint_logpost_last),
    }
