"""Diagnostics formatting for the Ching stockpiling sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
]


@dataclass(frozen=True)
class StockpilingChunkTrace:
    """Store the per-draw diagnostics collected for one MCMC chunk."""

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
    """Store the condensed scalar summary reported for one chunk."""

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


def _scalar_last(x: tf.Tensor) -> float:
    """Return the last scalar value from a tensor trace."""

    x_flat = tf.reshape(x, [-1])
    return float(x_flat[-1].numpy())


def _scalar_mean(x: tf.Tensor) -> float:
    """Return the mean of a tensor trace as a Python float."""

    return float(tf.reduce_mean(x).numpy())


def format_scalar(x: float, precision: int = 4) -> str:
    """Format a scalar value to fixed precision for diagnostics output."""

    return f"{x:.{precision}f}"


def summarize_chunk_trace(
    trace: StockpilingChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> StockpilingChunkSummary:
    """Convert a full chunk trace into the scalar reporting summary."""

    return StockpilingChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        beta_last=_scalar_last(trace.beta),
        mean_alpha_last=_scalar_last(trace.mean_alpha),
        mean_v_last=_scalar_last(trace.mean_v),
        mean_fc_last=_scalar_last(trace.mean_fc),
        mean_u_scale_last=_scalar_last(trace.mean_u_scale),
        joint_logpost_last=_scalar_last(trace.joint_logpost),
        beta_accept_mean=_scalar_mean(trace.beta_accept),
        alpha_accept_mean=_scalar_mean(trace.alpha_accept),
        v_accept_mean=_scalar_mean(trace.v_accept),
        fc_accept_mean=_scalar_mean(trace.fc_accept),
        u_scale_accept_mean=_scalar_mean(trace.u_scale_accept),
    )


def format_chunk_progress_line(summary: StockpilingChunkSummary) -> str:
    """Format the one-line progress report for a chunk summary."""

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
    """Print the chunk progress line and return its scalar summary."""

    summary = summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )
    print(format_chunk_progress_line(summary))
    return summary


def format_run_summary_line(
    summaries: Sequence[StockpilingChunkSummary],
) -> str:
    """Format the final run-level summary line across all chunks."""

    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    beta_accept_mean = sum(s.beta_accept_mean for s in summaries) / len(summaries)
    alpha_accept_mean = sum(s.alpha_accept_mean for s in summaries) / len(summaries)
    v_accept_mean = sum(s.v_accept_mean for s in summaries) / len(summaries)
    fc_accept_mean = sum(s.fc_accept_mean for s in summaries) / len(summaries)
    u_scale_accept_mean = sum(s.u_scale_accept_mean for s in summaries) / len(summaries)

    return (
        f"[Stockpiling] final"
        f" | chunks={len(summaries)}"
        f" | beta={format_scalar(last.beta_last)}"
        f" mean_alpha={format_scalar(last.mean_alpha_last)}"
        f" mean_v={format_scalar(last.mean_v_last)}"
        f" mean_fc={format_scalar(last.mean_fc_last)}"
        f" mean_u_scale={format_scalar(last.mean_u_scale_last)}"
        f" | logpost={format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_beta={format_scalar(beta_accept_mean)}"
        f" mean_acc_alpha={format_scalar(alpha_accept_mean)}"
        f" mean_acc_v={format_scalar(v_accept_mean)}"
        f" mean_acc_fc={format_scalar(fc_accept_mean)}"
        f" mean_acc_u_scale={format_scalar(u_scale_accept_mean)}"
    )


def report_run_summary(
    summaries: Sequence[StockpilingChunkSummary],
) -> None:
    """Print the final run-level summary line."""

    print(format_run_summary_line(summaries))
