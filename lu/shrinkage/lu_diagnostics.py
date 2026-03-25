"""Diagnostics formatting for the Lu shrinkage sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass(frozen=True)
class LuChunkTrace:
    """Store the per-draw diagnostics collected for one MCMC chunk."""

    beta_p: tf.Tensor
    beta_w: tf.Tensor
    sigma: tf.Tensor
    mean_E_bar: tf.Tensor
    norm_E_bar: tf.Tensor
    norm_njt: tf.Tensor
    gamma_active_share: tf.Tensor
    joint_logpost: tf.Tensor
    beta_accept: tf.Tensor
    r_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


@dataclass(frozen=True)
class LuChunkSummary:
    """Store the condensed scalar summary reported for one chunk."""

    chunk_idx: int
    total_chunks: int | None

    beta_p_last: float
    beta_w_last: float
    sigma_last: float
    mean_E_bar_last: float
    norm_E_bar_last: float
    norm_njt_last: float
    gamma_active_share_last: float
    joint_logpost_last: float

    beta_accept_mean: float
    r_accept_mean: float
    E_bar_accept_mean: float
    njt_accept_mean: float


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
    trace: LuChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> LuChunkSummary:
    """Convert a full chunk trace into the scalar reporting summary."""

    # Use the last sampled state together with mean acceptance rates over the chunk.
    return LuChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        beta_p_last=_scalar_last(trace.beta_p),
        beta_w_last=_scalar_last(trace.beta_w),
        sigma_last=_scalar_last(trace.sigma),
        mean_E_bar_last=_scalar_last(trace.mean_E_bar),
        norm_E_bar_last=_scalar_last(trace.norm_E_bar),
        norm_njt_last=_scalar_last(trace.norm_njt),
        gamma_active_share_last=_scalar_last(trace.gamma_active_share),
        joint_logpost_last=_scalar_last(trace.joint_logpost),
        beta_accept_mean=_scalar_mean(trace.beta_accept),
        r_accept_mean=_scalar_mean(trace.r_accept),
        E_bar_accept_mean=_scalar_mean(trace.E_bar_accept),
        njt_accept_mean=_scalar_mean(trace.njt_accept),
    )


def format_chunk_progress_line(summary: LuChunkSummary) -> str:
    """Format the one-line progress report for a chunk summary."""

    # Format the chunk label according to whether the total number of chunks is known.
    if summary.total_chunks is None:
        chunk_label = f"chunk={summary.chunk_idx}"
    else:
        chunk_label = f"chunk={summary.chunk_idx}/{summary.total_chunks}"

    # Report the current state and acceptance summaries for the chunk.
    return (
        f"[LuShrinkage] {chunk_label}"
        f" | beta_p={format_scalar(summary.beta_p_last)}"
        f" beta_w={format_scalar(summary.beta_w_last)}"
        f" sigma={format_scalar(summary.sigma_last)}"
        f" | mean_E_bar={format_scalar(summary.mean_E_bar_last)}"
        f" norm_E_bar={format_scalar(summary.norm_E_bar_last)}"
        f" norm_njt={format_scalar(summary.norm_njt_last)}"
        f" gamma_share={format_scalar(summary.gamma_active_share_last)}"
        f" | logpost={format_scalar(summary.joint_logpost_last)}"
        f" | acc_beta={format_scalar(summary.beta_accept_mean)}"
        f" acc_r={format_scalar(summary.r_accept_mean)}"
        f" acc_E_bar={format_scalar(summary.E_bar_accept_mean)}"
        f" acc_njt={format_scalar(summary.njt_accept_mean)}"
    )


def report_chunk_progress(
    trace: LuChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> LuChunkSummary:
    """Print the chunk progress line and return its scalar summary."""

    # Reduce the full trace to the scalar summary used in the printed report.
    summary = summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )
    print(format_chunk_progress_line(summary))
    return summary


def format_run_summary_line(
    summaries: Sequence[LuChunkSummary],
) -> str:
    """Format the final run-level summary line across all chunks."""

    # Require at least one chunk summary to define a final report.
    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    # Average the chunk-level acceptance summaries across the full run.
    beta_accept_mean = sum(s.beta_accept_mean for s in summaries) / len(summaries)
    r_accept_mean = sum(s.r_accept_mean for s in summaries) / len(summaries)
    E_bar_accept_mean = sum(s.E_bar_accept_mean for s in summaries) / len(summaries)
    njt_accept_mean = sum(s.njt_accept_mean for s in summaries) / len(summaries)

    # Report the final sampled state together with average acceptance rates.
    return (
        f"[LuShrinkage] final"
        f" | chunks={len(summaries)}"
        f" | beta_p={format_scalar(last.beta_p_last)}"
        f" beta_w={format_scalar(last.beta_w_last)}"
        f" sigma={format_scalar(last.sigma_last)}"
        f" | mean_E_bar={format_scalar(last.mean_E_bar_last)}"
        f" norm_E_bar={format_scalar(last.norm_E_bar_last)}"
        f" norm_njt={format_scalar(last.norm_njt_last)}"
        f" gamma_share={format_scalar(last.gamma_active_share_last)}"
        f" | logpost={format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_beta={format_scalar(beta_accept_mean)}"
        f" mean_acc_r={format_scalar(r_accept_mean)}"
        f" mean_acc_E_bar={format_scalar(E_bar_accept_mean)}"
        f" mean_acc_njt={format_scalar(njt_accept_mean)}"
    )


def report_run_summary(
    summaries: Sequence[LuChunkSummary],
) -> None:
    """Print the final run-level summary line."""

    print(format_run_summary_line(summaries))
