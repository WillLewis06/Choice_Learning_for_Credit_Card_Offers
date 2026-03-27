"""Diagnostics formatting for the choice-learn shrinkage sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf


@dataclass(frozen=True)
class ChoiceLearnChunkTrace:
    """Store the per-draw diagnostics collected for one MCMC chunk."""

    alpha: tf.Tensor
    mean_E_bar: tf.Tensor
    norm_E_bar: tf.Tensor
    norm_njt: tf.Tensor
    gamma_active_share: tf.Tensor
    joint_logpost: tf.Tensor
    alpha_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


@dataclass(frozen=True)
class ChoiceLearnChunkSummary:
    """Store the condensed scalar summary reported for one chunk."""

    chunk_idx: int
    total_chunks: int | None

    alpha_last: float
    mean_E_bar_last: float
    norm_E_bar_last: float
    norm_njt_last: float
    gamma_active_share_last: float
    joint_logpost_last: float

    alpha_accept_mean: float
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
    trace: ChoiceLearnChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> ChoiceLearnChunkSummary:
    """Convert a full chunk trace into the scalar reporting summary."""

    return ChoiceLearnChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        alpha_last=_scalar_last(trace.alpha),
        mean_E_bar_last=_scalar_last(trace.mean_E_bar),
        norm_E_bar_last=_scalar_last(trace.norm_E_bar),
        norm_njt_last=_scalar_last(trace.norm_njt),
        gamma_active_share_last=_scalar_last(trace.gamma_active_share),
        joint_logpost_last=_scalar_last(trace.joint_logpost),
        alpha_accept_mean=_scalar_mean(trace.alpha_accept),
        E_bar_accept_mean=_scalar_mean(trace.E_bar_accept),
        njt_accept_mean=_scalar_mean(trace.njt_accept),
    )


def format_chunk_progress_line(summary: ChoiceLearnChunkSummary) -> str:
    """Format the one-line progress report for a chunk summary."""

    if summary.total_chunks is None:
        chunk_label = f"chunk={summary.chunk_idx}"
    else:
        chunk_label = f"chunk={summary.chunk_idx}/{summary.total_chunks}"

    return (
        f"[ChoiceLearnShrinkage] {chunk_label}"
        f" | alpha={format_scalar(summary.alpha_last)}"
        f" | mean_E_bar={format_scalar(summary.mean_E_bar_last)}"
        f" norm_E_bar={format_scalar(summary.norm_E_bar_last)}"
        f" norm_njt={format_scalar(summary.norm_njt_last)}"
        f" gamma_share={format_scalar(summary.gamma_active_share_last)}"
        f" | logpost={format_scalar(summary.joint_logpost_last)}"
        f" | acc_alpha={format_scalar(summary.alpha_accept_mean)}"
        f" acc_E_bar={format_scalar(summary.E_bar_accept_mean)}"
        f" acc_njt={format_scalar(summary.njt_accept_mean)}"
    )


def report_chunk_progress(
    trace: ChoiceLearnChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> ChoiceLearnChunkSummary:
    """Print the chunk progress line and return its scalar summary."""

    summary = summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )
    print(format_chunk_progress_line(summary))
    return summary


def format_run_summary_line(
    summaries: Sequence[ChoiceLearnChunkSummary],
) -> str:
    """Format the final run-level summary line across all chunks."""

    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    alpha_accept_mean = sum(s.alpha_accept_mean for s in summaries) / len(summaries)
    E_bar_accept_mean = sum(s.E_bar_accept_mean for s in summaries) / len(summaries)
    njt_accept_mean = sum(s.njt_accept_mean for s in summaries) / len(summaries)

    return (
        f"[ChoiceLearnShrinkage] final"
        f" | chunks={len(summaries)}"
        f" | alpha={format_scalar(last.alpha_last)}"
        f" | mean_E_bar={format_scalar(last.mean_E_bar_last)}"
        f" norm_E_bar={format_scalar(last.norm_E_bar_last)}"
        f" norm_njt={format_scalar(last.norm_njt_last)}"
        f" gamma_share={format_scalar(last.gamma_active_share_last)}"
        f" | logpost={format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_alpha={format_scalar(alpha_accept_mean)}"
        f" mean_acc_E_bar={format_scalar(E_bar_accept_mean)}"
        f" mean_acc_njt={format_scalar(njt_accept_mean)}"
    )


def report_run_summary(
    summaries: Sequence[ChoiceLearnChunkSummary],
) -> None:
    """Print the final run-level summary line."""

    print(format_run_summary_line(summaries))
