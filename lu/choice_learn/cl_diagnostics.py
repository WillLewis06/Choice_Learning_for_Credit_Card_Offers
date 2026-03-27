"""Diagnostics formatting for the choice-learn shrinkage sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import tensorflow as tf


@dataclass(frozen=True)
class ChoiceLearnChunkSummary:
    """Store the reported Python-scalar diagnostics for one MCMC chunk."""

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

    return float(tf.reshape(x, [-1])[-1].numpy())


def _scalar_mean(x: tf.Tensor) -> float:
    """Return the mean of a tensor trace as a Python float."""

    return float(tf.reduce_mean(x).numpy())


def _format_scalar(x: float) -> str:
    """Format a diagnostic scalar with fixed precision."""

    return f"{x:.4f}"


def summarize_chunk_trace(
    trace: Mapping[str, tf.Tensor],
    chunk_idx: int,
    total_chunks: int | None = None,
) -> ChoiceLearnChunkSummary:
    """Reduce one raw chunk trace dictionary to a reported scalar summary."""

    # State diagnostics are reported at the end of the chunk.
    # Acceptance diagnostics are reported as chunk means.
    return ChoiceLearnChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        alpha_last=_scalar_last(trace["alpha"]),
        mean_E_bar_last=_scalar_last(trace["mean_E_bar"]),
        norm_E_bar_last=_scalar_last(trace["norm_E_bar"]),
        norm_njt_last=_scalar_last(trace["norm_njt"]),
        gamma_active_share_last=_scalar_last(trace["gamma_active_share"]),
        joint_logpost_last=_scalar_last(trace["joint_logpost"]),
        alpha_accept_mean=_scalar_mean(trace["alpha_accept"]),
        E_bar_accept_mean=_scalar_mean(trace["E_bar_accept"]),
        njt_accept_mean=_scalar_mean(trace["njt_accept"]),
    )


def format_chunk_progress_line(summary: ChoiceLearnChunkSummary) -> str:
    """Return the one-line progress report for one chunk summary."""

    if summary.total_chunks is None:
        chunk_label = f"chunk={summary.chunk_idx}"
    else:
        chunk_label = f"chunk={summary.chunk_idx}/{summary.total_chunks}"

    return (
        f"[ChoiceLearnShrinkage] {chunk_label}"
        f" | alpha={_format_scalar(summary.alpha_last)}"
        f" | mean_E_bar={_format_scalar(summary.mean_E_bar_last)}"
        f" | norm_E_bar={_format_scalar(summary.norm_E_bar_last)}"
        f" | norm_njt={_format_scalar(summary.norm_njt_last)}"
        f" | gamma_share={_format_scalar(summary.gamma_active_share_last)}"
        f" | logpost={_format_scalar(summary.joint_logpost_last)}"
        f" | acc_alpha={_format_scalar(summary.alpha_accept_mean)}"
        f" | acc_E_bar={_format_scalar(summary.E_bar_accept_mean)}"
        f" | acc_njt={_format_scalar(summary.njt_accept_mean)}"
    )


def format_run_summary_line(
    summaries: Sequence[ChoiceLearnChunkSummary],
) -> str:
    """Return the final run-level summary line across all reported chunks."""

    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    # The final state diagnostics come from the last reported chunk.
    # Acceptance diagnostics are averaged over chunk-level means.
    alpha_accept_mean = sum(summary.alpha_accept_mean for summary in summaries) / len(
        summaries
    )
    E_bar_accept_mean = sum(summary.E_bar_accept_mean for summary in summaries) / len(
        summaries
    )
    njt_accept_mean = sum(summary.njt_accept_mean for summary in summaries) / len(
        summaries
    )

    return (
        f"[ChoiceLearnShrinkage] final"
        f" | chunks={len(summaries)}"
        f" | alpha={_format_scalar(last.alpha_last)}"
        f" | mean_E_bar={_format_scalar(last.mean_E_bar_last)}"
        f" | norm_E_bar={_format_scalar(last.norm_E_bar_last)}"
        f" | norm_njt={_format_scalar(last.norm_njt_last)}"
        f" | gamma_share={_format_scalar(last.gamma_active_share_last)}"
        f" | logpost={_format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_alpha={_format_scalar(alpha_accept_mean)}"
        f" | mean_acc_E_bar={_format_scalar(E_bar_accept_mean)}"
        f" | mean_acc_njt={_format_scalar(njt_accept_mean)}"
    )
