"""Diagnostics formatting for the Bonus Q2 sampler."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class Bonus2ChunkTrace:
    """Store per-draw diagnostics for one MCMC chunk."""

    beta_intercept_mean: tf.Tensor
    beta_habit_mean: tf.Tensor
    beta_peer_mean: tf.Tensor
    a_mean: tf.Tensor
    b_mean: tf.Tensor
    joint_logpost: tf.Tensor

    beta_intercept_accept: tf.Tensor
    beta_habit_accept: tf.Tensor
    beta_peer_accept: tf.Tensor
    beta_weekend_accept: tf.Tensor
    a_accept: tf.Tensor
    b_accept: tf.Tensor


@dataclass(frozen=True)
class Bonus2ChunkSummary:
    """Store the scalar summary reported for one chunk."""

    chunk_idx: int
    total_chunks: int | None

    beta_intercept_mean_last: float
    beta_habit_mean_last: float
    beta_peer_mean_last: float
    a_mean_last: float
    b_mean_last: float
    joint_logpost_last: float

    beta_intercept_accept_mean: float
    beta_habit_accept_mean: float
    beta_peer_accept_mean: float
    beta_weekend_accept_mean: float
    a_accept_mean: float
    b_accept_mean: float


def _scalar_last(x: tf.Tensor) -> float:
    """Return the last scalar value from a tensor trace."""
    x_flat = tf.reshape(x, [-1])
    return float(x_flat[-1].numpy())


def _scalar_mean(x: tf.Tensor) -> float:
    """Return the mean of a tensor trace as a Python float."""
    return float(tf.reduce_mean(x).numpy())


def _trace_mean(x: tf.Tensor) -> tf.Tensor:
    """Return the per-draw mean over all non-leading dimensions."""
    if x.shape.rank == 1:
        return x
    x_flat = tf.reshape(x, (tf.shape(x)[0], -1))
    return tf.reduce_mean(x_flat, axis=1)


def _mean_acceptance(
    summaries: Sequence[Bonus2ChunkSummary],
    attr_name: str,
) -> float:
    """Return the run-level mean acceptance for one block."""
    return sum(getattr(summary, attr_name) for summary in summaries) / len(summaries)


def format_scalar(x: float, precision: int = 4) -> str:
    """Format a scalar value to fixed precision."""
    return f"{x:.{precision}f}"


def build_chunk_trace(
    z_beta_intercept_j: tf.Tensor,
    z_beta_habit_j: tf.Tensor,
    z_beta_peer_j: tf.Tensor,
    z_beta_weekend_jw: tf.Tensor,
    z_a_m: tf.Tensor,
    z_b_m: tf.Tensor,
    joint_logpost: tf.Tensor,
    beta_intercept_accept: tf.Tensor,
    beta_habit_accept: tf.Tensor,
    beta_peer_accept: tf.Tensor,
    beta_weekend_accept: tf.Tensor,
    a_accept: tf.Tensor,
    b_accept: tf.Tensor,
) -> Bonus2ChunkTrace:
    """Build a chunk trace from raw draws and acceptance indicators."""
    del z_beta_weekend_jw

    return Bonus2ChunkTrace(
        beta_intercept_mean=_trace_mean(z_beta_intercept_j),
        beta_habit_mean=_trace_mean(z_beta_habit_j),
        beta_peer_mean=_trace_mean(z_beta_peer_j),
        a_mean=_trace_mean(z_a_m),
        b_mean=_trace_mean(z_b_m),
        joint_logpost=joint_logpost,
        beta_intercept_accept=beta_intercept_accept,
        beta_habit_accept=beta_habit_accept,
        beta_peer_accept=beta_peer_accept,
        beta_weekend_accept=beta_weekend_accept,
        a_accept=a_accept,
        b_accept=b_accept,
    )


def summarize_chunk_trace(
    trace: Bonus2ChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> Bonus2ChunkSummary:
    """Convert a chunk trace into a scalar reporting summary."""
    return Bonus2ChunkSummary(
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        beta_intercept_mean_last=_scalar_last(trace.beta_intercept_mean),
        beta_habit_mean_last=_scalar_last(trace.beta_habit_mean),
        beta_peer_mean_last=_scalar_last(trace.beta_peer_mean),
        a_mean_last=_scalar_last(trace.a_mean),
        b_mean_last=_scalar_last(trace.b_mean),
        joint_logpost_last=_scalar_last(trace.joint_logpost),
        beta_intercept_accept_mean=_scalar_mean(trace.beta_intercept_accept),
        beta_habit_accept_mean=_scalar_mean(trace.beta_habit_accept),
        beta_peer_accept_mean=_scalar_mean(trace.beta_peer_accept),
        beta_weekend_accept_mean=_scalar_mean(trace.beta_weekend_accept),
        a_accept_mean=_scalar_mean(trace.a_accept),
        b_accept_mean=_scalar_mean(trace.b_accept),
    )


def format_chunk_progress_line(summary: Bonus2ChunkSummary) -> str:
    """Format the one-line progress report for one chunk."""
    if summary.total_chunks is None:
        chunk_label = f"chunk={summary.chunk_idx}"
    else:
        chunk_label = f"chunk={summary.chunk_idx}/{summary.total_chunks}"

    return (
        f"[Bonus2] {chunk_label}"
        f" | intercept={format_scalar(summary.beta_intercept_mean_last)}"
        f" habit={format_scalar(summary.beta_habit_mean_last)}"
        f" peer={format_scalar(summary.beta_peer_mean_last)}"
        f" a={format_scalar(summary.a_mean_last)}"
        f" b={format_scalar(summary.b_mean_last)}"
        f" | logpost={format_scalar(summary.joint_logpost_last)}"
        f" | acc_intercept={format_scalar(summary.beta_intercept_accept_mean)}"
        f" acc_habit={format_scalar(summary.beta_habit_accept_mean)}"
        f" acc_peer={format_scalar(summary.beta_peer_accept_mean)}"
        f" acc_weekend={format_scalar(summary.beta_weekend_accept_mean)}"
        f" acc_a={format_scalar(summary.a_accept_mean)}"
        f" acc_b={format_scalar(summary.b_accept_mean)}"
    )


def report_chunk_progress(
    trace: Bonus2ChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> Bonus2ChunkSummary:
    """Print the chunk progress line and return its summary."""
    summary = summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )
    print(format_chunk_progress_line(summary))
    return summary


def format_run_summary_line(
    summaries: Sequence[Bonus2ChunkSummary],
) -> str:
    """Format the final run-level summary line."""
    if len(summaries) == 0:
        raise ValueError("summaries must be non-empty.")

    last = summaries[-1]

    return (
        f"[Bonus2] final"
        f" | chunks={len(summaries)}"
        f" | intercept={format_scalar(last.beta_intercept_mean_last)}"
        f" habit={format_scalar(last.beta_habit_mean_last)}"
        f" peer={format_scalar(last.beta_peer_mean_last)}"
        f" a={format_scalar(last.a_mean_last)}"
        f" b={format_scalar(last.b_mean_last)}"
        f" | logpost={format_scalar(last.joint_logpost_last)}"
        f" | mean_acc_intercept="
        f"{format_scalar(_mean_acceptance(summaries, 'beta_intercept_accept_mean'))}"
        f" mean_acc_habit="
        f"{format_scalar(_mean_acceptance(summaries, 'beta_habit_accept_mean'))}"
        f" mean_acc_peer="
        f"{format_scalar(_mean_acceptance(summaries, 'beta_peer_accept_mean'))}"
        f" mean_acc_weekend="
        f"{format_scalar(_mean_acceptance(summaries, 'beta_weekend_accept_mean'))}"
        f" mean_acc_a={format_scalar(_mean_acceptance(summaries, 'a_accept_mean'))}"
        f" mean_acc_b={format_scalar(_mean_acceptance(summaries, 'b_accept_mean'))}"
    )


def report_run_summary(
    summaries: Sequence[Bonus2ChunkSummary],
) -> None:
    """Print the final run-level summary line."""
    print(format_run_summary_line(summaries))
