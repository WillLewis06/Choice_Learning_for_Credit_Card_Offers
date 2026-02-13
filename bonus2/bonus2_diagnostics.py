"""
bonus2/bonus2_diagnostics.py

TF-safe diagnostics printing for Bonus Q2 (habit + peer + DOW + seasonality) MNL model.

This mirrors the stockpiling_diagnostics.py pattern:
  - small formatting helpers
  - a single report_iteration_progress(...) used by the estimator each MCMC iteration
  - stateless, no running means or storage

Inputs:
  z: dict[str, tf.Tensor] containing *unconstrained* blocks with keys expected by
     bonus2_model.unconstrained_to_theta(...) (see below)
  it: tf.Tensor scalar int32 iteration index

Expected z keys:
  z_beta_market_mj : (M,J)
  z_beta_habit_j   : (J,)
  z_beta_peer_j    : (J,)
  z_decay_rate_j   : (J,)
  z_beta_dow_m     : (M,7)
  z_beta_dow_j     : (J,7)
  z_a_m            : (M,K)
  z_b_m            : (M,K)
  z_a_j            : (J,K)
  z_b_j            : (J,K)

Notes:
  - DOW and seasonal parameters are typically centered by identifiability constraints,
    so means are not informative; we print std/mean(abs) instead.
"""

from __future__ import annotations

import tensorflow as tf

from bonus2 import bonus2_model as model


def _fmt4(x: tf.Tensor) -> tf.Tensor:
    """Format tensor values as strings with 4 decimals (no scientific notation)."""
    x = tf.cast(x, tf.float64)
    return tf.strings.as_string(x, precision=4, scientific=False)


def _mean_abs(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float64)
    return tf.reduce_mean(tf.abs(x))


def _std(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float64)
    return tf.math.reduce_std(x)


def report_iteration_progress(
    z: dict[str, tf.Tensor], it: tf.Tensor, market_k: int = 3
) -> None:
    """
    Print a compact per-iteration summary.

    Prints:
      Line 1: global summaries of core parameter blocks in theta-space
      Line 2: a short vector of per-market summaries for the first `market_k` markets
    """
    # Ensure float64 z for transforms.
    z64 = {k: tf.cast(v, tf.float64) for k, v in z.items()}
    theta = model.unconstrained_to_theta(z64)

    # Core product-level parameters (informative means).
    mean_beta_habit = tf.reduce_mean(theta["beta_habit_j"])
    mean_beta_peer = tf.reduce_mean(theta["beta_peer_j"])
    mean_decay = tf.reduce_mean(theta["decay_rate_j"])

    # Market-product intercept shifts: mean absolute magnitude is typically more informative.
    mean_abs_beta_market = _mean_abs(theta["beta_market_mj"])

    # DOW and seasonality: print dispersion (std), not mean (often centered).
    std_beta_dow_m = _std(theta["beta_dow_m"])
    std_beta_dow_j = _std(theta["beta_dow_j"])
    std_a_m = _std(theta["a_m"])
    std_b_m = _std(theta["b_m"])
    std_a_j = _std(theta["a_j"])
    std_b_j = _std(theta["b_j"])

    tf.print(
        "[Bonus2] it=",
        it,
        "| mean(beta_habit)=",
        _fmt4(mean_beta_habit),
        ", mean(beta_peer)=",
        _fmt4(mean_beta_peer),
        ", mean(decay)=",
        _fmt4(mean_decay),
        ", mean(|beta_market|)=",
        _fmt4(mean_abs_beta_market),
        ", std(dow_m)=",
        _fmt4(std_beta_dow_m),
        ", std(dow_j)=",
        _fmt4(std_beta_dow_j),
        ", std(a_m)=",
        _fmt4(std_a_m),
        ", std(b_m)=",
        _fmt4(std_b_m),
        ", std(a_j)=",
        _fmt4(std_a_j),
        ", std(b_j)=",
        _fmt4(std_b_j),
    )

    # Per-market summaries for the first market_k markets:
    # - mean over products of beta_market_mj (can be near 0 depending on prior)
    # - std over weekdays of beta_dow_m[m,:] (useful under centering)
    beta_market_by_m = tf.reduce_mean(theta["beta_market_mj"], axis=1)  # (M,)
    dow_m_std_by_m = tf.math.reduce_std(theta["beta_dow_m"], axis=1)  # (M,)

    k = tf.minimum(tf.shape(beta_market_by_m)[0], tf.cast(market_k, tf.int32))
    tf.print(
        "[Bonus2] by_market[:",
        k,
        "] mean(beta_market)=",
        beta_market_by_m[:k],
        "| std(dow_m)=",
        dow_m_std_by_m[:k],
    )
