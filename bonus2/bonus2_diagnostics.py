"""
bonus2_diagnostics.py

TensorFlow-compatible diagnostics for Bonus Q2.

This module is safe to call from within @tf.function. It computes scalar summary
statistics using TensorFlow ops and prints exactly one line per call using tf.print.

Public API:
  - report_iteration_progress(z, it)
  - report_theta_summary(prefix, theta)
  - report_known_summary(lookback, K, season_period, decay)

No input validation is performed here. Inputs are assumed to be valid and
consistent with the Bonus2 model and estimator contracts.
"""

from __future__ import annotations

import tensorflow as tf

from bonus2 import bonus2_model as model


def _fmt4(x: tf.Tensor) -> tf.Tensor:
    """Format a scalar float tensor with 4 decimals (no scientific notation)."""
    return tf.strings.as_string(x, precision=4, scientific=False)


def _fmt_i(x: tf.Tensor) -> tf.Tensor:
    """Format a scalar integer tensor."""
    return tf.strings.as_string(x)


def _mean_or_zero(x: tf.Tensor) -> tf.Tensor:
    """Return mean(x) if x has at least one element, else 0.0 (float64)."""
    return tf.cond(
        tf.size(x) > 0,
        lambda: tf.reduce_mean(x),
        lambda: tf.constant(0.0, dtype=tf.float64),
    )


def _std_or_zero(x: tf.Tensor) -> tf.Tensor:
    """Return std(x) if x has at least one element, else 0.0 (float64)."""
    return tf.cond(
        tf.size(x) > 0,
        lambda: tf.math.reduce_std(x),
        lambda: tf.constant(0.0, dtype=tf.float64),
    )


def theta_stats(theta: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Compute scalar summary statistics for structural parameters theta."""
    beta_intercept_j = theta["beta_intercept_j"]  # (J,)
    beta_habit_j = theta["beta_habit_j"]  # (J,)
    beta_peer_j = theta["beta_peer_j"]  # (J,)
    beta_weekend_jw = theta["beta_weekend_jw"]  # (J,2)
    a_m = theta["a_m"]  # (M,K)
    b_m = theta["b_m"]  # (M,K)

    weekend_lift_j = beta_weekend_jw[:, 1] - beta_weekend_jw[:, 0]  # (J,)

    return {
        "b0": _mean_or_zero(beta_intercept_j),
        "bh": _mean_or_zero(beta_habit_j),
        "bp": _mean_or_zero(beta_peer_j),
        "wl_sd": _std_or_zero(weekend_lift_j),
        "am": _mean_or_zero(a_m),
        "bm": _mean_or_zero(b_m),
    }


def report_known_summary(
    lookback: tf.Tensor,
    K: tf.Tensor,
    season_period: tf.Tensor,
    decay: tf.Tensor,
) -> None:
    """Print known hyperparameters (safe inside @tf.function)."""
    line = tf.strings.join(
        [
            "[Bonus2] Known | lookback=",
            _fmt_i(lookback),
            " K=",
            _fmt_i(K),
            " season_period=",
            _fmt_i(season_period),
            " decay=",
            _fmt4(decay),
        ]
    )
    tf.print(line)


def report_theta_summary(prefix: str, theta: dict[str, tf.Tensor]) -> None:
    """Print a one-line theta summary (safe inside @tf.function)."""
    s = theta_stats(theta)
    line = tf.strings.join(
        [
            tf.convert_to_tensor(prefix),
            " | b0=",
            _fmt4(s["b0"]),
            " bh=",
            _fmt4(s["bh"]),
            " bp=",
            _fmt4(s["bp"]),
            " wl_sd=",
            _fmt4(s["wl_sd"]),
            " am=",
            _fmt4(s["am"]),
            " bm=",
            _fmt4(s["bm"]),
        ]
    )
    tf.print(line)


def report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
    """Print a one-line summary for the current iteration (safe inside @tf.function)."""
    theta = model.unconstrained_to_theta(z)
    s = theta_stats(theta)
    line = tf.strings.join(
        [
            "[Bonus2] it=",
            _fmt_i(it),
            " | b0=",
            _fmt4(s["b0"]),
            " bh=",
            _fmt4(s["bh"]),
            " bp=",
            _fmt4(s["bp"]),
            " wl_sd=",
            _fmt4(s["wl_sd"]),
            " am=",
            _fmt4(s["am"]),
            " bm=",
            _fmt4(s["bm"]),
        ]
    )
    tf.print(line)
