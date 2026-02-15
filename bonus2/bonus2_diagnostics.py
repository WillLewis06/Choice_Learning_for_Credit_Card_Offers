"""
bonus2_diagnostics.py

Diagnostics / printing helpers for Bonus Q2 (updated spec).

This file is intentionally simple:
  - No identifiability / centering adjustments (no longer needed under updated spec).
  - Prints lightweight summaries of theta at True / Init / per-iteration.

Updated theta keys:
  beta_market_j  (J,)
  beta_habit_j   (J,)
  beta_peer_j    (J,)
  beta_dow_j     (J,2)
  a_m            (M,K)
  b_m            (M,K)

Known inputs printed separately:
  L, K, season_period, decay
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from bonus2 import bonus2_model as model


def report_known_summary(L: int, K: int, season_period: int, decay: float) -> None:
    """Print known hyperparameters / inputs."""
    print(
        f"[Bonus2] Known | L={int(L)} , K={int(K)} , season_period={int(season_period)} , decay={float(decay):.4f}"
    )


def _theta_stats(theta: dict[str, Any]) -> dict[str, float]:
    """Compute a small set of scalar stats from theta."""
    beta_market_j = np.asarray(theta["beta_market_j"], dtype=np.float64)
    beta_habit_j = np.asarray(theta["beta_habit_j"], dtype=np.float64)
    beta_peer_j = np.asarray(theta["beta_peer_j"], dtype=np.float64)
    beta_dow_j = np.asarray(theta["beta_dow_j"], dtype=np.float64)
    a_m = np.asarray(theta["a_m"], dtype=np.float64)
    b_m = np.asarray(theta["b_m"], dtype=np.float64)

    return {
        "mean_beta_market": float(beta_market_j.mean()),
        "mean_beta_habit": float(beta_habit_j.mean()),
        "mean_beta_peer": float(beta_peer_j.mean()),
        "mean_a_m": float(a_m.mean()) if a_m.size else 0.0,
        "mean_b_m": float(b_m.mean()) if b_m.size else 0.0,
        "std_beta_dow_j": float(beta_dow_j.std()),
    }


def report_theta_summary(stage: str, theta: dict[str, Any]) -> None:
    """Print a concise theta summary."""
    s = _theta_stats(theta)
    print(
        f"[Bonus2] {stage} | mean beta_market={s['mean_beta_market']:.3f} "
        f"beta_habit={s['mean_beta_habit']:.3f} beta_peer={s['mean_beta_peer']:.3f} "
        f"a_m={s['mean_a_m']:.3f} b_m={s['mean_b_m']:.3f} | std beta_dow_j={s['std_beta_dow_j']:.3f}"
    )


def _py_print_iter(
    it: int,
    mean_beta_market: float,
    mean_beta_habit: float,
    mean_beta_peer: float,
    mean_a_m: float,
    mean_b_m: float,
    std_beta_dow_j: float,
) -> None:
    """Python-side print called through tf.py_function."""
    print(
        f"[Bonus2] it={int(it)} | mean beta_market={mean_beta_market:.3f} "
        f"beta_habit={mean_beta_habit:.3f} beta_peer={mean_beta_peer:.3f} "
        f"a_m={mean_a_m:.3f} b_m={mean_b_m:.3f} | std beta_dow_j={std_beta_dow_j:.3f}"
    )


def report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
    """Print a per-iteration summary of the current z -> theta."""
    z64 = {k: tf.cast(v, tf.float64) for k, v in z.items()}
    theta = model.unconstrained_to_theta(z64)

    mean_beta_market = tf.reduce_mean(theta["beta_market_j"])
    mean_beta_habit = tf.reduce_mean(theta["beta_habit_j"])
    mean_beta_peer = tf.reduce_mean(theta["beta_peer_j"])
    mean_a_m = (
        tf.reduce_mean(theta["a_m"])
        if tf.size(theta["a_m"]) > 0
        else tf.constant(0.0, tf.float64)
    )
    mean_b_m = (
        tf.reduce_mean(theta["b_m"])
        if tf.size(theta["b_m"]) > 0
        else tf.constant(0.0, tf.float64)
    )
    std_beta_dow_j = tf.math.reduce_std(theta["beta_dow_j"])

    tf.py_function(
        func=_py_print_iter,
        inp=[
            it,
            mean_beta_market,
            mean_beta_habit,
            mean_beta_peer,
            mean_a_m,
            mean_b_m,
            std_beta_dow_j,
        ],
        Tout=[],
    )
