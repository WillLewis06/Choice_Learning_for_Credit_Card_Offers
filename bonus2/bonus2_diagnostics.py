"""
bonus2/bonus2_diagnostics.py

Diagnostics printing for Bonus Q2.

Target output format (no quotes, fixed decimals, compact tokens):

[Bonus2] True | mean beta_market=-0.008 beta_habit=-0.009 beta_peer=-0.003 decay=0.874 a_m=0.000 b_m=0.000 | std beta_dow_m=0.000 beta_dow_j=0.000 a_j=0.000 b_j=0.000
[Bonus2] Init | mean ...
[Bonus2] Known | L=2 , K=1 , season_period=365 , kappa_decay=7.0000
[Bonus2] it=0 | mean ... | std ...

Key constraints:
- No string-tensor printing (tf.print on string tensors adds quotes).
- Iteration printing happens inside tf.function, so use tf.py_function to do Python-side formatting/printing.
- True/Init/Known are typically printed from orchestration (eager Python), so use Python print directly.
- Apply identifiability constraints before reporting std for:
    beta_dow_m, beta_dow_j, a_j, b_j
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from bonus2 import bonus2_model as model


# =============================================================================
# NumPy-side identifiability (for True/Init prints from orchestration)
# =============================================================================


def _apply_identification_np(
    beta_dow_m: np.ndarray,
    beta_dow_j: np.ndarray,
    a_j: np.ndarray,
    b_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror model.apply_identifiability_constraints_tf in NumPy.

    - beta_dow_m: center within market over weekdays (axis=1)
    - beta_dow_j: center within product over weekdays (axis=1),
                  then center across products per weekday (axis=0)
    - a_j, b_j:   center across products per harmonic (axis=0)
    """
    beta_dow_m = np.asarray(beta_dow_m, dtype=np.float64)
    beta_dow_j = np.asarray(beta_dow_j, dtype=np.float64)
    a_j = np.asarray(a_j, dtype=np.float64)
    b_j = np.asarray(b_j, dtype=np.float64)

    if beta_dow_m.size:
        beta_dow_m = beta_dow_m - beta_dow_m.mean(axis=1, keepdims=True)

    if beta_dow_j.size:
        beta_dow_j = beta_dow_j - beta_dow_j.mean(axis=1, keepdims=True)
        beta_dow_j = beta_dow_j - beta_dow_j.mean(axis=0, keepdims=True)

    if a_j.size:
        a_j = a_j - a_j.mean(axis=0, keepdims=True)
    if b_j.size:
        b_j = b_j - b_j.mean(axis=0, keepdims=True)

    return beta_dow_m, beta_dow_j, a_j, b_j


# =============================================================================
# Formatting (Python)
# =============================================================================


def _clean_float(x: float, nd: int) -> float:
    # Avoid "-0.000"
    if abs(x) < 0.5 * (10.0 ** (-nd)):
        return 0.0
    return x


def _fmt(x: float, nd: int) -> str:
    return f"{_clean_float(float(x), nd):.{nd}f}"


def _format_line(tag: str, vals: Mapping[str, float], nd: int = 3) -> str:
    return (
        f"[Bonus2] {tag} | mean "
        f"beta_market={_fmt(vals['beta_market'], nd)} "
        f"beta_habit={_fmt(vals['beta_habit'], nd)} "
        f"beta_peer={_fmt(vals['beta_peer'], nd)} "
        f"decay={_fmt(vals['decay'], nd)} "
        f"a_m={_fmt(vals['a_m'], nd)} "
        f"b_m={_fmt(vals['b_m'], nd)}"
        f" | std "
        f"beta_dow_m={_fmt(vals['beta_dow_m'], nd)} "
        f"beta_dow_j={_fmt(vals['beta_dow_j'], nd)} "
        f"a_j={_fmt(vals['a_j'], nd)} "
        f"b_j={_fmt(vals['b_j'], nd)}"
    )


# =============================================================================
# Public API (orchestration prints: True/Init/Known)
# =============================================================================


def report_theta_summary(stage: str, theta_or_init: Mapping[str, Any]) -> None:
    """
    Print True/Init using the target one-line format.

    - If theta_or_init contains expanded keys (e.g., 'beta_market_mj'), it is treated as theta.
    - Otherwise it is treated as scalar init dict with keys:
        beta_market, beta_habit, beta_peer, decay_rate, a_m, b_m
      and std-block is zeros.
    """
    if "beta_market_mj" in theta_or_init:
        beta_market_mj = np.asarray(theta_or_init["beta_market_mj"], dtype=np.float64)
        beta_habit_j = np.asarray(theta_or_init["beta_habit_j"], dtype=np.float64)
        beta_peer_j = np.asarray(theta_or_init["beta_peer_j"], dtype=np.float64)
        decay_rate_j = np.asarray(theta_or_init["decay_rate_j"], dtype=np.float64)
        a_m = np.asarray(theta_or_init["a_m"], dtype=np.float64)
        b_m = np.asarray(theta_or_init["b_m"], dtype=np.float64)

        beta_dow_m = np.asarray(theta_or_init["beta_dow_m"], dtype=np.float64)
        beta_dow_j = np.asarray(theta_or_init["beta_dow_j"], dtype=np.float64)
        a_j = np.asarray(theta_or_init["a_j"], dtype=np.float64)
        b_j = np.asarray(theta_or_init["b_j"], dtype=np.float64)

        beta_dow_m, beta_dow_j, a_j, b_j = _apply_identification_np(
            beta_dow_m=beta_dow_m,
            beta_dow_j=beta_dow_j,
            a_j=a_j,
            b_j=b_j,
        )

        vals = {
            "beta_market": float(beta_market_mj.mean()) if beta_market_mj.size else 0.0,
            "beta_habit": float(beta_habit_j.mean()) if beta_habit_j.size else 0.0,
            "beta_peer": float(beta_peer_j.mean()) if beta_peer_j.size else 0.0,
            "decay": float(decay_rate_j.mean()) if decay_rate_j.size else 0.0,
            "a_m": float(a_m.mean()) if a_m.size else 0.0,
            "b_m": float(b_m.mean()) if b_m.size else 0.0,
            "beta_dow_m": float(beta_dow_m.std()) if beta_dow_m.size else 0.0,
            "beta_dow_j": float(beta_dow_j.std()) if beta_dow_j.size else 0.0,
            "a_j": float(a_j.std()) if a_j.size else 0.0,
            "b_j": float(b_j.std()) if b_j.size else 0.0,
        }
        print(_format_line(stage, vals, nd=3), flush=True)
        return

    # Scalar init dict
    vals = {
        "beta_market": float(theta_or_init["beta_market"]),
        "beta_habit": float(theta_or_init["beta_habit"]),
        "beta_peer": float(theta_or_init["beta_peer"]),
        "decay": float(theta_or_init["decay_rate"]),
        "a_m": float(theta_or_init["a_m"]),
        "b_m": float(theta_or_init["b_m"]),
        "beta_dow_m": 0.0,
        "beta_dow_j": 0.0,
        "a_j": 0.0,
        "b_j": 0.0,
    }
    print(_format_line(stage, vals, nd=3), flush=True)


def report_known_summary(
    L: int, K: int, season_period: int, kappa_decay: float
) -> None:
    print(
        f"[Bonus2] Known | L={int(L)} , K={int(K)} , season_period={int(season_period)} , kappa_decay={float(kappa_decay):.4f}",
        flush=True,
    )


# =============================================================================
# Iteration printing (inside tf.function): use tf.py_function for exact formatting
# =============================================================================


def _py_print_iter(it: np.ndarray, *vals: np.ndarray) -> np.int32:
    it_i = int(it)
    keys = [
        "beta_market",
        "beta_habit",
        "beta_peer",
        "decay",
        "a_m",
        "b_m",
        "beta_dow_m",
        "beta_dow_j",
        "a_j",
        "b_j",
    ]
    out: dict[str, float] = {}
    for k, v in zip(keys, vals, strict=True):
        out[k] = float(np.asarray(v).reshape(()))
    print(_format_line(f"it={it_i}", out, nd=3), flush=True)
    return np.int32(0)


def report_iteration_progress(z: Mapping[str, tf.Tensor], it: tf.Tensor) -> None:
    """
    Print:
      [Bonus2] it=<it> | mean ... | std ...

    Uses model.unconstrained_to_theta(z) and applies identifiability constraints (TF-side),
    then passes scalars to Python via tf.py_function for exact formatting (no quotes).
    """
    it = tf.cast(it, tf.int32)
    z64 = {k: tf.cast(v, tf.float64) for k, v in z.items()}
    theta = model.unconstrained_to_theta(z64)

    # Apply identifiability (TF-side, consistent with model)
    beta_dow_m, beta_dow_j, a_j, b_j = model.apply_identifiability_constraints_tf(
        beta_dow_m=tf.cast(theta["beta_dow_m"], tf.float64),
        beta_dow_j=tf.cast(theta["beta_dow_j"], tf.float64),
        a_j=tf.cast(theta["a_j"], tf.float64),
        b_j=tf.cast(theta["b_j"], tf.float64),
    )

    # Scalars (TF-side)
    beta_market = tf.reduce_mean(tf.cast(theta["beta_market_mj"], tf.float64))
    beta_habit = tf.reduce_mean(tf.cast(theta["beta_habit_j"], tf.float64))
    beta_peer = tf.reduce_mean(tf.cast(theta["beta_peer_j"], tf.float64))
    decay = tf.reduce_mean(tf.cast(theta["decay_rate_j"], tf.float64))
    a_m = tf.reduce_mean(tf.cast(theta["a_m"], tf.float64))
    b_m = tf.reduce_mean(tf.cast(theta["b_m"], tf.float64))

    std_beta_dow_m = tf.math.reduce_std(beta_dow_m)
    std_beta_dow_j = tf.math.reduce_std(beta_dow_j)
    std_a_j = tf.math.reduce_std(a_j)
    std_b_j = tf.math.reduce_std(b_j)

    # Python-side formatted print
    _ = tf.py_function(
        func=_py_print_iter,
        inp=[
            it,
            beta_market,
            beta_habit,
            beta_peer,
            decay,
            a_m,
            b_m,
            std_beta_dow_m,
            std_beta_dow_j,
            std_a_j,
            std_b_j,
        ],
        Tout=tf.int32,
    )
