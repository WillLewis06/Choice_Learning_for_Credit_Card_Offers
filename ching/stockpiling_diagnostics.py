"""
Diagnostics and reporting for the Ching stockpiling sampler.

This module provides TF-compatible reporting utilities intended to run inside
tf.function. It does not maintain running sums or posterior means; accumulation
is owned by the estimator.

There is no burn-in or thinning logic here.
"""

from __future__ import annotations

from typing import Mapping, Protocol

import tensorflow as tf


class _HasZ(Protocol):
    """Minimal structural type for objects exposing unconstrained parameters z."""

    z: Mapping[str, tf.Tensor]


def _round4(x: tf.Tensor) -> tf.Tensor:
    """
    Format tensor values as strings with 4 decimal places (no scientific notation).

    Returns:
      tf.Tensor: scalar string tensor suitable for tf.print.
    """
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(estimator: _HasZ, it: int | tf.Tensor) -> None:
    """
    Print a one-line summary of the current MCMC state.

    Designed to be cheap and informative:
      - global means of beta, alpha, v, fc, lambda_c (post-transform)
      - mean u_scale (post-transform)

    Implementation notes:
      - Uses tf.print so it can run inside tf.function.
      - Avoids .numpy() and Python-side formatting.
    """
    z = estimator.z

    beta = tf.math.sigmoid(z["z_beta"])
    alpha = tf.exp(z["z_alpha"])
    v = tf.exp(z["z_v"])
    fc = tf.exp(z["z_fc"])
    lambda_c = tf.math.sigmoid(z["z_lambda"])
    u_scale = tf.exp(z["z_u_scale"])

    beta_mean = tf.reduce_mean(beta)
    alpha_mean = tf.reduce_mean(alpha)
    v_mean = tf.reduce_mean(v)
    fc_mean = tf.reduce_mean(fc)
    lambda_c_mean = tf.reduce_mean(lambda_c)
    u_scale_mean = tf.reduce_mean(u_scale)

    tf.print(
        "[Stockpiling] it=",
        it,
        " | mean(beta)=",
        _round4(beta_mean),
        ", mean(alpha)=",
        _round4(alpha_mean),
        ", mean(v)=",
        _round4(v_mean),
        ", mean(fc)=",
        _round4(fc_mean),
        ", mean(lambda_c)=",
        _round4(lambda_c_mean),
        " | u_scale_mean=",
        _round4(u_scale_mean),
    )
