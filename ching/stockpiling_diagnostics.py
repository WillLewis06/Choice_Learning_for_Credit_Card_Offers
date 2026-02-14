"""
Diagnostics and reporting utilities for the stockpiling sampler.

This module is TF-compatible (safe inside tf.function). It reports on the *current*
unconstrained state z by transforming to constrained theta via:

  stockpiling_model.unconstrained_to_theta

"""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf


from ching.stockpiling_model import unconstrained_to_theta


def _fmt4(x: tf.Tensor) -> tf.Tensor:
    """Format tensor values as strings with 4 decimals (no scientific notation)."""
    x = tf.cast(x, tf.float64)
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(z: Mapping[str, tf.Tensor], it: tf.Tensor) -> None:
    """
    Print a compact summary of the current MCMC state.

    Args:
      z: mapping containing z_* tensors:
           z_beta (1,), z_alpha (J,), z_v (J,), z_fc (J,), z_u_scale (M,)
      it: iteration counter (tf.Tensor scalar)
    """
    it = tf.cast(it, tf.int32)

    z64 = {k: tf.cast(v, tf.float64) for k, v in z.items()}
    theta = unconstrained_to_theta(z64)

    mean_beta = tf.reduce_mean(theta["beta"])
    mean_alpha = tf.reduce_mean(theta["alpha"])
    mean_v = tf.reduce_mean(theta["v"])
    mean_fc = tf.reduce_mean(theta["fc"])
    mean_u_scale = tf.reduce_mean(theta["u_scale"])

    tf.print(
        "[Stockpiling] it=",
        it,
        " | mean(beta)=",
        _fmt4(mean_beta),
        ", mean(alpha)=",
        _fmt4(mean_alpha),
        ", mean(v)=",
        _fmt4(mean_v),
        ", mean(fc)=",
        _fmt4(mean_fc),
        " | mean(u_scale)=",
        _fmt4(mean_u_scale),
    )
