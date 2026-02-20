"""
ching/stockpiling_diagnostics.py

Printing utilities for monitoring the stockpiling MCMC sampler.
"""

from __future__ import annotations

import tensorflow as tf

from ching.stockpiling_model import unconstrained_to_theta


def _fmt4(x: tf.Tensor) -> tf.Tensor:
    """Format numeric tensor values with 4 decimals (no scientific notation)."""
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
    """Print a compact summary of the current MCMC state.

    Args:
      z: unconstrained parameter dict with keys:
           z_beta (1,), z_alpha (J,), z_v (J,), z_fc (J,), z_u_scale (M,)
      it: iteration counter (scalar)
    """
    it_t = tf.cast(tf.convert_to_tensor(it), tf.int32)

    theta = unconstrained_to_theta(z)

    beta = tf.reshape(theta["beta"], ())
    mean_alpha = tf.reduce_mean(theta["alpha"])
    mean_v = tf.reduce_mean(theta["v"])
    mean_fc = tf.reduce_mean(theta["fc"])
    mean_u_scale = tf.reduce_mean(theta["u_scale"])

    tf.print(
        "[Stockpiling] it=",
        it_t,
        " | beta=",
        _fmt4(beta),
        ", mean(alpha)=",
        _fmt4(mean_alpha),
        ", mean(v)=",
        _fmt4(mean_v),
        ", mean(fc)=",
        _fmt4(mean_fc),
        " | mean(u_scale)=",
        _fmt4(mean_u_scale),
    )
