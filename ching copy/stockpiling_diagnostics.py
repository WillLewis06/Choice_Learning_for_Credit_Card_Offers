"""
Diagnostics and reporting utilities for the stockpiling sampler.

This module is TF-compatible (safe inside tf.function). It reports on the *current*
unconstrained state z by transforming to constrained theta via:

  ching.stockpiling_model.unconstrained_to_theta

This module:
  - does not maintain running sums or posterior means (owned by the estimator)
  - contains no burn-in/thinning logic
"""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf

from ching import stockpiling_model as model


def _fmt4(x: tf.Tensor) -> tf.Tensor:
    """Format tensor values as strings with 4 decimals (no scientific notation)."""
    x = tf.cast(x, tf.float64)
    return tf.strings.as_string(x, precision=4, scientific=False)


def report_iteration_progress(
    z: Mapping[str, tf.Tensor],
    it: tf.Tensor,
    *,
    lambda_by_market_k: int = 3,
) -> None:
    """
    Print a compact summary of the current MCMC state.

    Prints:
      - Line 1: global means of constrained parameters (beta, alpha, v, fc, lambda, u_scale)
      - Line 2: first k entries of per-market mean(lambda) for heterogeneity visibility

    Args:
      z: mapping containing z_* tensors:
           z_beta, z_alpha, z_v, z_fc (M,J)
           z_lambda (M,N)
           z_u_scale (M,)
      it: iteration counter (tf.Tensor scalar)
      lambda_by_market_k: number of markets to print in the second line
    """
    it = tf.cast(it, tf.int32)

    # Ensure float64 for stable transforms and summaries.
    z64 = {k: tf.cast(v, tf.float64) for k, v in z.items()}
    theta = model.unconstrained_to_theta(z64)

    # Schema-driven global means (stable ordering).
    spec = [
        ("beta", "beta"),
        ("alpha", "alpha"),
        ("v", "v"),
        ("fc", "fc"),
        ("lambda", "lambda"),
        ("u_scale", "u_scale"),
    ]
    means = {name: tf.reduce_mean(theta[key]) for (name, key) in spec}

    tf.print(
        "[Stockpiling] it=",
        it,
        " | mean(beta)=",
        _fmt4(means["beta"]),
        ", mean(alpha)=",
        _fmt4(means["alpha"]),
        ", mean(v)=",
        _fmt4(means["v"]),
        ", mean(fc)=",
        _fmt4(means["fc"]),
        ", mean(lambda)=",
        _fmt4(means["lambda"]),
        " | mean(u_scale)=",
        _fmt4(means["u_scale"]),
    )
