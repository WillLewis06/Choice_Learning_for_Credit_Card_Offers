"""
Proposal-scale tuning for the choice-learn + Lu sparse-shock sampler.

This module tunes the scalar proposal scales (k_*) used by the MCMC kernels.
Tuning is performed once before sampling and then frozen.

Tuning objective:
  - Run a short pilot chain of length `pilot_length`.
  - Compute the average per-iteration acceptance rate.
  - Adapt k multiplicatively until the acceptance rate lies in
    [target_low, target_high], or until `max_rounds` is reached.

Accepted conventions:
  - step_fn(theta, k) -> (theta_new, acc_inc)
  - acc_inc is a float64 scalar in [0,1]
      * scalar proposals: 0/1
      * batched proposals: mean acceptance across the batch for that step
      * sweep proposals: mean acceptance across the sweep for that step

The tuning code deliberately avoids mutating the sampler state. Each tuning run
operates on local copies of the relevant parameter block with all other blocks
held fixed.
"""

from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf

from lu.choice_learn.cl_updates import (
    update_alpha,
    update_E_bar,
    update_njt,
)
from lu.choice_learn.cl_validate_input import (
    tune_k_validate_input,
    tune_shrinkage_validate_input,
)


def _lu_k0(d: tf.Tensor) -> tf.Tensor:
    """Return the Lu default initialization k0 = 2.38 / sqrt(d)."""
    d = tf.cast(d, tf.float64)
    return tf.constant(2.38, tf.float64) / tf.sqrt(
        tf.maximum(d, tf.constant(1.0, tf.float64))
    )


def _leading_none_shape_invariant(shape: tf.TensorShape) -> tf.TensorShape:
    """Construct a safe shape invariant for loop-carried `theta`."""
    if shape.rank is None:
        return tf.TensorShape(None)
    if shape.rank == 0:
        return shape
    dims = shape.as_list()
    dims[0] = None
    return tf.TensorShape(dims)


def tune_k(
    theta0: tf.Tensor,
    step_fn: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
    k0: tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    name: str,
) -> tf.Tensor:
    """Tune a scalar proposal scale k for a single parameter block."""
    k = tf.convert_to_tensor(k0, dtype=tf.float64)
    theta = tf.convert_to_tensor(theta0, dtype=tf.float64)

    tune_k_validate_input(
        k0=k,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name=name,
    )

    pilot_length_t = tf.constant(int(pilot_length), dtype=tf.int32)

    theta_inv_shape = _leading_none_shape_invariant(theta.shape)
    factor_t = tf.constant(float(factor), tf.float64)

    @tf.function(reduce_retracing=True)
    def _pilot(theta_in: tf.Tensor, k_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run `pilot_length` steps of step_fn and return (theta_end, acc_sum)."""
        i0 = tf.constant(0, tf.int32)
        acc0 = tf.constant(0.0, tf.float64)

        def cond(i, theta_cur, acc_sum):
            return i < pilot_length_t

        def body(i, theta_cur, acc_sum):
            theta_new, acc_inc = step_fn(theta_cur, k_in)
            acc_sum = acc_sum + tf.cast(acc_inc, tf.float64)
            return i + 1, theta_new, acc_sum

        _, theta_out, acc_sum = tf.while_loop(
            cond,
            body,
            loop_vars=(i0, theta_in, acc0),
            shape_invariants=(i0.shape, theta_inv_shape, acc0.shape),
            parallel_iterations=1,
        )
        return theta_out, acc_sum

    for round_id in range(int(max_rounds)):
        theta_end, acc_sum = _pilot(theta, k)
        acc_rate = float((acc_sum / tf.cast(pilot_length_t, tf.float64)).numpy())

        k_before = float(k.numpy())

        if acc_rate < target_low:
            action = "shrink"
            k = k / factor_t
        elif acc_rate > target_high:
            action = "grow"
            k = k * factor_t
        else:
            action = "ok"

        k_after = float(k.numpy())

        print(
            f"[ChoiceLearnShrinkage:Tune:{name}] "
            f"round={round_id} | "
            f"k={k_before:.4f}->{k_after:.4f} | "
            f"acc={acc_rate:.3f} | "
            f"action={action}"
        )

        theta = theta_end

        if action == "ok":
            break

    return k


def tune_shrinkage(shrink):
    """Tune proposal scales for the choice-learn + Lu sparse-shock sampler.

    Tuned k values are returned in the order used by the sampler:
      (k_alpha_tuned, k_E_bar_tuned, k_njt_tuned)
    """
    tune_shrinkage_validate_input(shrink)

    pilot_length = shrink.pilot_length
    ridge_t = tf.convert_to_tensor(shrink.ridge, dtype=tf.float64)
    target_low = shrink.target_low
    target_high = shrink.target_high
    max_rounds = shrink.max_rounds
    factor_rw = shrink.factor_rw
    factor_tmh = shrink.factor_tmh

    # Initial k's use Lu's dimension scaling heuristic.
    k_alpha0 = _lu_k0(tf.constant(1.0, tf.float64))  # scalar
    k_E_bar0 = _lu_k0(tf.constant(1.0, tf.float64))  # elementwise (T,)

    J_int = int(shrink.J)
    k_njt0 = _lu_k0(tf.constant(J_int, dtype=tf.float64))  # one market's (J,) block

    # Fixed data tensors.
    qjt = shrink.qjt
    q0t = shrink.q0t
    delta_cl = shrink.delta_cl

    # Snapshot current state. Tuning holds all non-target blocks fixed.
    alpha0 = shrink.alpha.read_value()
    E_bar0 = shrink.E_bar.read_value()
    njt0 = shrink.njt.read_value()
    gamma0 = shrink.gamma.read_value()
    phi0 = shrink.phi.read_value()

    posterior = shrink.posterior
    rng = shrink.rng

    # ------------------------------------------------------------
    # alpha: scalar RW-MH (acc_inc is 0/1)
    # ------------------------------------------------------------
    def step_alpha(
        theta_alpha: tf.Tensor, k_alpha: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        alpha_new, accepted = update_alpha(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=theta_alpha,
            E_bar=E_bar0,
            njt=njt0,
            k_alpha=k_alpha,
        )
        return alpha_new, tf.cast(accepted, tf.float64)

    k_alpha_tuned = tune_k(
        theta0=alpha0,
        step_fn=step_alpha,
        k0=k_alpha0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_rw,
        name="alpha",
    )

    # ------------------------------------------------------------
    # E_bar: batched RW-MH over (T,) (acc_inc is mean accepted across markets)
    # ------------------------------------------------------------
    def step_E_bar(
        theta_E_bar: tf.Tensor, k_E_bar: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        E_bar_new, accepted_vec = update_E_bar(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha0,
            E_bar=theta_E_bar,
            njt=njt0,
            gamma=gamma0,
            phi=phi0,
            k_E_bar=k_E_bar,
        )
        acc_inc = tf.reduce_mean(tf.cast(accepted_vec, tf.float64))
        return E_bar_new, acc_inc

    k_E_bar_tuned = tune_k(
        theta0=E_bar0,
        step_fn=step_E_bar,
        k0=k_E_bar0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_rw,
        name="E_bar",
    )

    # ------------------------------------------------------------
    # njt: TMH sweep across markets (acc_inc is mean accepted across markets)
    # ------------------------------------------------------------
    def step_njt(theta_njt: tf.Tensor, k_njt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        njt_new, acc_sum = update_njt(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha0,
            E_bar=E_bar0,
            njt=theta_njt,
            gamma=gamma0,
            phi=phi0,
            k_njt=k_njt,
            ridge=ridge_t,
        )
        T_t = tf.cast(tf.shape(theta_njt)[0], tf.float64)
        acc_inc = acc_sum / tf.maximum(T_t, tf.constant(1.0, tf.float64))
        return njt_new, acc_inc

    k_njt_tuned = tune_k(
        theta0=njt0,
        step_fn=step_njt,
        k0=k_njt0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_tmh,
        name="njt",
    )

    return k_alpha_tuned, k_E_bar_tuned, k_njt_tuned
