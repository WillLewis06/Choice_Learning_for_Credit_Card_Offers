"""
Proposal-scale tuning for the Lu shrinkage sampler.

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

from lu.shrinkage.lu_updates import (
    update_E_bar,
    update_beta,
    update_njt,
    update_r,
)
from lu.shrinkage.lu_validate_input import (
    tune_k_validate_input,
    tune_shrinkage_validate_input,
)


def _lu_k0(d: tf.Tensor) -> tf.Tensor:
    """Return the Lu default initialization k0 = 2.38 / sqrt(d).

    This is the common random-walk scaling heuristic, where d is the dimension
    of the parameter block being updated. For scalar updates, d=1.
    """
    d = tf.cast(d, tf.float64)
    return tf.constant(2.38, tf.float64) / tf.sqrt(
        tf.maximum(d, tf.constant(1.0, tf.float64))
    )


def _leading_none_shape_invariant(shape: tf.TensorShape) -> tf.TensorShape:
    """Construct a safe shape invariant for loop-carried `theta`.

    `tune_k` runs a tf.while_loop over pilot iterations. For batched or unknown
    leading dimensions, the loop-carried tensor may have a dynamic first axis.
    This helper:
      - leaves rank-0 shapes unchanged,
      - for rank>=1 sets the first dim to None and preserves known trailing dims.

    This is only used for `theta` (not for scalars like k or acc_sum).
    """
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
    """Tune a scalar proposal scale k for a single parameter block.

    This is a generic routine used for all blocks (r, beta, E_bar, njt). It
    repeatedly runs a short pilot chain and updates k by a multiplicative factor:

      if acc_rate < target_low:   k <- k / factor
      if acc_rate > target_high:  k <- k * factor
      else:                       stop

    The pilot runner is compiled once per step_fn (closure) so the Python callable
    is not passed through TF tracing each round.

    Args:
        theta0: Initial state for this block (shape depends on the block).
        step_fn: Function implementing one MCMC step for this block.
        k0: Initial proposal scale.
        pilot_length: Number of iterations per pilot chain.
        target_low: Lower bound for the target acceptance rate.
        target_high: Upper bound for the target acceptance rate.
        max_rounds: Maximum number of tuning rounds.
        factor: Multiplicative update factor (>1).
        name: Short label used only for printed diagnostics.

    Returns:
        k: Tuned proposal scale as a scalar tf.float64 tensor.
    """
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

    # theta can be a vector/matrix with a dynamic leading dimension. This shape
    # invariant keeps TF while_loop happy without over-constraining shapes.
    theta_inv_shape = _leading_none_shape_invariant(theta.shape)
    factor_t = tf.constant(float(factor), tf.float64)

    @tf.function(reduce_retracing=True)
    def _pilot(theta_in: tf.Tensor, k_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Run `pilot_length` steps of step_fn and return (theta_end, acc_sum)."""
        i0 = tf.constant(0, tf.int32)
        acc0 = tf.constant(0.0, tf.float64)

        def cond(i, theta_cur, acc_sum):
            """Continue until the pilot chain reaches pilot_length steps."""
            return i < pilot_length_t

        def body(i, theta_cur, acc_sum):
            """Advance the pilot chain by one step and accumulate acceptance."""
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
        # Run a pilot chain at the current k.
        theta_end, acc_sum = _pilot(theta, k)
        acc_rate = float((acc_sum / tf.cast(pilot_length_t, tf.float64)).numpy())

        # Record k before/after for one-line logging.
        k_before = float(k.numpy())

        # Multiplicative adjustment toward the target acceptance interval.
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
            f"[LuShrinkage:Tune:{name}] "
            f"round={round_id} | "
            f"k={k_before:.4f}->{k_after:.4f} | "
            f"acc={acc_rate:.3f} | "
            f"action={action}"
        )

        # Carry the final theta into the next round so the pilot chains continue
        # from a representative state rather than restarting at theta0.
        theta = theta_end

        if action == "ok":
            break

    return k


def tune_shrinkage(shrink):
    """Tune proposal scales for the Lu shrinkage sampler.

    This function constructs per-block step functions that:
      - update only the target block,
      - hold all other blocks fixed using read_value() snapshots, and
      - return a scalar acceptance increment in [0,1].

    The tuned k values are returned in the order used by the sampler:
      (k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned)

    Returns:
        k_r_tuned: Tuned RW-MH scale for r.
        k_E_bar_tuned: Tuned RW-MH scale for E_bar.
        k_beta_tuned: Tuned TMH scale for (beta_p, beta_w).
        k_njt_tuned: Tuned TMH scale for njt market updates.
    """
    tune_shrinkage_validate_input(shrink)

    # Read tuning configuration from the estimator instance.
    pilot_length = shrink.pilot_length
    ridge_t = tf.convert_to_tensor(shrink.ridge, dtype=tf.float64)
    target_low = shrink.target_low
    target_high = shrink.target_high
    max_rounds = shrink.max_rounds
    factor_rw = shrink.factor_rw
    factor_tmh = shrink.factor_tmh

    # Initial k's use Lu's dimension scaling heuristic.
    k_r0 = _lu_k0(tf.constant(1.0, tf.float64))  # scalar
    k_beta0 = _lu_k0(tf.constant(2.0, tf.float64))  # (beta_p, beta_w)
    k_E_bar0 = _lu_k0(tf.constant(1.0, tf.float64))  # elementwise (T,)

    # k_njt0 depends on J (dimension of one market's njt_t vector).
    J_int = int(shrink.J)
    k_njt0 = _lu_k0(tf.constant(J_int, dtype=tf.float64))

    # Fixed data tensors.
    qjt = shrink.qjt
    q0t = shrink.q0t
    pjt = shrink.pjt
    wjt = shrink.wjt

    # Snapshot current state. Tuning uses these fixed values for all non-target blocks.
    beta_p0 = shrink.beta_p.read_value()
    beta_w0 = shrink.beta_w.read_value()
    r0 = shrink.r.read_value()
    E_bar0 = shrink.E_bar.read_value()
    njt0 = shrink.njt.read_value()
    gamma0 = shrink.gamma.read_value()
    phi0 = shrink.phi.read_value()

    posterior = shrink.posterior
    rng = shrink.rng

    # ------------------------------------------------------------
    # r: scalar RW-MH (acc_inc is 0/1)
    # ------------------------------------------------------------
    def step_r(theta_r: tf.Tensor, k_r: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """One pilot step for r, returning (r_new, acc_inc)."""
        r_new, accepted = update_r(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p0,
            beta_w=beta_w0,
            r=theta_r,
            E_bar=E_bar0,
            njt=njt0,
            k_r=k_r,
        )
        return r_new, tf.cast(accepted, tf.float64)

    k_r_tuned = tune_k(
        theta0=r0,
        step_fn=step_r,
        k0=k_r0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_rw,
        name="r",
    )

    # ------------------------------------------------------------
    # beta: 2D TMH (acc_inc is 0/1)
    # ------------------------------------------------------------
    def step_beta(
        theta_beta: tf.Tensor, k_beta: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """One pilot step for (beta_p, beta_w), returning (beta_vec_new, acc_inc)."""
        bp0 = theta_beta[0]
        bw0 = theta_beta[1]
        bp_new, bw_new, accepted = update_beta(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=bp0,
            beta_w=bw0,
            r=r0,
            E_bar=E_bar0,
            njt=njt0,
            k_beta=k_beta,
            ridge=ridge_t,
        )
        theta_new = tf.stack([bp_new, bw_new], axis=0)
        return theta_new, tf.cast(accepted, tf.float64)

    beta_vec0 = tf.stack([beta_p0, beta_w0], axis=0)
    k_beta_tuned = tune_k(
        theta0=beta_vec0,
        step_fn=step_beta,
        k0=k_beta0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_tmh,
        name="beta",
    )

    # ------------------------------------------------------------
    # E_bar: batched RW-MH over (T,) (acc_inc is mean accepted across markets)
    # ------------------------------------------------------------
    def step_E_bar(
        theta_E_bar: tf.Tensor, k_E_bar: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """One pilot step for E_bar, returning (E_bar_new, acc_inc)."""
        E_bar_new, accepted_vec = update_E_bar(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p0,
            beta_w=beta_w0,
            r=r0,
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
        """One pilot step for njt, returning (njt_new, acc_inc)."""
        njt_new, acc_sum = update_njt(
            posterior=posterior,
            rng=rng,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p0,
            beta_w=beta_w0,
            r=r0,
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

    return k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned
