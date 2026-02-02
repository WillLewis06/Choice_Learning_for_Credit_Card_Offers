from __future__ import annotations
from typing import Callable, Tuple
import tensorflow as tf
from market_shock_estimators.lu_updates import (
    update_E_bar,
    update_beta,
    update_njt,
    update_r,
)
from market_shock_estimators.lu_validate_input import (
    tune_k_validate_input,
    tune_shrinkage_validate_input,
)


def _lu_k0(d: tf.Tensor) -> tf.Tensor:
    """Lu default initialization: k0 = 2.38 / sqrt(d)."""
    d = tf.cast(d, tf.float64)
    return tf.constant(2.38, tf.float64) / tf.sqrt(
        tf.maximum(d, tf.constant(1.0, tf.float64))
    )


def _leading_none_shape_invariant(shape: tf.TensorShape) -> tf.TensorShape:
    """
    Shape invariant for tf.while_loop loop-carried state:
      - rank 0: unchanged
      - rank >= 1: first dim set to None, remaining dims preserved if known
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
    """
    Generic step-size tuner with one-line per-round diagnostics.

    Runs short pilot chains of length `pilot_length` and adapts scalar k so that the
    average per-iteration acceptance rate lies in [target_low, target_high].

    Conventions:
      - step_fn(theta, k) -> (theta_new, acc_inc)
      - acc_inc is a float64 scalar in [0, 1]
        * for scalar proposals: 0/1
        * for batched proposals: mean acceptance across the batch in that step
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

    theta_inv_shape = _leading_none_shape_invariant(theta.shape)
    factor_t = tf.constant(float(factor), tf.float64)

    # Compile one pilot runner per step_fn to avoid passing Python callables as
    # tf.function arguments.
    @tf.function(reduce_retracing=True)
    def _pilot(theta_in: tf.Tensor, k_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
            f"[LuShrinkage:Tune:{name}] "
            f"round={round_id} | "
            f"k={k_before:.4f}->{k_after:.4f} | "
            f"acc={acc_rate:.3f} | "
            f"action={action}"
        )

        last_acc_rate = acc_rate
        last_theta = theta_end
        theta = theta_end

        if action == "ok":
            break

    return k


def tune_shrinkage(shrink):
    """
    Tune proposal scales for the Lu shrinkage sampler (tune once, then freeze).

    Returns:
      (k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned)
    """

    tune_shrinkage_validate_input(shrink)

    pilot_length = shrink.pilot_length
    ridge_t = tf.convert_to_tensor(shrink.ridge, dtype=tf.float64)
    target_low = shrink.target_low
    target_high = shrink.target_high
    max_rounds = shrink.max_rounds
    factor_rw = shrink.factor_rw
    factor_tmh = shrink.factor_tmh

    # Initial k's (Lu scaling).
    k_r0 = _lu_k0(tf.constant(1.0, tf.float64))
    k_beta0 = _lu_k0(tf.constant(2.0, tf.float64))
    k_E_bar0 = _lu_k0(tf.constant(1.0, tf.float64))

    # J for k_njt0 (static by contract via __init__ validation).
    J_int = int(shrink.J)
    k_njt0 = _lu_k0(tf.constant(J_int, dtype=tf.float64))

    # Fixed data tensors.
    qjt = shrink.qjt
    q0t = shrink.q0t
    pjt = shrink.pjt
    wjt = shrink.wjt

    # Fixed "other" parameters for each tuning run (do not mutate shrink state).
    beta_p0 = shrink.beta_p.read_value()
    beta_w0 = shrink.beta_w.read_value()
    r0 = shrink.r.read_value()
    E_bar0 = shrink.E_bar.read_value()
    njt0 = shrink.njt.read_value()
    gamma0 = shrink.gamma.read_value()
    phi0 = shrink.phi.read_value()

    posterior = shrink.posterior
    rng = shrink.rng

    # ---- step_r: scalar RW-MH; acc_inc ∈ {0,1}
    def step_r(theta_r: tf.Tensor, k_r: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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

    # ---- step_beta: 2D TMH; acc_inc ∈ {0,1}
    def step_beta(
        theta_beta: tf.Tensor, k_beta: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

    # ---- step_E_bar: batched RW-MH over (T,); acc_inc = mean accept across markets
    def step_E_bar(
        theta_E_bar: tf.Tensor, k_E_bar: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

    # ---- step_njt: market sweep TMH; acc_inc = mean accept across markets
    def step_njt(theta_njt: tf.Tensor, k_njt: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
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
