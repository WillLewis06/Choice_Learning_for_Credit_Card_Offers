from __future__ import annotations

from dataclasses import replace
from typing import Callable

import tensorflow as tf

from lu.shrinkage.lu_updates import (
    E_bar_one_step,
    beta_one_step,
    njt_one_step,
    r_one_step,
)


def _lu_k0(d: tf.Tensor) -> tf.Tensor:
    d = tf.cast(d, tf.float64)
    return tf.constant(2.38, tf.float64) / tf.sqrt(
        tf.maximum(d, tf.constant(1.0, tf.float64))
    )


def _tune_block(
    theta0: tf.Tensor,
    step_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    k0: tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    name: str,
    seed: tf.Tensor | int | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta0)
    k = tf.convert_to_tensor(k0, dtype=theta.dtype)

    if seed is None:
        seed = tf.random.uniform(
            shape=(2,),
            minval=0,
            maxval=2**31 - 1,
            dtype=tf.int32,
        )
    else:
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        if seed.shape.rank == 0:
            seed = tf.stack([seed, tf.constant(0, dtype=tf.int32)])

    pilot_length_t = tf.constant(pilot_length, dtype=tf.int32)
    factor_t = tf.constant(factor, dtype=theta.dtype)
    target_low_t = tf.constant(target_low, dtype=theta.dtype)
    target_high_t = tf.constant(target_high, dtype=theta.dtype)

    @tf.function(reduce_retracing=True)
    def _pilot(
        theta_in: tf.Tensor,
        k_in: tf.Tensor,
        seed_in: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        i0 = tf.constant(0, dtype=tf.int32)
        acc0 = tf.constant(0.0, dtype=theta_in.dtype)

        def cond(i, _theta_cur, _acc_sum, _seed_cur):
            del _theta_cur, _acc_sum, _seed_cur
            return i < pilot_length_t

        def body(i, theta_cur, acc_sum, seed_cur):
            seeds = tf.random.experimental.stateless_split(seed_cur, num=2)
            next_seed = seeds[0]
            step_seed = seeds[1]
            theta_new, acc_inc = step_fn(theta_cur, k_in, step_seed)
            acc_sum = acc_sum + tf.cast(acc_inc, theta_in.dtype)
            return i + 1, theta_new, acc_sum, next_seed

        _, theta_out, acc_sum, seed_out = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(i0, theta_in, acc0, seed_in),
            parallel_iterations=1,
        )
        return theta_out, acc_sum, seed_out

    for round_id in range(max_rounds):
        theta_end, acc_sum, seed = _pilot(theta, k, seed)
        acc_rate = acc_sum / tf.cast(pilot_length_t, theta.dtype)

        k_before = float(k.numpy())
        acc_rate_py = float(acc_rate.numpy())

        if acc_rate < target_low_t:
            action = "shrink"
            k = k / factor_t
        elif acc_rate > target_high_t:
            action = "grow"
            k = k * factor_t
        else:
            action = "ok"

        k_after = float(k.numpy())

        print(
            f"[LuShrinkage:Tune:{name}] "
            f"round={round_id} | "
            f"k={k_before:.4f}->{k_after:.4f} | "
            f"acc={acc_rate_py:.3f} | "
            f"action={action}"
        )

        theta = theta_end

        if action == "ok":
            break

    return k, theta


def tune_shrinkage(
    posterior,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    initial_state,
    shrinkage_config,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    seed: tf.Tensor | int | None = None,
):
    local_state = initial_state

    if seed is None:
        seed = tf.random.uniform(
            shape=(2,),
            minval=0,
            maxval=2**31 - 1,
            dtype=tf.int32,
        )
    else:
        seed = tf.convert_to_tensor(seed, dtype=tf.int32)
        if seed.shape.rank == 0:
            seed = tf.stack([seed, tf.constant(0, dtype=tf.int32)])

    block_seeds = tf.random.experimental.stateless_split(seed, num=4)

    k_beta0 = (
        tf.constant(shrinkage_config.k_beta, dtype=posterior.dtype)
        if float(shrinkage_config.k_beta) > 0.0
        else tf.cast(_lu_k0(tf.constant(2.0, dtype=tf.float64)), posterior.dtype)
    )
    k_r0 = (
        tf.constant(shrinkage_config.k_r, dtype=posterior.dtype)
        if float(shrinkage_config.k_r) > 0.0
        else tf.cast(_lu_k0(tf.constant(1.0, dtype=tf.float64)), posterior.dtype)
    )
    k_E_bar0 = (
        tf.constant(shrinkage_config.k_E_bar, dtype=posterior.dtype)
        if float(shrinkage_config.k_E_bar) > 0.0
        else tf.cast(_lu_k0(tf.constant(1.0, dtype=tf.float64)), posterior.dtype)
    )
    k_njt0 = (
        tf.constant(shrinkage_config.k_njt, dtype=posterior.dtype)
        if float(shrinkage_config.k_njt) > 0.0
        else tf.cast(_lu_k0(tf.cast(tf.shape(pjt)[1], tf.float64)), posterior.dtype)
    )

    beta_vec0 = tf.stack([local_state.beta_p, local_state.beta_w], axis=0)

    def step_beta(
        theta_beta: tf.Tensor,
        k_beta: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        beta_p_new, beta_w_new, accepted = beta_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=theta_beta[0],
            beta_w=theta_beta[1],
            r=local_state.r,
            E_bar=local_state.E_bar,
            njt=local_state.njt,
            k_beta=k_beta,
            seed=step_seed,
        )
        return tf.stack([beta_p_new, beta_w_new], axis=0), accepted

    k_beta_tuned, beta_vec_end = _tune_block(
        theta0=beta_vec0,
        step_fn=step_beta,
        k0=k_beta0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name="beta",
        seed=block_seeds[0],
    )
    local_state = local_state._replace(
        beta_p=beta_vec_end[0],
        beta_w=beta_vec_end[1],
    )

    def step_r(
        theta_r: tf.Tensor,
        k_r: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        r_new, accepted = r_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=local_state.beta_p,
            beta_w=local_state.beta_w,
            r=theta_r,
            E_bar=local_state.E_bar,
            njt=local_state.njt,
            k_r=k_r,
            seed=step_seed,
        )
        return r_new, accepted

    k_r_tuned, r_end = _tune_block(
        theta0=local_state.r,
        step_fn=step_r,
        k0=k_r0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name="r",
        seed=block_seeds[1],
    )
    local_state = local_state._replace(r=r_end)

    def step_E_bar(
        theta_E_bar: tf.Tensor,
        k_E_bar: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        E_bar_new, accepted = E_bar_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=local_state.beta_p,
            beta_w=local_state.beta_w,
            r=local_state.r,
            E_bar=theta_E_bar,
            njt=local_state.njt,
            k_E_bar=k_E_bar,
            seed=step_seed,
        )
        return E_bar_new, accepted

    k_E_bar_tuned, E_bar_end = _tune_block(
        theta0=local_state.E_bar,
        step_fn=step_E_bar,
        k0=k_E_bar0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name="E_bar",
        seed=block_seeds[2],
    )
    local_state = local_state._replace(E_bar=E_bar_end)

    def step_njt(
        theta_njt: tf.Tensor,
        k_njt: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        njt_new, accepted = njt_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=local_state.beta_p,
            beta_w=local_state.beta_w,
            r=local_state.r,
            E_bar=local_state.E_bar,
            njt=theta_njt,
            gamma=local_state.gamma,
            k_njt=k_njt,
            seed=step_seed,
        )
        return njt_new, accepted

    k_njt_tuned, _ = _tune_block(
        theta0=local_state.njt,
        step_fn=step_njt,
        k0=k_njt0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name="njt",
        seed=block_seeds[3],
    )

    return replace(
        shrinkage_config,
        k_beta=float(k_beta_tuned.numpy()),
        k_r=float(k_r_tuned.numpy()),
        k_E_bar=float(k_E_bar_tuned.numpy()),
        k_njt=float(k_njt_tuned.numpy()),
    )
