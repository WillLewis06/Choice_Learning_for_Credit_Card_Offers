"""Proposal-scale tuning for the choice-learn shrinkage sampler."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import tensorflow as tf

from lu.choice_learn.cl_updates import (
    E_bar_one_step,
    alpha_one_step,
    njt_one_step,
)


def _cl_k0(d: tf.Tensor) -> tf.Tensor:
    """Return the default random-walk scale for a block of dimension d."""

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
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tune one proposal scale with repeated pilot runs for a single block.

    Shrink, grow, or keep the proposal scale according to whether the pilot
    acceptance rate falls below, above, or inside the target band.
    """

    theta = theta0
    k = k0

    pilot_length_t = tf.constant(pilot_length, dtype=tf.int32)
    factor_t = tf.constant(factor, dtype=tf.float64)
    target_low_t = tf.constant(target_low, dtype=tf.float64)
    target_high_t = tf.constant(target_high, dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _pilot(
        theta_in: tf.Tensor,
        k_in: tf.Tensor,
        seed_in: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run a fixed-length pilot chain for one block."""

        i0 = tf.constant(0, dtype=tf.int32)
        acc0 = tf.constant(0.0, dtype=tf.float64)

        def cond(i, theta_cur, acc_sum, seed_cur):
            """Continue until the pilot run reaches its target length."""

            return i < pilot_length_t

        def body(i, theta_cur, acc_sum, seed_cur):
            """Apply one block update and accumulate acceptance."""

            seeds = tf.random.experimental.stateless_split(seed_cur, num=2)
            next_seed = seeds[0]
            step_seed = seeds[1]

            theta_new, acc_inc = step_fn(theta_cur, k_in, step_seed)
            acc_sum = acc_sum + tf.cast(acc_inc, tf.float64)
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
        acc_rate = acc_sum / tf.cast(pilot_length_t, tf.float64)

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
            f"[ChoiceLearnShrinkage:Tune:{name}] "
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
    delta_cl: tf.Tensor,
    initial_state,
    shrinkage_config,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    seed: tf.Tensor,
):
    """Tune proposal scales for all continuous choice-learn shrinkage blocks.

    Tune alpha, E_bar, and njt sequentially using short pilot runs, then return
    the shrinkage config with updated proposal scales.
    """

    local_state = initial_state
    block_seeds = tf.random.experimental.stateless_split(seed, num=3)

    k_alpha0 = (
        tf.constant(shrinkage_config.k_alpha, dtype=tf.float64)
        if float(shrinkage_config.k_alpha) > 0.0
        else _cl_k0(tf.constant(1.0, dtype=tf.float64))
    )
    k_E_bar0 = (
        tf.constant(shrinkage_config.k_E_bar, dtype=tf.float64)
        if float(shrinkage_config.k_E_bar) > 0.0
        else _cl_k0(tf.constant(1.0, dtype=tf.float64))
    )
    k_njt0 = (
        tf.constant(shrinkage_config.k_njt, dtype=tf.float64)
        if float(shrinkage_config.k_njt) > 0.0
        else _cl_k0(tf.cast(tf.shape(delta_cl)[1], tf.float64))
    )

    def step_alpha(
        theta_alpha: tf.Tensor,
        k_alpha: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one alpha update during tuning."""

        alpha_new, accepted = alpha_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=theta_alpha,
            E_bar=local_state.E_bar,
            njt=local_state.njt,
            k_alpha=k_alpha,
            seed=step_seed,
        )
        return alpha_new, accepted

    k_alpha_tuned, alpha_end = _tune_block(
        theta0=local_state.alpha,
        step_fn=step_alpha,
        k0=k_alpha0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor,
        name="alpha",
        seed=block_seeds[0],
    )

    local_state = local_state._replace(alpha=alpha_end)

    def step_E_bar(
        theta_E_bar: tf.Tensor,
        k_E_bar: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one full E_bar sweep during tuning."""

        E_bar_new, accepted = E_bar_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=local_state.alpha,
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
        seed=block_seeds[1],
    )

    local_state = local_state._replace(E_bar=E_bar_end)

    def step_njt(
        theta_njt: tf.Tensor,
        k_njt: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one full njt sweep during tuning."""

        njt_new, accepted = njt_one_step(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=local_state.alpha,
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
        seed=block_seeds[2],
    )

    return replace(
        shrinkage_config,
        k_alpha=float(k_alpha_tuned.numpy()),
        k_E_bar=float(k_E_bar_tuned.numpy()),
        k_njt=float(k_njt_tuned.numpy()),
    )
