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

    Each pilot run starts from the endpoint of the previous pilot run rather than
    resetting to the original state. The function returns the tuned proposal
    scale together with the final block state reached by the pilot chain.
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
        """Run a fixed-length pilot chain for one block at the current scale."""

        i0 = tf.constant(0, dtype=tf.int32)
        acc0 = tf.constant(0.0, dtype=tf.float64)

        def cond(
            step_index: tf.Tensor,
            theta_cur: tf.Tensor,
            acc_sum: tf.Tensor,
            seed_cur: tf.Tensor,
        ) -> tf.Tensor:
            """Continue until the pilot run reaches its configured length."""

            del theta_cur, acc_sum, seed_cur
            return step_index < pilot_length_t

        def body(
            step_index: tf.Tensor,
            theta_cur: tf.Tensor,
            acc_sum: tf.Tensor,
            seed_cur: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            """Apply one update and accumulate the pilot acceptance count."""

            # stateless_split returns a single tensor of shape (2, 2), so index
            # the split seeds explicitly rather than tuple-unpacking a tensor.
            split_seeds = tf.random.experimental.stateless_split(seed_cur, num=2)
            next_seed = split_seeds[0]
            step_seed = split_seeds[1]

            theta_new, acc_inc = step_fn(theta_cur, k_in, step_seed)
            acc_sum = acc_sum + tf.cast(acc_inc, tf.float64)
            return step_index + 1, theta_new, acc_sum, next_seed

        _, theta_out, acc_sum, seed_out = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(i0, theta_in, acc0, seed_in),
            parallel_iterations=1,
        )
        return theta_out, acc_sum, seed_out

    for round_index in range(max_rounds):
        theta_end, acc_sum, seed = _pilot(theta, k, seed)
        acc_rate = acc_sum / tf.cast(pilot_length_t, tf.float64)

        # Convert to Python only for reporting and outer-loop control.
        k_before = float(k.numpy())
        acc_rate_value = float(acc_rate.numpy())

        if acc_rate_value < target_low:
            action = "shrink"
            k = k / factor_t
        elif acc_rate_value > target_high:
            action = "grow"
            k = k * factor_t
        else:
            action = "ok"

        k_after = float(k.numpy())

        print(
            f"[ChoiceLearnShrinkage:Tune:{name}] "
            f"round={round_index} | "
            f"k={k_before:.4f}->{k_after:.4f} | "
            f"acc={acc_rate_value:.3f} | "
            f"action={action}"
        )

        # Carry the pilot endpoint forward so later rounds continue locally.
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
    seed: tf.Tensor,
):
    """Tune proposal scales for the continuous shrinkage blocks.

    This tunes ``alpha``, then ``E_bar``, then ``njt`` sequentially using the
    tuning controls stored on ``shrinkage_config``. The function returns an
    updated config with tuned proposal scales only; it does not return a tuned
    chain state.
    """

    local_state = initial_state

    # Split the top-level tuning seed into one seed per tuned block.
    block_seeds = tf.random.experimental.stateless_split(seed, num=3)

    k_alpha0 = tf.constant(shrinkage_config.k_alpha, dtype=tf.float64)
    k_E_bar0 = tf.constant(shrinkage_config.k_E_bar, dtype=tf.float64)
    k_njt0 = tf.constant(shrinkage_config.k_njt, dtype=tf.float64)

    def step_alpha(
        theta_alpha: tf.Tensor,
        k_alpha: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one alpha update during pilot tuning."""

        return alpha_one_step(
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

    k_alpha_tuned, alpha_end = _tune_block(
        theta0=local_state.alpha,
        step_fn=step_alpha,
        k0=k_alpha0,
        pilot_length=shrinkage_config.pilot_length,
        target_low=shrinkage_config.target_low,
        target_high=shrinkage_config.target_high,
        max_rounds=shrinkage_config.max_rounds,
        factor=shrinkage_config.factor,
        name="alpha",
        seed=block_seeds[0],
    )

    # Later block tuning conditions on the tuned alpha endpoint.
    local_state = local_state._replace(alpha=alpha_end)

    def step_E_bar(
        theta_E_bar: tf.Tensor,
        k_E_bar: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one full E_bar sweep during pilot tuning."""

        return E_bar_one_step(
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

    k_E_bar_tuned, E_bar_end = _tune_block(
        theta0=local_state.E_bar,
        step_fn=step_E_bar,
        k0=k_E_bar0,
        pilot_length=shrinkage_config.pilot_length,
        target_low=shrinkage_config.target_low,
        target_high=shrinkage_config.target_high,
        max_rounds=shrinkage_config.max_rounds,
        factor=shrinkage_config.factor,
        name="E_bar",
        seed=block_seeds[1],
    )

    # Later block tuning conditions on the tuned E_bar endpoint.
    local_state = local_state._replace(E_bar=E_bar_end)

    def step_njt(
        theta_njt: tf.Tensor,
        k_njt: tf.Tensor,
        step_seed: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply one full njt sweep during pilot tuning."""

        return njt_one_step(
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

    k_njt_tuned, _ = _tune_block(
        theta0=local_state.njt,
        step_fn=step_njt,
        k0=k_njt0,
        pilot_length=shrinkage_config.pilot_length,
        target_low=shrinkage_config.target_low,
        target_high=shrinkage_config.target_high,
        max_rounds=shrinkage_config.max_rounds,
        factor=shrinkage_config.factor,
        name="njt",
        seed=block_seeds[2],
    )

    # Return a config with updated proposal scales; the tuned pilot state is not
    # propagated out of this function.
    return replace(
        shrinkage_config,
        k_alpha=float(k_alpha_tuned.numpy()),
        k_E_bar=float(k_E_bar_tuned.numpy()),
        k_njt=float(k_njt_tuned.numpy()),
    )
