"""
Proposal-scale tuning for the choice-learn + Lu shrinkage sampler.

Tuning runs short pilot chains and multiplicatively adjusts proposal scales until
the average acceptance rate lies in [target_low, target_high], or max_rounds is
reached.

Conventions:
  - step_fn(theta, k) -> (theta_new, acc_inc)
  - acc_inc is a tf.float64 scalar in [0, 1]
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from lu.choice_learn.cl_updates import (
    update_alpha,
    update_E_bar,
    update_njt,
)


def tune_k(
    theta0: tf.Tensor,
    step_fn: Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    k0: tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    name: str,
) -> tf.Tensor:
    """Tune a scalar proposal scale k for a single parameter block."""
    theta = theta0
    k = k0
    factor_t = tf.constant(float(factor), tf.float64)

    for round_id in range(int(max_rounds)):
        theta_cur = theta
        acc_sum = tf.constant(0.0, tf.float64)

        for _ in range(int(pilot_length)):
            theta_cur, acc_inc = step_fn(theta_cur, k)
            acc_sum = acc_sum + acc_inc

        acc_rate = float(acc_sum.numpy()) / float(pilot_length)
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
            f"round={round_id} | k={k_before:.6g}->{k_after:.6g} | acc={acc_rate:.3f} | action={action}"
        )

        theta = theta_cur

        if action == "ok":
            break

    return k


def tune_shrinkage(shrink):
    """Tune (k_alpha, k_E_bar, k_njt) using pilot runs without mutating chain RNG.

    Required fields on `shrink`:
      - pilot_length, ridge, target_low, target_high, max_rounds, factor_rw, factor_tmh
      - k_alpha0, k_E_bar0, k_njt0
      - tune_seed
      - T
      - qjt, q0t, delta_cl
      - alpha, E_bar, njt, gamma, phi  (tf.Variable)
      - posterior
    """
    pilot_length = int(shrink.pilot_length)
    ridge = shrink.ridge
    target_low = float(shrink.target_low)
    target_high = float(shrink.target_high)
    max_rounds = int(shrink.max_rounds)
    factor_rw = float(shrink.factor_rw)
    factor_tmh = float(shrink.factor_tmh)

    k_alpha0 = shrink.k_alpha0
    k_E_bar0 = shrink.k_E_bar0
    k_njt0 = shrink.k_njt0
    tune_seed = int(shrink.tune_seed)

    qjt = shrink.qjt
    q0t = shrink.q0t
    delta_cl = shrink.delta_cl
    posterior = shrink.posterior

    # Snapshot the current sampler state; tuning holds non-target blocks fixed.
    alpha0 = shrink.alpha.read_value()
    E_bar0 = shrink.E_bar.read_value()
    njt0 = shrink.njt.read_value()
    gamma0 = shrink.gamma.read_value()
    phi0 = shrink.phi.read_value()

    one = tf.constant(1.0, tf.float64)
    zero = tf.constant(0.0, tf.float64)
    T_float = tf.constant(float(shrink.T), tf.float64)

    # Dedicated RNGs for tuning (do not use the sampler RNG).
    rng_alpha = tf.random.Generator.from_seed(tune_seed + 1)
    rng_E_bar = tf.random.Generator.from_seed(tune_seed + 2)
    rng_njt = tf.random.Generator.from_seed(tune_seed + 3)

    def step_alpha(
        theta_alpha: tf.Tensor, k_alpha: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        alpha_new, accepted = update_alpha(
            posterior=posterior,
            rng=rng_alpha,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=theta_alpha,
            E_bar=E_bar0,
            njt=njt0,
            k_alpha=k_alpha,
        )
        acc_inc = tf.where(accepted, one, zero)
        return alpha_new, acc_inc

    k_alpha = tune_k(
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

    def step_E_bar(
        theta_E_bar: tf.Tensor, k_E_bar: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        E_bar_new, accepted_vec = update_E_bar(
            posterior=posterior,
            rng=rng_E_bar,
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
        acc_inc = tf.reduce_mean(tf.where(accepted_vec, one, zero))
        return E_bar_new, acc_inc

    k_E_bar = tune_k(
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

    def step_njt(theta_njt: tf.Tensor, k_njt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        njt_new, acc_sum = update_njt(
            posterior=posterior,
            rng=rng_njt,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha0,
            E_bar=E_bar0,
            njt=theta_njt,
            gamma=gamma0,
            phi=phi0,
            k_njt=k_njt,
            ridge=ridge,
        )
        acc_inc = acc_sum / T_float
        return njt_new, acc_inc

    k_njt = tune_k(
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

    return k_alpha, k_E_bar, k_njt
