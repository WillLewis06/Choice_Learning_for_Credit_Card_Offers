# lu_shrinkage_tuning.py
from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from market_shock_estimators.lu_shrinkage_kernels import rw_mh_step, tmh_step


def _lu_k0(d: tf.Tensor) -> tf.Tensor:
    """Lu default initialization: k0 = 2.38 / sqrt(d)."""
    d = tf.cast(d, tf.float64)
    return tf.constant(2.38, tf.float64) / tf.sqrt(
        tf.maximum(d, tf.constant(1.0, tf.float64))
    )


def _lbfgs_mode(
    theta0: tf.Tensor, logp_fn: Callable[[tf.Tensor], tf.Tensor], max_lbfgs_iters: int
) -> tf.Tensor:
    """Return argmax_theta logp_fn(theta) via L-BFGS starting from theta0."""
    theta0 = tf.convert_to_tensor(theta0, dtype=tf.float64)

    def val_and_grad(x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            val = -tf.cast(logp_fn(x), tf.float64)
        g = tape.gradient(val, x)
        return val, g

    res = tfp.optimizer.lbfgs_minimize(
        val_and_grad,
        initial_position=theta0,
        max_iterations=int(max_lbfgs_iters),
    )
    return tf.where(tf.cast(res.converged, tf.bool), res.position, theta0)


def _pilot_run(
    theta0: tf.Tensor,
    k: tf.Tensor,
    pilot_length_t: tf.Tensor,
    step_fn: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Run a pilot chain of length `pilot_length_t` in TF control flow.

    Returns:
      theta_end: final state
      acc_sum: float64 scalar acceptance count
    """
    theta0 = tf.convert_to_tensor(theta0, dtype=tf.float64)
    k = tf.convert_to_tensor(k, dtype=tf.float64)
    pilot_length_t = tf.convert_to_tensor(pilot_length_t, dtype=tf.int32)

    def cond(i, theta, acc_sum):
        return i < pilot_length_t

    def body(i, theta, acc_sum):
        theta_new, accepted = step_fn(theta, k)
        acc_sum = acc_sum + tf.cast(accepted, tf.float64)
        return i + 1, theta_new, acc_sum

    i0 = tf.constant(0, tf.int32)
    acc0 = tf.constant(0.0, tf.float64)
    _, theta_end, acc_sum = tf.while_loop(cond, body, loop_vars=(i0, theta0, acc0))
    return theta_end, acc_sum


def tune_k(
    theta0: tf.Tensor,
    step_fn: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
    k0: tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Iteratively tune a scalar proposal scale k to an acceptance band.

    Each round:
      - run a pilot chain of length `pilot_length` (in TF)
      - compute acceptance rate
      - adjust k multiplicatively until acc in [target_low, target_high]
        or max_rounds reached

    Returns (k_tuned, acc_rate_last, theta_end_last_round).
    """
    if pilot_length <= 0:
        raise ValueError("pilot_length must be positive.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")
    if not (0.0 < target_low < target_high < 1.0):
        raise ValueError("Require 0 < target_low < target_high < 1.")

    theta0 = tf.convert_to_tensor(theta0, dtype=tf.float64)
    k = tf.maximum(
        tf.convert_to_tensor(k0, dtype=tf.float64), tf.constant(1e-12, tf.float64)
    )

    pilot_length_t = tf.constant(int(pilot_length), tf.int32)
    target_low_t = tf.constant(float(target_low), tf.float64)
    target_high_t = tf.constant(float(target_high), tf.float64)
    factor_t = tf.constant(float(factor), tf.float64)

    last_acc = tf.constant(0.0, tf.float64)
    last_theta_end = tf.identity(theta0)

    step_name = getattr(step_fn, "__name__", "step")
    print(
        f"[LuShrinkage:Tune:{step_name}] start | "
        f"pilot_length={pilot_length} | target=[{target_low},{target_high}] | "
        f"max_rounds={max_rounds} | k0={float(k.numpy()):.4f}"
    )

    @tf.function
    def _pilot(theta_init: tf.Tensor, k_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _pilot_run(theta_init, k_in, pilot_length_t, step_fn)

    for r in range(int(max_rounds)):
        theta_end, acc_sum = _pilot(theta0, k)
        acc_rate = acc_sum / tf.cast(pilot_length, tf.float64)

        last_acc = acc_rate
        last_theta_end = theta_end

        decision = "keep"
        if acc_rate < target_low_t:
            decision = "shrink"
        elif acc_rate > target_high_t:
            decision = "grow"

        print(
            f"[LuShrinkage:Tune:{step_name}] "
            f"round={r} | k={float(k.numpy()):.4f} | "
            f"acc={float(acc_rate.numpy()):.3f} | action={decision}"
        )

        if acc_rate < target_low_t:
            k = tf.maximum(k / factor_t, tf.constant(1e-12, tf.float64))
            continue

        if acc_rate > target_high_t:
            k = tf.maximum(k * factor_t, tf.constant(1e-12, tf.float64))
            continue

        break

    print(
        f"[LuShrinkage:Tune:{step_name}] done | "
        f"k={float(k.numpy()):.4f} | acc={float(last_acc.numpy()):.3f}"
    )
    return k, last_acc, last_theta_end


def _stack_global_data(
    shrink,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Return stacked tensors needed for full-sample likelihood evaluation.

    Expected shapes:
      qjt, pjt, wjt, njt: (T,J)
      q0t, E_bar: (T,)
    """
    qjt = tf.convert_to_tensor(shrink.qjt, dtype=tf.float64)
    q0t = tf.convert_to_tensor(shrink.q0t, dtype=tf.float64)
    pjt = tf.convert_to_tensor(shrink.pjt, dtype=tf.float64)
    wjt = tf.convert_to_tensor(shrink.wjt, dtype=tf.float64)
    E_bar = tf.convert_to_tensor(shrink.E_bar, dtype=tf.float64)
    njt = tf.convert_to_tensor(shrink.njt, dtype=tf.float64)
    return qjt, q0t, pjt, wjt, E_bar, njt


def _make_step_r(
    *,
    posterior,
    rng: tf.random.Generator,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    def step_r(theta: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def logp_r(r_val: tf.Tensor) -> tf.Tensor:
            ll = posterior.full_loglik(
                qjt=qjt,
                q0t=q0t,
                pjt=pjt,
                wjt=wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r_val,
                E_bar=E_bar,
                njt=njt,
            )
            return tf.cast(ll, tf.float64) + tf.cast(
                posterior.logprior_r(r=r_val), tf.float64
            )

        theta_new, accepted, _ = rw_mh_step(theta0=theta, logp_fn=logp_r, k=k, rng=rng)
        return theta_new, accepted

    return step_r


def _make_step_beta(
    *,
    posterior,
    rng: tf.random.Generator,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    r_fixed: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    ridge_t: tf.Tensor,
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    def step_beta(theta: tf.Tensor, k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        def logp_beta(beta_vec: tf.Tensor) -> tf.Tensor:
            beta_p = beta_vec[0]
            beta_w = beta_vec[1]
            ll = posterior.full_loglik(
                qjt=qjt,
                q0t=q0t,
                pjt=pjt,
                wjt=wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r_fixed,
                E_bar=E_bar,
                njt=njt,
            )
            lp = posterior.logprior_beta(beta_p=beta_p, beta_w=beta_w)
            return tf.cast(ll, tf.float64) + tf.cast(lp, tf.float64)

        theta_new, accepted = tmh_step(
            theta0=theta, logp_fn=logp_beta, ridge=ridge_t, rng=rng, k=k
        )
        return theta_new, accepted

    return step_beta


@tf.function
def _pilot_E_bar_all_markets(
    *,
    E_bar_vec: tf.Tensor,  # (T,)
    k_run: tf.Tensor,  # scalar
    pilot_length_t: tf.Tensor,  # int32 scalar
    qjt: tf.Tensor,  # (T,J)
    q0t: tf.Tensor,  # (T,)
    pjt: tf.Tensor,  # (T,J)
    wjt: tf.Tensor,  # (T,J)
    beta_p: tf.Tensor,  # scalar
    beta_w: tf.Tensor,  # scalar
    r: tf.Tensor,  # scalar
    njt: tf.Tensor,  # (T,J)
    gamma: tf.Tensor,  # (T,J)
    phi: tf.Tensor,  # (T,)
    posterior,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Run RW-MH pilots for all markets' E_bar in TF control flow.

    Returns:
      E_bar_new: (T,)
      acc_sum: float64 scalar
    """
    E_bar_vec = tf.convert_to_tensor(E_bar_vec, tf.float64)
    T_t = tf.shape(E_bar_vec)[0]
    k_run = tf.convert_to_tensor(k_run, tf.float64)
    pilot_length_t = tf.convert_to_tensor(pilot_length_t, tf.int32)

    ta = tf.TensorArray(tf.float64, size=T_t)
    ta = ta.unstack(E_bar_vec)

    def cond_t(t, ta_in, acc_sum):
        return t < T_t

    def body_t(t, ta_in, acc_sum):
        E_bar_t = ta_in.read(t)

        qjt_t = qjt[t]
        q0t_t = q0t[t]
        pjt_t = pjt[t]
        wjt_t = wjt[t]
        njt_t = njt[t]
        gamma_t = gamma[t]
        phi_t = phi[t]

        def logp_E_bar_t(E_bar_t_val: tf.Tensor) -> tf.Tensor:
            return posterior.market_logpost(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar_t_val,
                njt_t=njt_t,
                gamma_t=gamma_t,
                phi_t=phi_t,
            )

        def cond_s(s, theta, acc_sum_s):
            return s < pilot_length_t

        def body_s(s, theta, acc_sum_s):
            theta_new, accepted, _ = rw_mh_step(
                theta0=theta, logp_fn=logp_E_bar_t, k=k_run, rng=rng
            )
            acc_sum_s = acc_sum_s + tf.cast(accepted, tf.float64)
            return s + 1, theta_new, acc_sum_s

        s0 = tf.constant(0, tf.int32)
        _, E_bar_t_new, acc_sum = tf.while_loop(
            cond_s, body_s, loop_vars=(s0, E_bar_t, acc_sum)
        )

        ta_out = ta_in.write(t, E_bar_t_new)
        return t + 1, ta_out, acc_sum

    t0 = tf.constant(0, tf.int32)
    acc0 = tf.constant(0.0, tf.float64)
    _, ta_out, acc_sum = tf.while_loop(cond_t, body_t, loop_vars=(t0, ta, acc0))
    return ta_out.stack(), acc_sum


@tf.function
def _pilot_njt_all_markets(
    *,
    njt_mat: tf.Tensor,  # (T,J)
    k_run: tf.Tensor,  # scalar
    pilot_length_t: tf.Tensor,  # int32 scalar
    ridge_t: tf.Tensor,  # scalar
    qjt: tf.Tensor,  # (T,J)
    q0t: tf.Tensor,  # (T,)
    pjt: tf.Tensor,  # (T,J)
    wjt: tf.Tensor,  # (T,J)
    beta_p: tf.Tensor,  # scalar
    beta_w: tf.Tensor,  # scalar
    r: tf.Tensor,  # scalar
    E_bar: tf.Tensor,  # (T,)
    gamma: tf.Tensor,  # (T,J)
    phi: tf.Tensor,  # (T,)
    posterior,
    rng: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Run TMH pilots for all markets' njt in TF control flow.

    Returns:
      njt_new: (T,J)
      acc_sum: float64 scalar
    """
    njt_mat = tf.convert_to_tensor(njt_mat, tf.float64)
    T_t = tf.shape(njt_mat)[0]
    k_run = tf.convert_to_tensor(k_run, tf.float64)
    pilot_length_t = tf.convert_to_tensor(pilot_length_t, tf.int32)
    ridge_t = tf.convert_to_tensor(ridge_t, tf.float64)

    ta = tf.TensorArray(tf.float64, size=T_t)
    ta = ta.unstack(njt_mat)

    def cond_t(t, ta_in, acc_sum):
        return t < T_t

    def body_t(t, ta_in, acc_sum):
        theta = ta_in.read(t)

        qjt_t = qjt[t]
        q0t_t = q0t[t]
        pjt_t = pjt[t]
        wjt_t = wjt[t]
        gamma_t = gamma[t]
        phi_t = phi[t]
        E_bar_t = E_bar[t]

        def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
            return posterior.market_logpost(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar_t,
                njt_t=njt_t_val,
                gamma_t=gamma_t,
                phi_t=phi_t,
            )

        def cond_s(s, theta_s, acc_sum_s):
            return s < pilot_length_t

        def body_s(s, theta_s, acc_sum_s):
            theta_new, accepted = tmh_step(
                theta0=theta_s,
                logp_fn=logp_njt_t,
                ridge=ridge_t,
                rng=rng,
                k=k_run,
            )
            acc_sum_s = acc_sum_s + tf.cast(accepted, tf.float64)
            return s + 1, theta_new, acc_sum_s

        s0 = tf.constant(0, tf.int32)
        _, theta_new, acc_sum = tf.while_loop(
            cond_s, body_s, loop_vars=(s0, theta, acc_sum)
        )

        ta_out = ta_in.write(t, theta_new)
        return t + 1, ta_out, acc_sum

    t0 = tf.constant(0, tf.int32)
    acc0 = tf.constant(0.0, tf.float64)
    _, ta_out, acc_sum = tf.while_loop(cond_t, body_t, loop_vars=(t0, ta, acc0))
    return ta_out.stack(), acc_sum


def _tune_r(
    *,
    shrink,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor_rw: float,
    k_r0: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    theta_r0 = (
        shrink.r.read_value()
        if hasattr(shrink.r, "read_value")
        else tf.convert_to_tensor(shrink.r, tf.float64)
    )

    step_r = _make_step_r(
        posterior=shrink.posterior,
        rng=shrink.rng,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        beta_p=tf.convert_to_tensor(shrink.beta_p, tf.float64),
        beta_w=tf.convert_to_tensor(shrink.beta_w, tf.float64),
        E_bar=E_bar,
        njt=njt,
    )

    k_r_tuned, acc_r, _ = tune_k(
        theta0=tf.cast(theta_r0, tf.float64),
        step_fn=step_r,
        k0=tf.cast(k_r0, tf.float64),
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_rw,
    )
    print(
        f"[LuShrinkage:Tune] k_r final: "
        f"{float(k_r0.numpy()):.4f} -> {float(k_r_tuned.numpy()):.4f} | "
        f"acc={float(acc_r.numpy()):.3f}"
    )
    return k_r_tuned, acc_r


def _tune_beta(
    *,
    shrink,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor_tmh: float,
    k_beta0: tf.Tensor,
    ridge_t: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    beta0 = tf.stack(
        [tf.cast(shrink.beta_p, tf.float64), tf.cast(shrink.beta_w, tf.float64)], axis=0
    )

    step_beta = _make_step_beta(
        posterior=shrink.posterior,
        rng=shrink.rng,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        r_fixed=tf.cast(shrink.r, tf.float64),
        E_bar=E_bar,
        njt=njt,
        ridge_t=ridge_t,
    )

    k_beta_tuned, acc_beta, _ = tune_k(
        theta0=beta0,
        step_fn=step_beta,
        k0=tf.cast(k_beta0, tf.float64),
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_tmh,
    )
    print(
        f"[LuShrinkage:Tune] k_beta final: "
        f"{float(k_beta0.numpy()):.4f} -> {float(k_beta_tuned.numpy()):.4f} | "
        f"acc={float(acc_beta.numpy()):.3f}"
    )
    return k_beta_tuned, acc_beta


def _tune_E_bar(
    *,
    shrink,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor_rw: float,
    k_E_bar0: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    k_E_bar = tf.cast(k_E_bar0, tf.float64)
    eps_k = tf.constant(1e-12, tf.float64)

    pilot_length_t = tf.constant(int(pilot_length), tf.int32)
    target_low_t = tf.constant(float(target_low), tf.float64)
    target_high_t = tf.constant(float(target_high), tf.float64)
    factor_rw_t = tf.constant(float(factor_rw), tf.float64)

    qjt = tf.convert_to_tensor(shrink.qjt, tf.float64)
    q0t = tf.convert_to_tensor(shrink.q0t, tf.float64)
    pjt = tf.convert_to_tensor(shrink.pjt, tf.float64)
    wjt = tf.convert_to_tensor(shrink.wjt, tf.float64)
    njt = tf.convert_to_tensor(shrink.njt, tf.float64)
    gamma = tf.convert_to_tensor(shrink.gamma, tf.float64)
    phi = tf.convert_to_tensor(shrink.phi, tf.float64)

    beta_p = tf.convert_to_tensor(shrink.beta_p, tf.float64)
    beta_w = tf.convert_to_tensor(shrink.beta_w, tf.float64)
    r = tf.convert_to_tensor(shrink.r, tf.float64)

    E_bar_vec0 = tf.convert_to_tensor(shrink.E_bar, tf.float64)
    total_steps = tf.cast(tf.size(E_bar_vec0) * pilot_length, tf.float64)

    print(
        "[LuShrinkage:Tune:E_bar] start | "
        f"target=[{target_low},{target_high}] | max_rounds={max_rounds} | "
        f"steps_per_round={int(total_steps.numpy())}"
    )

    E_bar_vec = tf.identity(E_bar_vec0)
    acc_E_bar = tf.constant(0.0, tf.float64)

    for r_i in range(int(max_rounds)):
        k_run = tf.maximum(k_E_bar, eps_k)
        E_bar_vec, acc_sum = _pilot_E_bar_all_markets(
            E_bar_vec=E_bar_vec,
            k_run=k_run,
            pilot_length_t=pilot_length_t,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            njt=njt,
            gamma=gamma,
            phi=phi,
            posterior=shrink.posterior,
            rng=shrink.rng,
        )

        acc_E_bar = acc_sum / total_steps

        decision = "keep"
        if acc_E_bar < target_low_t:
            decision = "shrink"
        elif acc_E_bar > target_high_t:
            decision = "grow"

        print(
            f"[LuShrinkage:Tune:E_bar] "
            f"round={r_i} | k={float(k_E_bar.numpy()):.4f} | "
            f"acc={float(acc_E_bar.numpy()):.3f} | action={decision}"
        )

        if acc_E_bar < target_low_t:
            k_E_bar = tf.maximum(k_E_bar / factor_rw_t, eps_k)
            continue

        if acc_E_bar > target_high_t:
            k_E_bar = tf.maximum(k_E_bar * factor_rw_t, eps_k)
            continue

        break

    print(
        f"[LuShrinkage:Tune] k_E_bar final: "
        f"{float(k_E_bar0.numpy()):.4f} -> {float(k_E_bar.numpy()):.4f} | "
        f"acc={float(acc_E_bar.numpy()):.3f}"
    )
    return k_E_bar, acc_E_bar


def _tune_njt(
    *,
    shrink,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor_tmh: float,
    k_njt0: tf.Tensor,
    ridge_t: tf.Tensor,
    max_lbfgs_iters: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    k_njt = tf.cast(k_njt0, tf.float64)
    eps_k = tf.constant(1e-12, tf.float64)

    pilot_length_t = tf.constant(int(pilot_length), tf.int32)
    target_low_t = tf.constant(float(target_low), tf.float64)
    target_high_t = tf.constant(float(target_high), tf.float64)
    factor_tmh_t = tf.constant(float(factor_tmh), tf.float64)

    qjt = tf.convert_to_tensor(shrink.qjt, tf.float64)
    q0t = tf.convert_to_tensor(shrink.q0t, tf.float64)
    pjt = tf.convert_to_tensor(shrink.pjt, tf.float64)
    wjt = tf.convert_to_tensor(shrink.wjt, tf.float64)
    gamma = tf.convert_to_tensor(shrink.gamma, tf.float64)
    phi = tf.convert_to_tensor(shrink.phi, tf.float64)
    E_bar = tf.convert_to_tensor(shrink.E_bar, tf.float64)

    beta_p = tf.convert_to_tensor(shrink.beta_p, tf.float64)
    beta_w = tf.convert_to_tensor(shrink.beta_w, tf.float64)
    r = tf.convert_to_tensor(shrink.r, tf.float64)

    # Precompute per-market mode (mu_t) once in Python via L-BFGS.
    mu_list = []

    def _make_logp_njt_full(t_: int):
        qjt_t = qjt[t_]
        q0t_t = q0t[t_]
        pjt_t = pjt[t_]
        wjt_t = wjt[t_]
        gamma_t = gamma[t_]
        phi_t = phi[t_]
        E_bar_t = E_bar[t_]

        def logp_njt_full(njt_t_val: tf.Tensor) -> tf.Tensor:
            return shrink.posterior.market_logpost(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar_t,
                njt_t=njt_t_val,
                gamma_t=gamma_t,
                phi_t=phi_t,
            )

        return logp_njt_full

    T_int = int(shrink.T)
    for t in range(T_int):
        logp_t = _make_logp_njt_full(t)
        theta0_t = tf.cast(tf.convert_to_tensor(shrink.njt[t]), tf.float64)
        mu_t = _lbfgs_mode(theta0_t, logp_t, max_lbfgs_iters=max_lbfgs_iters)
        mu_list.append(mu_t)

    njt_mat = tf.stack(mu_list, axis=0)  # (T,J)
    total_steps = tf.cast(tf.shape(njt_mat)[0] * pilot_length, tf.float64)

    print(
        "[LuShrinkage:Tune:njt] start | "
        f"target=[{target_low},{target_high}] | max_rounds={max_rounds} | "
        f"steps_per_round={int(total_steps.numpy())}"
    )

    acc_njt = tf.constant(0.0, tf.float64)

    for r_i in range(int(max_rounds)):
        k_run = tf.maximum(k_njt, eps_k)
        njt_mat, acc_sum = _pilot_njt_all_markets(
            njt_mat=njt_mat,
            k_run=k_run,
            pilot_length_t=pilot_length_t,
            ridge_t=ridge_t,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar=E_bar,
            gamma=gamma,
            phi=phi,
            posterior=shrink.posterior,
            rng=shrink.rng,
        )

        acc_njt = acc_sum / total_steps

        decision = "keep"
        if acc_njt < target_low_t:
            decision = "shrink"
        elif acc_njt > target_high_t:
            decision = "grow"

        print(
            f"[LuShrinkage:Tune:njt] "
            f"round={r_i} | k={float(k_njt.numpy()):.4f} | "
            f"acc={float(acc_njt.numpy()):.3f} | action={decision}"
        )

        if acc_njt < target_low_t:
            k_njt = tf.maximum(k_njt / factor_tmh_t, eps_k)
            continue

        if acc_njt > target_high_t:
            k_njt = tf.maximum(k_njt * factor_tmh_t, eps_k)
            continue

        break

    print(
        f"[LuShrinkage:Tune] k_njt final: "
        f"{float(k_njt0.numpy()):.4f} -> {float(k_njt.numpy()):.4f} | "
        f"acc={float(acc_njt.numpy()):.3f}"
    )
    return k_njt, acc_njt


def tune_shrinkage(shrink):
    """
    Tune proposal scales for the Lu shrinkage sampler (tune once, then freeze).

    Returns:
      (k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned)
    """
    required = [
        "pilot_length",
        "ridge",
        "max_lbfgs_iters",
        "target_low",
        "target_high",
        "max_rounds",
        "factor_rw",
        "factor_tmh",
        "T",
        "qjt",
        "q0t",
        "pjt",
        "wjt",
        "beta_p",
        "beta_w",
        "r",
        "E_bar",
        "njt",
        "gamma",
        "phi",
        "posterior",
        "rng",
    ]
    missing = [name for name in required if not hasattr(shrink, name)]
    if missing:
        raise AttributeError(
            "tune_shrinkage requires shrink to define: " + ", ".join(missing)
        )

    pilot_length = int(shrink.pilot_length)
    ridge = float(shrink.ridge)
    max_lbfgs_iters = int(shrink.max_lbfgs_iters)

    target_low = float(shrink.target_low)
    target_high = float(shrink.target_high)
    max_rounds = int(shrink.max_rounds)
    factor_rw = float(shrink.factor_rw)
    factor_tmh = float(shrink.factor_tmh)

    ridge_t = tf.convert_to_tensor(ridge, dtype=tf.float64)

    k_r0 = _lu_k0(tf.constant(1.0, tf.float64))
    k_beta0 = _lu_k0(tf.constant(2.0, tf.float64))
    k_E_bar0 = _lu_k0(tf.constant(1.0, tf.float64))

    njt0_any = tf.convert_to_tensor(shrink.njt[0], dtype=tf.float64)
    J_static = njt0_any.shape[0]
    J_int = (
        int(J_static) if J_static is not None else int(tf.shape(njt0_any)[0].numpy())
    )
    k_njt0 = _lu_k0(tf.cast(tf.constant(J_int), tf.float64))

    print(
        "[LuShrinkage:Tune] begin | pilot_length=",
        pilot_length,
        "| T=",
        int(getattr(shrink, "T")),
        "| J=",
        J_int,
        "| ridge=",
        ridge,
        "| max_lbfgs_iters=",
        max_lbfgs_iters,
    )
    print(
        f"[LuShrinkage:Tune] k0 defaults | "
        f"k_r0={float(k_r0.numpy()):.4f} | "
        f"k_E_bar0={float(k_E_bar0.numpy()):.4f} | "
        f"k_beta0={float(k_beta0.numpy()):.4f} | "
        f"k_njt0={float(k_njt0.numpy()):.4f}"
    )

    qjt, q0t, pjt, wjt, E_bar, njt = _stack_global_data(shrink)

    k_r_tuned, _ = _tune_r(
        shrink=shrink,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor_rw=factor_rw,
        k_r0=k_r0,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        E_bar=E_bar,
        njt=njt,
    )

    k_beta_tuned, _ = _tune_beta(
        shrink=shrink,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor_tmh=factor_tmh,
        k_beta0=k_beta0,
        ridge_t=ridge_t,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        E_bar=E_bar,
        njt=njt,
    )

    k_E_bar_tuned, _ = _tune_E_bar(
        shrink=shrink,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor_rw=factor_rw,
        k_E_bar0=k_E_bar0,
    )

    k_njt_tuned, _ = _tune_njt(
        shrink=shrink,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor_tmh=factor_tmh,
        k_njt0=k_njt0,
        ridge_t=ridge_t,
        max_lbfgs_iters=max_lbfgs_iters,
    )

    print("[LuShrinkage:Tune] done")
    return k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned
