# lu_shrinkage_tuning.py
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from market_shock_estimators.lu_shrinkage_kernels import rw_mh_step, tmh_step


def _lu_k0(d: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    """Lu default initialization: k0 = 2.38 / sqrt(d)."""
    d = tf.cast(d, dtype)
    return tf.cast(2.38, dtype) / tf.sqrt(tf.maximum(d, tf.cast(1.0, dtype)))


def _lbfgs_mode(theta0: tf.Tensor, logp_fn, max_lbfgs_iters: int) -> tf.Tensor:
    """Return argmax_theta logp_fn(theta) via L-BFGS starting from theta0."""
    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    def val_and_grad(x):
        x = tf.convert_to_tensor(x, dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(x)
            val = -logp_fn(x)
        g = tape.gradient(val, x)
        return val, g

    res = tfp.optimizer.lbfgs_minimize(
        val_and_grad,
        initial_position=theta0,
        max_iterations=max_lbfgs_iters,
    )
    mu = tf.where(tf.cast(res.converged, tf.bool), res.position, theta0)
    return mu


def tune_k(
    theta0: tf.Tensor,
    step_fn,
    k0: float | tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Iteratively tune a scalar proposal scale k to an acceptance band.

    Each round:
      - run a pilot chain of length `pilot_length`
      - compute acceptance rate
      - adjust k multiplicatively until acc in [target_low, target_high]
        or max_rounds reached

    Assumes `step_fn(theta, k)` returns (theta_new, accepted).

    Returns (k_tuned, acc_rate_last, theta_end_last_round).
    """
    if pilot_length <= 0:
        raise ValueError("pilot_length must be positive.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")
    if not (0.0 < target_low < target_high < 1.0):
        raise ValueError("Require 0 < target_low < target_high < 1.")

    theta0 = tf.convert_to_tensor(theta0)
    dtype = theta0.dtype

    k = tf.cast(tf.convert_to_tensor(k0), dtype)
    eps_k = tf.cast(1e-12, dtype)
    k = tf.maximum(k, eps_k)

    target_low_t = tf.cast(target_low, dtype)
    target_high_t = tf.cast(target_high, dtype)
    factor_t = tf.cast(factor, dtype)
    last_acc = tf.cast(0.0, dtype)
    last_theta_end = tf.identity(theta0)

    step_name = getattr(step_fn, "__name__", "step")

    print(
        f"[LuShrinkage:Tune:{step_name}] start | "
        f"pilot_length={pilot_length} | "
        f"target=[{target_low},{target_high}] | "
        f"max_rounds={max_rounds} | "
        f"k0={float(k.numpy())}"
    )

    for _round in range(int(max_rounds)):
        print(
            f"[LuShrinkage:Tune:{step_name}] round={_round} | " f"k={float(k.numpy())}"
        )

        theta = tf.identity(theta0)  # restart each round for comparability
        acc_sum = tf.cast(0.0, dtype)

        for _ in range(int(pilot_length)):
            theta, accepted = step_fn(theta, k)
            acc_sum += tf.cast(accepted, dtype)

        acc_rate = acc_sum / tf.cast(pilot_length, dtype)
        last_acc = acc_rate
        last_theta_end = theta

        print(
            f"[LuShrinkage:Tune:{step_name}] round={_round} | "
            f"acc={float(acc_rate.numpy()):.3f} | "
            f"acc_sum={float(acc_sum.numpy())}"
        )

        if acc_rate < target_low_t:
            k_next = tf.maximum(k / factor_t, eps_k)
            print(
                f"[LuShrinkage:Tune:{step_name}] round={_round} | "
                f"acc<{target_low} -> shrink k: {float(k.numpy())} -> {float(k_next.numpy())}"
            )
            k = k_next
            continue
        if acc_rate > target_high_t:
            k_next = tf.maximum(k * factor_t, eps_k)
            print(
                f"[LuShrinkage:Tune:{step_name}] round={_round} | "
                f"acc>{target_high} -> grow k: {float(k.numpy())} -> {float(k_next.numpy())}"
            )
            k = k_next
            continue

        print(
            f"[LuShrinkage:Tune:{step_name}] round={_round} | "
            f"acc in band -> stop | k={float(k.numpy())}"
        )
        break

    print(
        f"[LuShrinkage:Tune:{step_name}] done | "
        f"k={float(k.numpy())} | acc={float(last_acc.numpy()):.3f}"
    )

    return k, last_acc, last_theta_end


def tune_shrinkage(shrink):
    """Tune proposal scales for the Lu shrinkage sampler (tune once, then freeze).

    Expects `shrink` to provide:
      - pilot_length, ridge, max_lbfgs_iters
      - T, qjt, q0t, pjt, wjt
      - beta_p, beta_w, r, E_bar, njt, gamma, phi
      - posterior, rng

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

    pilot_length = int(getattr(shrink, "pilot_length"))
    if pilot_length <= 0:
        raise ValueError("pilot_length must be positive.")

    ridge = float(getattr(shrink, "ridge"))
    max_lbfgs_iters = int(getattr(shrink, "max_lbfgs_iters"))

    target_low = float(getattr(shrink, "target_low"))
    target_high = float(getattr(shrink, "target_high"))
    max_rounds = int(getattr(shrink, "max_rounds"))
    factor_rw = float(getattr(shrink, "factor_rw"))
    factor_tmh = float(getattr(shrink, "factor_tmh"))

    # Keep tuning numerics in float64 (consistent with your logpost codepaths)
    dtype = tf.float64
    ridge_t = tf.cast(tf.convert_to_tensor(ridge), dtype)

    # -----------------------------
    # Lu default k0 = 2.38 / sqrt(d)
    # -----------------------------
    # r is scalar
    k_r0 = _lu_k0(tf.constant(1.0, dtype=dtype), dtype)

    # beta is length 2: [beta_p, beta_w]
    k_beta0 = _lu_k0(tf.constant(2.0, dtype=dtype), dtype)

    # E_bar[t] is scalar per market in this implementation
    k_E_bar0 = _lu_k0(tf.constant(1.0, dtype=dtype), dtype)

    # njt[t] is length J
    njt0_any = tf.convert_to_tensor(shrink.njt[0])
    J = tf.cast(tf.shape(njt0_any)[0], dtype)
    k_njt0 = _lu_k0(J, dtype)

    print(
        "[LuShrinkage:Tune] begin | pilot_length=",
        pilot_length,
        "| T=",
        int(getattr(shrink, "T")),
        "| J=",
        int(tf.shape(njt0_any)[0].numpy()),
        "| ridge=",
        ridge,
        "| max_lbfgs_iters=",
        max_lbfgs_iters,
    )
    print(
        "[LuShrinkage:Tune] k0 defaults | k_r0=",
        float(k_r0.numpy()),
        "| k_E_bar0=",
        float(k_E_bar0.numpy()),
        "| k_beta0=",
        float(k_beta0.numpy()),
        "| k_njt0=",
        float(k_njt0.numpy()),
    )

    # -----------------------------
    # 1) Tune k_r (RW-MH on scalar r; log posterior sums over markets)
    # -----------------------------
    def step_r(theta, k):
        def logp_r(r_val: tf.Tensor) -> tf.Tensor:
            ll = tf.constant(0.0, dtype=dtype)
            for t in range(shrink.T):
                ll += shrink.posterior.market_loglik(
                    qjt_t=shrink.qjt[t],
                    q0t_t=shrink.q0t[t],
                    pjt_t=shrink.pjt[t],
                    wjt_t=shrink.wjt[t],
                    beta_p=shrink.beta_p,
                    beta_w=shrink.beta_w,
                    r=r_val,
                    E_bar_t=shrink.E_bar[t],
                    njt_t=shrink.njt[t],
                )
            return ll + shrink.posterior.logprior_r(r=r_val)

        theta_new, accepted, _ = rw_mh_step(
            theta0=tf.cast(theta, dtype),
            logp_fn=logp_r,
            k=tf.cast(k, dtype),
            rng=shrink.rng,
        )
        return theta_new, accepted

    theta_r0 = (
        shrink.r.read_value()
        if hasattr(shrink.r, "read_value")
        else tf.convert_to_tensor(shrink.r)
    )

    k_r_tuned, acc_r, _ = tune_k(
        theta0=tf.cast(theta_r0, dtype),
        step_fn=step_r,
        k0=k_r0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_rw,
    )
    print(
        "[LuShrinkage:Tune] k_r final:",
        float(k_r0.numpy()),
        "->",
        float(k_r_tuned.numpy()),
        "| acc_r=",
        float(acc_r.numpy()),
    )

    # -----------------------------
    # 2) Tune k_beta (TMH on beta vector; log posterior sums over markets)
    # -----------------------------
    def step_beta(theta, k):
        def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
            beta_p = theta_vec[0]
            beta_w = theta_vec[1]
            ll = tf.constant(0.0, dtype=dtype)
            for t in range(shrink.T):
                ll += shrink.posterior.market_loglik(
                    qjt_t=shrink.qjt[t],
                    q0t_t=shrink.q0t[t],
                    pjt_t=shrink.pjt[t],
                    wjt_t=shrink.wjt[t],
                    beta_p=beta_p,
                    beta_w=beta_w,
                    r=shrink.r,
                    E_bar_t=shrink.E_bar[t],
                    njt_t=shrink.njt[t],
                )
            return ll + shrink.posterior.logprior_beta(beta_p=beta_p, beta_w=beta_w)

        theta_new, accepted = tmh_step(
            theta0=tf.cast(theta, dtype),
            logp_fn=logp_beta,
            ridge=ridge_t,
            rng=shrink.rng,
            k=tf.cast(k, dtype),
        )
        return theta_new, accepted

    beta0 = tf.stack(
        [tf.cast(shrink.beta_p, dtype), tf.cast(shrink.beta_w, dtype)], axis=0
    )
    k_beta_tuned, acc_beta, _ = tune_k(
        theta0=beta0,
        step_fn=step_beta,
        k0=k_beta0,
        pilot_length=pilot_length,
        target_low=target_low,
        target_high=target_high,
        max_rounds=max_rounds,
        factor=factor_tmh,
    )
    print(
        "[LuShrinkage:Tune] k_beta final:",
        float(k_beta0.numpy()),
        "->",
        float(k_beta_tuned.numpy()),
        "| acc_beta=",
        float(acc_beta.numpy()),
    )

    # -----------------------------
    # 3) Tune k_E_bar (RW-MH per market; aggregate acceptance)
    # -----------------------------

    k_E_bar = tf.cast(k_E_bar0, dtype)
    eps_k = tf.cast(1e-12, dtype)
    total_steps = tf.cast(float(shrink.T * pilot_length), dtype)
    acc_E_bar = tf.cast(0.0, dtype)

    print(
        "[LuShrinkage:Tune:E_bar] start | "
        f"target=[{target_low},{target_high}] | "
        f"max_rounds={max_rounds} | "
        f"steps_per_round={int(total_steps.numpy())}"
    )

    for _round in range(int(max_rounds)):
        print(f"[LuShrinkage:Tune:E_bar] round={_round} | k={float(k_E_bar.numpy())}")

        total_acc = tf.cast(0.0, dtype)
        k_E_bar_run = tf.maximum(k_E_bar, eps_k)

        for t in range(shrink.T):
            theta = tf.cast(tf.convert_to_tensor(shrink.E_bar[t]), dtype)

            def logp_E_bar_t(E_bar_t_val: tf.Tensor) -> tf.Tensor:
                return shrink.posterior.market_logpost(
                    qjt_t=shrink.qjt[t],
                    q0t_t=shrink.q0t[t],
                    pjt_t=shrink.pjt[t],
                    wjt_t=shrink.wjt[t],
                    beta_p=shrink.beta_p,
                    beta_w=shrink.beta_w,
                    r=shrink.r,
                    E_bar_t=E_bar_t_val,
                    njt_t=shrink.njt[t],
                    gamma_t=shrink.gamma[t],
                    phi_t=shrink.phi[t],
                )

            for _ in range(int(pilot_length)):
                theta, accepted, _ = rw_mh_step(
                    theta0=theta,
                    logp_fn=logp_E_bar_t,
                    k=k_E_bar_run,
                    rng=shrink.rng,
                )
                total_acc += tf.cast(accepted, dtype)

        acc_E_bar = total_acc / total_steps

        print(
            f"[LuShrinkage:Tune:E_bar] round={_round} | "
            f"acc={float(acc_E_bar.numpy()):.3f} | "
            f"acc_sum={float(total_acc.numpy())}"
        )

        if acc_E_bar < tf.cast(target_low, dtype):
            k_next = tf.maximum(k_E_bar / tf.cast(factor_rw, dtype), eps_k)
            print(
                f"[LuShrinkage:Tune:E_bar] round={_round} | "
                f"acc<{target_low} -> shrink k: {float(k_E_bar.numpy())} -> {float(k_next.numpy())}"
            )
            k_E_bar = k_next
            continue
        if acc_E_bar > tf.cast(target_high, dtype):
            k_next = tf.maximum(k_E_bar * tf.cast(factor_rw, dtype), eps_k)
            print(
                f"[LuShrinkage:Tune:E_bar] round={_round} | "
                f"acc>{target_high} -> grow k: {float(k_E_bar.numpy())} -> {float(k_next.numpy())}"
            )
            k_E_bar = k_next
            continue

        print(f"[LuShrinkage:Tune:E_bar] round={_round} | acc in band -> stop")
        break

    k_E_bar_tuned = k_E_bar
    print(
        "[LuShrinkage:Tune] k_E_bar final:",
        float(k_E_bar0.numpy()),
        "->",
        float(k_E_bar_tuned.numpy()),
        "| acc_E_bar=",
        float(acc_E_bar.numpy()),
    )

    # -----------------------------
    # 4) Tune k_njt (TMH per market; aggregate acceptance)
    # -----------------------------
    k_njt = tf.cast(k_njt0, dtype)
    acc_njt = tf.cast(0.0, dtype)

    print(
        "[LuShrinkage:Tune:njt] start | "
        f"target=[{target_low},{target_high}] | "
        f"max_rounds={max_rounds} | "
        f"steps_per_round={int(total_steps.numpy())}"
    )

    # Precompute per-market logp functions and conditional modes once (freeze mu across rounds).
    logp_njt_full_fns = []
    mu_list = []

    def _make_logp_njt_full(t_: int):
        def logp_njt_full(njt_t_val: tf.Tensor) -> tf.Tensor:
            return shrink.posterior.market_logpost(
                qjt_t=shrink.qjt[t_],
                q0t_t=shrink.q0t[t_],
                pjt_t=shrink.pjt[t_],
                wjt_t=shrink.wjt[t_],
                beta_p=shrink.beta_p,
                beta_w=shrink.beta_w,
                r=shrink.r,
                E_bar_t=shrink.E_bar[t_],
                njt_t=njt_t_val,
                gamma_t=shrink.gamma[t_],
                phi_t=shrink.phi[t_],
            )

        return logp_njt_full

    for t in range(shrink.T):
        logp_fn_t = _make_logp_njt_full(t)
        logp_njt_full_fns.append(logp_fn_t)

        theta0_t = tf.cast(tf.convert_to_tensor(shrink.njt[t]), dtype)
        mu_t = _lbfgs_mode(theta0_t, logp_fn_t, max_lbfgs_iters=max_lbfgs_iters)
        mu_list.append(mu_t)

    for _round in range(int(max_rounds)):
        print(f"[LuShrinkage:Tune:njt] round={_round} | k={float(k_njt.numpy())}")

        total_acc = tf.cast(0.0, dtype)
        k_njt_run = tf.maximum(k_njt, eps_k)

        for t in range(shrink.T):
            logp_njt_full = logp_njt_full_fns[t]

            # Frozen warm-start at the precomputed conditional mode.
            theta = mu_list[t]

            for _ in range(int(pilot_length)):
                theta_new, accepted = tmh_step(
                    theta0=theta,
                    logp_fn=logp_njt_full,
                    ridge=ridge_t,
                    rng=shrink.rng,
                    k=k_njt_run,
                )
                theta = theta_new
                total_acc += tf.cast(accepted, dtype)

        acc_njt = total_acc / total_steps

        print(
            f"[LuShrinkage:Tune:njt] round={_round} | "
            f"acc={float(acc_njt.numpy()):.3f} | "
            f"acc_sum={float(total_acc.numpy())}"
        )

        if acc_njt < tf.cast(target_low, dtype):
            k_next = tf.maximum(k_njt / tf.cast(factor_tmh, dtype), eps_k)
            print(
                f"[LuShrinkage:Tune:njt] round={_round} | "
                f"acc<{target_low} -> shrink k: {float(k_njt.numpy())} -> {float(k_next.numpy())}"
            )
            k_njt = k_next
            continue
        if acc_njt > tf.cast(target_high, dtype):
            k_next = tf.maximum(k_njt * tf.cast(factor_tmh, dtype), eps_k)
            print(
                f"[LuShrinkage:Tune:njt] round={_round} | "
                f"acc>{target_high} -> grow k: {float(k_njt.numpy())} -> {float(k_next.numpy())}"
            )
            k_njt = k_next
            continue

        print(f"[LuShrinkage:Tune:njt] round={_round} | acc in band -> stop")
        break

    k_njt_tuned = k_njt
    print(
        "[LuShrinkage:Tune] k_njt final:",
        float(k_njt0.numpy()),
        "->",
        float(k_njt_tuned.numpy()),
        "| acc_njt=",
        float(acc_njt.numpy()),
    )

    print("[LuShrinkage:Tune] done")

    return k_r_tuned, k_E_bar_tuned, k_beta_tuned, k_njt_tuned
