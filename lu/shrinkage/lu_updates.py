from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from lu.shrinkage.lu_posterior import LuPosteriorTF

tfmcmc = tfp.mcmc


def _normalize_seed(seed: tf.Tensor | None) -> tf.Tensor:
    if seed is None:
        return tf.constant([0, 0], dtype=tf.int32)

    seed_t = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed_t.shape.rank == 0:
        return tf.stack([seed_t, tf.constant(0, dtype=tf.int32)])
    return seed_t


def _make_rw_kernel(
    target_log_prob_fn,
    scale: tf.Tensor,
) -> tfmcmc.RandomWalkMetropolis:
    return tfmcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfmcmc.random_walk_normal_fn(scale=scale),
    )


def _accepted_float(
    kernel_results,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    return tf.cast(kernel_results.is_accepted, dtype)


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_one_step(
    posterior: LuPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_beta: tf.Tensor,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    seed = _normalize_seed(seed)
    k_beta = tf.convert_to_tensor(k_beta, dtype=posterior.dtype)

    def target_log_prob_fn(beta: tf.Tensor) -> tf.Tensor:
        return posterior.beta_block_logpost(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta[0],
            beta_w=beta[1],
            r=r,
            E_bar=E_bar,
            njt=njt,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_beta)
    beta0 = tf.stack([beta_p, beta_w])
    kernel_results = kernel.bootstrap_results(beta0)
    beta_new, kernel_results = kernel.one_step(
        current_state=beta0,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = _accepted_float(kernel_results=kernel_results, dtype=posterior.dtype)
    return beta_new[0], beta_new[1], accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def r_one_step(
    posterior: LuPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_r: tf.Tensor,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _normalize_seed(seed)
    k_r = tf.convert_to_tensor(k_r, dtype=posterior.dtype)

    def target_log_prob_fn(r_val: tf.Tensor) -> tf.Tensor:
        return posterior.r_block_logpost(
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

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_r)
    kernel_results = kernel.bootstrap_results(r)
    r_new, kernel_results = kernel.one_step(
        current_state=r,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = _accepted_float(kernel_results=kernel_results, dtype=posterior.dtype)
    return r_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def _E_bar_market_one_step(
    posterior: LuPosteriorTF,
    qjt_t: tf.Tensor,
    q0t_t: tf.Tensor,
    pjt_t: tf.Tensor,
    wjt_t: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar_t: tf.Tensor,
    njt_t: tf.Tensor,
    k_E_bar: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    def target_log_prob_fn(E_bar_val: tf.Tensor) -> tf.Tensor:
        return posterior.E_bar_block_logpost(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_val,
            njt_t=njt_t,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_E_bar)
    kernel_results = kernel.bootstrap_results(E_bar_t)
    E_bar_t_new, kernel_results = kernel.one_step(
        current_state=E_bar_t,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = _accepted_float(kernel_results=kernel_results, dtype=posterior.dtype)
    return E_bar_t_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def E_bar_one_step(
    posterior: LuPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_E_bar: tf.Tensor,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _normalize_seed(seed)
    k_E_bar = tf.convert_to_tensor(k_E_bar, dtype=posterior.dtype)

    T_t = tf.shape(E_bar)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=T_t)

    ta_E_bar = tf.TensorArray(
        dtype=posterior.dtype,
        size=T_t,
        element_shape=(),
    ).unstack(E_bar)
    accepted0 = tf.constant(0.0, dtype=posterior.dtype)

    def cond(t, ta_in, accepted_sum):
        del ta_in, accepted_sum
        return t < T_t

    def body(t, ta_in, accepted_sum):
        E_bar_t_old = ta_in.read(t)

        E_bar_t_new, accepted_t = _E_bar_market_one_step(
            posterior=posterior,
            qjt_t=qjt[t],
            q0t_t=q0t[t],
            pjt_t=pjt[t],
            wjt_t=wjt[t],
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t_old,
            njt_t=njt[t],
            k_E_bar=k_E_bar,
            seed=seeds[t],
        )

        ta_out = ta_in.write(t, E_bar_t_new)
        return t + 1, ta_out, accepted_sum + accepted_t

    _, ta_E_bar, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_E_bar,
            accepted0,
        ),
    )

    E_bar_new = ta_E_bar.stack()
    E_bar_new = tf.ensure_shape(E_bar_new, E_bar.shape)
    accept_rate = accepted_sum / tf.cast(T_t, posterior.dtype)
    return E_bar_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def _njt_market_one_step(
    posterior: LuPosteriorTF,
    qjt_t: tf.Tensor,
    q0t_t: tf.Tensor,
    pjt_t: tf.Tensor,
    wjt_t: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar_t: tf.Tensor,
    njt_t: tf.Tensor,
    gamma_t: tf.Tensor,
    k_njt: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    def target_log_prob_fn(njt_val: tf.Tensor) -> tf.Tensor:
        return posterior.njt_block_logpost(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t,
            njt_t=njt_val,
            gamma_t=gamma_t,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_njt)
    kernel_results = kernel.bootstrap_results(njt_t)
    njt_t_new, kernel_results = kernel.one_step(
        current_state=njt_t,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = _accepted_float(kernel_results=kernel_results, dtype=posterior.dtype)
    return njt_t_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def njt_one_step(
    posterior: LuPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    k_njt: tf.Tensor,
    seed: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    seed = _normalize_seed(seed)
    k_njt = tf.convert_to_tensor(k_njt, dtype=posterior.dtype)

    T_t = tf.shape(njt)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=T_t)

    ta_njt = tf.TensorArray(
        dtype=posterior.dtype,
        size=T_t,
        element_shape=njt.shape[1:],
    ).unstack(njt)
    accepted0 = tf.constant(0.0, dtype=posterior.dtype)

    def cond(t, ta_in, accepted_sum):
        del ta_in, accepted_sum
        return t < T_t

    def body(t, ta_in, accepted_sum):
        njt_t_old = ta_in.read(t)

        njt_t_new, accepted_t = _njt_market_one_step(
            posterior=posterior,
            qjt_t=qjt[t],
            q0t_t=q0t[t],
            pjt_t=pjt[t],
            wjt_t=wjt[t],
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar[t],
            njt_t=njt_t_old,
            gamma_t=gamma[t],
            k_njt=k_njt,
            seed=seeds[t],
        )

        ta_out = ta_in.write(t, njt_t_new)
        return t + 1, ta_out, accepted_sum + accepted_t

    _, ta_njt, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_njt,
            accepted0,
        ),
    )

    njt_new = ta_njt.stack()
    njt_new = tf.ensure_shape(njt_new, njt.shape)
    accept_rate = accepted_sum / tf.cast(T_t, posterior.dtype)
    return njt_new, accept_rate
