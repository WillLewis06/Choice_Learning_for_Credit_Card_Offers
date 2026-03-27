"""One-step MCMC updates for the choice-learn shrinkage sampler."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from lu.choice_learn.cl_posterior import ChoiceLearnPosteriorTF
from lu.lu_gibbs import gibbs_gamma


def _make_rw_kernel(
    target_log_prob_fn,
    scale: tf.Tensor,
) -> tfp.mcmc.RandomWalkMetropolis:
    """Construct a random-walk Metropolis kernel with Gaussian proposals."""

    return tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale),
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def alpha_one_step(
    posterior: ChoiceLearnPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_alpha: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for alpha."""

    def target_log_prob_fn(alpha_val: tf.Tensor) -> tf.Tensor:
        """Evaluate the alpha-block log posterior at the proposed value."""

        return posterior.alpha_block_logpost(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha_val,
            E_bar=E_bar,
            njt=njt,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_alpha)
    kernel_results = kernel.bootstrap_results(alpha)
    alpha_new, kernel_results = kernel.one_step(
        current_state=alpha,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return alpha_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def _E_bar_market_one_step(
    posterior: ChoiceLearnPosteriorTF,
    qjt_t: tf.Tensor,
    q0t_t: tf.Tensor,
    delta_cl_t: tf.Tensor,
    alpha: tf.Tensor,
    E_bar_t: tf.Tensor,
    njt_t: tf.Tensor,
    k_E_bar: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single market's E_bar_t."""

    def target_log_prob_fn(E_bar_val: tf.Tensor) -> tf.Tensor:
        """Evaluate the one-market log posterior for E_bar_t."""

        return posterior.E_bar_block_logpost(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            delta_cl_t=delta_cl_t,
            alpha=alpha,
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

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return E_bar_t_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def E_bar_one_step(
    posterior: ChoiceLearnPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_E_bar: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full sweep of Metropolis updates over E_bar."""

    T_t = tf.shape(E_bar)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=T_t)

    ta_E_bar = tf.TensorArray(
        dtype=tf.float64,
        size=T_t,
        element_shape=(),
    ).unstack(E_bar)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(t, ta_in, accepted_sum):
        """Continue until all markets have been updated."""

        return t < T_t

    def body(t, ta_in, accepted_sum):
        """Update one market's E_bar_t and accumulate acceptance."""

        E_bar_t_old = ta_in.read(t)

        E_bar_t_new, accepted_t = _E_bar_market_one_step(
            posterior=posterior,
            qjt_t=qjt[t],
            q0t_t=q0t[t],
            delta_cl_t=delta_cl[t],
            alpha=alpha,
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

    accept_rate = accepted_sum / tf.cast(T_t, tf.float64)
    return E_bar_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def _njt_market_one_step(
    posterior: ChoiceLearnPosteriorTF,
    qjt_t: tf.Tensor,
    q0t_t: tf.Tensor,
    delta_cl_t: tf.Tensor,
    alpha: tf.Tensor,
    E_bar_t: tf.Tensor,
    njt_t: tf.Tensor,
    gamma_t: tf.Tensor,
    k_njt: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single market's njt_t."""

    def target_log_prob_fn(njt_val: tf.Tensor) -> tf.Tensor:
        """Evaluate the one-market log posterior for njt_t."""

        return posterior.njt_block_logpost(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            delta_cl_t=delta_cl_t,
            alpha=alpha,
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

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return njt_t_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def njt_one_step(
    posterior: ChoiceLearnPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    k_njt: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full sweep of Metropolis updates over njt."""

    T_t = tf.shape(njt)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=T_t)

    ta_njt = tf.TensorArray(
        dtype=tf.float64,
        size=T_t,
        element_shape=njt.shape[1:],
    ).unstack(njt)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(t, ta_in, accepted_sum):
        """Continue until all markets have been updated."""

        return t < T_t

    def body(t, ta_in, accepted_sum):
        """Update one market's njt_t and accumulate acceptance."""

        njt_t_old = ta_in.read(t)

        njt_t_new, accepted_t = _njt_market_one_step(
            posterior=posterior,
            qjt_t=qjt[t],
            q0t_t=q0t[t],
            delta_cl_t=delta_cl[t],
            alpha=alpha,
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

    accept_rate = accepted_sum / tf.cast(T_t, tf.float64)
    return njt_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def gamma_one_step(
    posterior: ChoiceLearnPosteriorTF,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Perform one Gibbs sweep for gamma under the collapsed inclusion prior."""

    return gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        T0_sq=posterior.T0_sq,
        T1_sq=posterior.T1_sq,
        seed=seed,
    )
