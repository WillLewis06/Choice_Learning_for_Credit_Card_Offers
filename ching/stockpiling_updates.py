"""One-step MCMC updates for the Ching-style stockpiling sampler."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from ching.stockpiling_posterior import StockpilingPosteriorTF

__all__ = [
    "beta_one_step",
    "alpha_one_step",
    "v_one_step",
    "fc_one_step",
    "u_scale_one_step",
]


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
def _replace_1d_entry(x: tf.Tensor, idx: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    """Return a copy of x with x[idx] replaced by value."""

    value = tf.reshape(value, ())
    return tf.tensor_scatter_nd_update(x, indices=[[idx]], updates=[value])


@tf.function(jit_compile=True, reduce_retracing=True)
def beta_one_step(
    posterior: StockpilingPosteriorTF,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_beta: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for the z_beta block."""

    def target_log_prob_fn(z_beta_val: tf.Tensor) -> tf.Tensor:
        return posterior.beta_block_logpost(
            z_beta=z_beta_val,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
        )

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_beta)
    kernel_results = kernel.bootstrap_results(z_beta)
    z_beta_new, kernel_results = kernel.one_step(
        current_state=z_beta,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_beta_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def _alpha_product_one_step(
    posterior: StockpilingPosteriorTF,
    j: tf.Tensor,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_alpha_j: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single product's z_alpha entry."""

    z_alpha_j = z_alpha[j]

    def target_log_prob_fn(z_alpha_j_val: tf.Tensor) -> tf.Tensor:
        z_alpha_prop = _replace_1d_entry(z_alpha, j, z_alpha_j_val)
        return posterior.alpha_block_logpost(
            z_beta=z_beta,
            z_alpha=z_alpha_prop,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
        )[j]

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_alpha_j)
    kernel_results = kernel.bootstrap_results(z_alpha_j)
    z_alpha_j_new, kernel_results = kernel.one_step(
        current_state=z_alpha_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_alpha_j_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def alpha_one_step(
    posterior: StockpilingPosteriorTF,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_alpha: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full product-by-product Metropolis sweep over z_alpha."""

    J = tf.shape(z_alpha)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)

    ta_alpha = tf.TensorArray(
        dtype=tf.float64,
        size=J,
        element_shape=(),
    ).unstack(z_alpha)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(j, ta_in, accepted_sum):
        return j < J

    def body(j, ta_in, accepted_sum):
        z_alpha_curr = ta_in.stack()
        z_alpha_j_new, accepted_j = _alpha_product_one_step(
            posterior=posterior,
            j=j,
            z_beta=z_beta,
            z_alpha=z_alpha_curr,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            k_alpha_j=k_alpha[j],
            seed=seeds[j],
        )
        ta_out = ta_in.write(j, z_alpha_j_new)
        return j + 1, ta_out, accepted_sum + accepted_j

    _, ta_alpha, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_alpha,
            accepted0,
        ),
    )

    z_alpha_new = ta_alpha.stack()
    z_alpha_new = tf.ensure_shape(z_alpha_new, z_alpha.shape)
    accept_rate = accepted_sum / tf.cast(J, tf.float64)
    return z_alpha_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def _v_product_one_step(
    posterior: StockpilingPosteriorTF,
    j: tf.Tensor,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_v_j: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single product's z_v entry."""

    z_v_j = z_v[j]

    def target_log_prob_fn(z_v_j_val: tf.Tensor) -> tf.Tensor:
        z_v_prop = _replace_1d_entry(z_v, j, z_v_j_val)
        return posterior.v_block_logpost(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v_prop,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
        )[j]

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_v_j)
    kernel_results = kernel.bootstrap_results(z_v_j)
    z_v_j_new, kernel_results = kernel.one_step(
        current_state=z_v_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_v_j_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def v_one_step(
    posterior: StockpilingPosteriorTF,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_v: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full product-by-product Metropolis sweep over z_v."""

    J = tf.shape(z_v)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)

    ta_v = tf.TensorArray(
        dtype=tf.float64,
        size=J,
        element_shape=(),
    ).unstack(z_v)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(j, ta_in, accepted_sum):
        return j < J

    def body(j, ta_in, accepted_sum):
        z_v_curr = ta_in.stack()
        z_v_j_new, accepted_j = _v_product_one_step(
            posterior=posterior,
            j=j,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v_curr,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            k_v_j=k_v[j],
            seed=seeds[j],
        )
        ta_out = ta_in.write(j, z_v_j_new)
        return j + 1, ta_out, accepted_sum + accepted_j

    _, ta_v, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_v,
            accepted0,
        ),
    )

    z_v_new = ta_v.stack()
    z_v_new = tf.ensure_shape(z_v_new, z_v.shape)
    accept_rate = accepted_sum / tf.cast(J, tf.float64)
    return z_v_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def _fc_product_one_step(
    posterior: StockpilingPosteriorTF,
    j: tf.Tensor,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_fc_j: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single product's z_fc entry."""

    z_fc_j = z_fc[j]

    def target_log_prob_fn(z_fc_j_val: tf.Tensor) -> tf.Tensor:
        z_fc_prop = _replace_1d_entry(z_fc, j, z_fc_j_val)
        return posterior.fc_block_logpost(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc_prop,
            z_u_scale=z_u_scale,
        )[j]

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_fc_j)
    kernel_results = kernel.bootstrap_results(z_fc_j)
    z_fc_j_new, kernel_results = kernel.one_step(
        current_state=z_fc_j,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_fc_j_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def fc_one_step(
    posterior: StockpilingPosteriorTF,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_fc: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full product-by-product Metropolis sweep over z_fc."""

    J = tf.shape(z_fc)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)

    ta_fc = tf.TensorArray(
        dtype=tf.float64,
        size=J,
        element_shape=(),
    ).unstack(z_fc)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(j, ta_in, accepted_sum):
        return j < J

    def body(j, ta_in, accepted_sum):
        z_fc_curr = ta_in.stack()
        z_fc_j_new, accepted_j = _fc_product_one_step(
            posterior=posterior,
            j=j,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc_curr,
            z_u_scale=z_u_scale,
            k_fc_j=k_fc[j],
            seed=seeds[j],
        )
        ta_out = ta_in.write(j, z_fc_j_new)
        return j + 1, ta_out, accepted_sum + accepted_j

    _, ta_fc, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_fc,
            accepted0,
        ),
    )

    z_fc_new = ta_fc.stack()
    z_fc_new = tf.ensure_shape(z_fc_new, z_fc.shape)
    accept_rate = accepted_sum / tf.cast(J, tf.float64)
    return z_fc_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def _u_scale_market_one_step(
    posterior: StockpilingPosteriorTF,
    m: tf.Tensor,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_u_scale_m: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one Metropolis update for a single market's z_u_scale entry."""

    z_u_scale_m = z_u_scale[m]

    def target_log_prob_fn(z_u_scale_m_val: tf.Tensor) -> tf.Tensor:
        z_u_scale_prop = _replace_1d_entry(z_u_scale, m, z_u_scale_m_val)
        return posterior.u_scale_block_logpost(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale_prop,
        )[m]

    kernel = _make_rw_kernel(target_log_prob_fn=target_log_prob_fn, scale=k_u_scale_m)
    kernel_results = kernel.bootstrap_results(z_u_scale_m)
    z_u_scale_m_new, kernel_results = kernel.one_step(
        current_state=z_u_scale_m,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return z_u_scale_m_new, accepted


@tf.function(jit_compile=True, reduce_retracing=True)
def u_scale_one_step(
    posterior: StockpilingPosteriorTF,
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
    k_u_scale: tf.Tensor,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perform one full market-by-market Metropolis sweep over z_u_scale."""

    if posterior.fix_u_scale:
        return z_u_scale, tf.constant(0.0, dtype=tf.float64)

    M = tf.shape(z_u_scale)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=M)

    ta_u_scale = tf.TensorArray(
        dtype=tf.float64,
        size=M,
        element_shape=(),
    ).unstack(z_u_scale)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(m, ta_in, accepted_sum):
        return m < M

    def body(m, ta_in, accepted_sum):
        z_u_scale_curr = ta_in.stack()
        z_u_scale_m_new, accepted_m = _u_scale_market_one_step(
            posterior=posterior,
            m=m,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale_curr,
            k_u_scale_m=k_u_scale[m],
            seed=seeds[m],
        )
        ta_out = ta_in.write(m, z_u_scale_m_new)
        return m + 1, ta_out, accepted_sum + accepted_m

    _, ta_u_scale, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            ta_u_scale,
            accepted0,
        ),
    )

    z_u_scale_new = ta_u_scale.stack()
    z_u_scale_new = tf.ensure_shape(z_u_scale_new, z_u_scale.shape)
    accept_rate = accepted_sum / tf.cast(M, tf.float64)
    return z_u_scale_new, accept_rate
