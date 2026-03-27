"""One-step MCMC updates for the Ching-style stockpiling sampler."""

from __future__ import annotations

from collections.abc import Callable

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
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    scale: tf.Tensor,
) -> tfp.mcmc.RandomWalkMetropolis:
    """Construct a Gaussian random-walk Metropolis kernel."""
    return tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=scale),
    )


@tf.function(jit_compile=True)
def _replace_1d_entry(x: tf.Tensor, idx: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    """Return a copy of x with x[idx] replaced by value."""
    return tf.tensor_scatter_nd_update(
        x,
        indices=[[idx]],
        updates=[tf.reshape(value, ())],
    )


@tf.function(jit_compile=True)
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
    """Run one Metropolis update for the scalar beta block."""

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


@tf.function(jit_compile=True)
def _alpha_entry_one_step(
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
    """Run one Metropolis update for a single alpha entry."""
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


@tf.function(jit_compile=True)
def _v_entry_one_step(
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
    """Run one Metropolis update for a single v entry."""
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


@tf.function(jit_compile=True)
def _fc_entry_one_step(
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
    """Run one Metropolis update for a single fc entry."""
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


@tf.function(jit_compile=True)
def _u_scale_entry_one_step(
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
    """Run one Metropolis update for a single u_scale entry."""
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


@tf.function(jit_compile=True)
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
    """Run one full coordinate-wise Metropolis sweep over alpha."""
    J = tf.shape(z_alpha)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(i: tf.Tensor, _: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
        return i < J

    def body(
        i: tf.Tensor,
        current_alpha: tf.Tensor,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        value_new, accepted_i = _alpha_entry_one_step(
            posterior=posterior,
            j=i,
            z_beta=z_beta,
            z_alpha=current_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            k_alpha_j=k_alpha[i],
            seed=seeds[i],
        )
        current_alpha = _replace_1d_entry(current_alpha, i, value_new)
        current_alpha = tf.ensure_shape(current_alpha, z_alpha.shape)
        return i + 1, current_alpha, accepted_sum + accepted_i

    _, z_alpha_new, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0, dtype=tf.int32), z_alpha, accepted0),
    )
    return z_alpha_new, accepted_sum / tf.cast(J, tf.float64)


@tf.function(jit_compile=True)
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
    """Run one full coordinate-wise Metropolis sweep over v."""
    J = tf.shape(z_v)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(i: tf.Tensor, _: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
        return i < J

    def body(
        i: tf.Tensor,
        current_v: tf.Tensor,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        value_new, accepted_i = _v_entry_one_step(
            posterior=posterior,
            j=i,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=current_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
            k_v_j=k_v[i],
            seed=seeds[i],
        )
        current_v = _replace_1d_entry(current_v, i, value_new)
        current_v = tf.ensure_shape(current_v, z_v.shape)
        return i + 1, current_v, accepted_sum + accepted_i

    _, z_v_new, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0, dtype=tf.int32), z_v, accepted0),
    )
    return z_v_new, accepted_sum / tf.cast(J, tf.float64)


@tf.function(jit_compile=True)
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
    """Run one full coordinate-wise Metropolis sweep over fc."""
    J = tf.shape(z_fc)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=J)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(i: tf.Tensor, _: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
        return i < J

    def body(
        i: tf.Tensor,
        current_fc: tf.Tensor,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        value_new, accepted_i = _fc_entry_one_step(
            posterior=posterior,
            j=i,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=current_fc,
            z_u_scale=z_u_scale,
            k_fc_j=k_fc[i],
            seed=seeds[i],
        )
        current_fc = _replace_1d_entry(current_fc, i, value_new)
        current_fc = tf.ensure_shape(current_fc, z_fc.shape)
        return i + 1, current_fc, accepted_sum + accepted_i

    _, z_fc_new, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0, dtype=tf.int32), z_fc, accepted0),
    )
    return z_fc_new, accepted_sum / tf.cast(J, tf.float64)


@tf.function(jit_compile=True)
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
    """Run one full coordinate-wise Metropolis sweep over u_scale."""
    M = tf.shape(z_u_scale)[0]
    seeds = tf.random.experimental.stateless_split(seed, num=M)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(i: tf.Tensor, _: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
        return i < M

    def body(
        i: tf.Tensor,
        current_u_scale: tf.Tensor,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        value_new, accepted_i = _u_scale_entry_one_step(
            posterior=posterior,
            m=i,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=current_u_scale,
            k_u_scale_m=k_u_scale[i],
            seed=seeds[i],
        )
        current_u_scale = _replace_1d_entry(current_u_scale, i, value_new)
        current_u_scale = tf.ensure_shape(current_u_scale, z_u_scale.shape)
        return i + 1, current_u_scale, accepted_sum + accepted_i

    _, z_u_scale_new, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(tf.constant(0, dtype=tf.int32), z_u_scale, accepted0),
    )
    return z_u_scale_new, accepted_sum / tf.cast(M, tf.float64)
