"""One-step MCMC updates for the choice-learn shrinkage sampler."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

from lu.choice_learn.cl_posterior import ChoiceLearnPosteriorTF
from lu.lu_gibbs import gibbs_gamma


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
    """Perform one random-walk Metropolis update for the scalar alpha block."""

    def target_log_prob_fn(alpha_val: tf.Tensor) -> tf.Tensor:
        """Return the alpha-block log posterior at the proposed value."""

        return posterior.alpha_block_logpost(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha_val,
            E_bar=E_bar,
            njt=njt,
        )

    # Use the tuned proposal scale directly for the scalar alpha update.
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=k_alpha),
    )
    kernel_results = kernel.bootstrap_results(alpha)
    alpha_new, kernel_results = kernel.one_step(
        current_state=alpha,
        previous_kernel_results=kernel_results,
        seed=seed,
    )

    accepted = tf.cast(kernel_results.is_accepted, tf.float64)
    return alpha_new, accepted


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
    """Perform one full Metropolis sweep over the market-level E_bar vector."""

    num_markets = tf.shape(E_bar)[0]

    # Split one sweep seed into deterministic market-level proposal seeds.
    market_seeds = tf.random.experimental.stateless_split(seed, num=num_markets)

    # Store the updated scalar state for each market as the sweep progresses.
    E_bar_array = tf.TensorArray(
        dtype=tf.float64,
        size=num_markets,
        element_shape=(),
    ).unstack(E_bar)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(
        market_index: tf.Tensor,
        E_bar_array_in: tf.TensorArray,
        accepted_sum: tf.Tensor,
    ) -> tf.Tensor:
        """Continue until every market has been updated once."""

        del E_bar_array_in, accepted_sum
        return market_index < num_markets

    def body(
        market_index: tf.Tensor,
        E_bar_array_in: tf.TensorArray,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.TensorArray, tf.Tensor]:
        """Update one market-level common shock and accumulate acceptance."""

        E_bar_old = E_bar_array_in.read(market_index)

        def target_log_prob_fn(E_bar_val: tf.Tensor) -> tf.Tensor:
            """Return the one-market E_bar block log posterior."""

            return posterior.E_bar_block_logpost(
                qjt_t=qjt[market_index],
                q0t_t=q0t[market_index],
                delta_cl_t=delta_cl[market_index],
                alpha=alpha,
                E_bar_t=E_bar_val,
                njt_t=njt[market_index],
            )

        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=k_E_bar),
        )
        kernel_results = kernel.bootstrap_results(E_bar_old)
        E_bar_new, kernel_results = kernel.one_step(
            current_state=E_bar_old,
            previous_kernel_results=kernel_results,
            seed=market_seeds[market_index],
        )

        accepted = tf.cast(kernel_results.is_accepted, tf.float64)
        E_bar_array_out = E_bar_array_in.write(market_index, E_bar_new)
        return market_index + 1, E_bar_array_out, accepted_sum + accepted

    _, E_bar_array, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            E_bar_array,
            accepted0,
        ),
    )

    E_bar_new = E_bar_array.stack()

    # Preserve the static vector shape so higher-level compiled loops do not
    # see this block output as shape (None,).
    E_bar_new = tf.ensure_shape(E_bar_new, E_bar.shape)

    accept_rate = accepted_sum / tf.cast(num_markets, tf.float64)
    return E_bar_new, accept_rate


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
    """Perform one full Metropolis sweep over the market-by-product shock matrix."""

    num_markets = tf.shape(njt)[0]

    # Split one sweep seed into deterministic market-level proposal seeds.
    market_seeds = tf.random.experimental.stateless_split(seed, num=num_markets)

    # Store the updated product-shock vector for each market during the sweep.
    njt_array = tf.TensorArray(
        dtype=tf.float64,
        size=num_markets,
        element_shape=njt.shape[1:],
    ).unstack(njt)
    accepted0 = tf.constant(0.0, dtype=tf.float64)

    def cond(
        market_index: tf.Tensor,
        njt_array_in: tf.TensorArray,
        accepted_sum: tf.Tensor,
    ) -> tf.Tensor:
        """Continue until every market shock vector has been updated once."""

        del njt_array_in, accepted_sum
        return market_index < num_markets

    def body(
        market_index: tf.Tensor,
        njt_array_in: tf.TensorArray,
        accepted_sum: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.TensorArray, tf.Tensor]:
        """Update one market shock vector conditional on its gamma pattern."""

        njt_old = njt_array_in.read(market_index)

        def target_log_prob_fn(njt_val: tf.Tensor) -> tf.Tensor:
            """Return the one-market njt block log posterior."""

            return posterior.njt_block_logpost(
                qjt_t=qjt[market_index],
                q0t_t=q0t[market_index],
                delta_cl_t=delta_cl[market_index],
                alpha=alpha,
                E_bar_t=E_bar[market_index],
                njt_t=njt_val,
                gamma_t=gamma[market_index],
            )

        kernel = tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=k_njt),
        )
        kernel_results = kernel.bootstrap_results(njt_old)
        njt_new_market, kernel_results = kernel.one_step(
            current_state=njt_old,
            previous_kernel_results=kernel_results,
            seed=market_seeds[market_index],
        )

        accepted = tf.cast(kernel_results.is_accepted, tf.float64)
        njt_array_out = njt_array_in.write(market_index, njt_new_market)
        return market_index + 1, njt_array_out, accepted_sum + accepted

    _, njt_array, accepted_sum = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            njt_array,
            accepted0,
        ),
    )

    njt_new = njt_array.stack()

    # Preserve the static matrix shape so compiled callers do not see this block
    # output as shape (None, J) or (None, None).
    njt_new = tf.ensure_shape(njt_new, njt.shape)

    accept_rate = accepted_sum / tf.cast(num_markets, tf.float64)
    return njt_new, accept_rate


@tf.function(jit_compile=True, reduce_retracing=True)
def gamma_one_step(
    posterior: ChoiceLearnPosteriorTF,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Perform one Gibbs sweep for gamma under the collapsed inclusion prior."""

    # The gamma update is exact conditional on the current continuous state.
    return gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        T0_sq=posterior.T0_sq,
        T1_sq=posterior.T1_sq,
        seed=seed,
    )
