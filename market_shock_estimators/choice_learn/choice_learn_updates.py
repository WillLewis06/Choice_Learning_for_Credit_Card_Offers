"""
Block-wise MCMC updates for the choice-learn + Lu market-shocks estimator.

These functions implement one update per parameter block, holding the remaining
blocks fixed. The estimator's main MCMC iteration calls these in a fixed order.

Model (systematic utility):
  delta_{t,j} = alpha * delta_cl_{t,j} + E_bar[t] + njt[t,j]

Design notes:
  - RW-MH is used for scalar or vector blocks with simple random-walk proposals.
  - TMH is used for dense blocks where local curvature improves proposals.
  - Gibbs steps are used where the conditional posterior is conjugate.

Conventions:
  - `posterior` supplies log-likelihood and log-prior helpers plus hyperparameters
    (e.g. spike/slab variances and Beta prior parameters).
  - `rng` is a tf.random.Generator passed through so the entire chain is
    reproducible and graph-compatible.
  - All tensors are tf.float64.
"""

from __future__ import annotations

import tensorflow as tf

from market_shock_estimators.choice_learn.choice_learn_kernels import (
    gibbs_gamma,
    gibbs_phi,
    rw_mh_step,
    tmh_step,
)


@tf.function(reduce_retracing=True)
def update_alpha(
    posterior,
    rng: tf.random.Generator,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    k_alpha: tf.Tensor,
):
    """RW-MH update for the global scaling parameter alpha.

    Target density:
      log p(alpha | rest) ∝
          sum_t loglik_t(alpha, E_bar, njt; delta_cl)
        + logprior_global(alpha)

    Returns:
        alpha_new: Updated alpha (scalar).
        accepted: Boolean scalar acceptance indicator.
    """

    def logp_alpha(alpha_val: tf.Tensor) -> tf.Tensor:
        ll_t = posterior.loglik_vec(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha_val,
            E_bar=E_bar,
            njt=njt,
        )
        ll = tf.reduce_sum(ll_t)
        lp = posterior.logprior_global(alpha=alpha_val)
        return ll + lp

    alpha_new, accepted = rw_mh_step(
        theta0=alpha, logp_fn=logp_alpha, k=k_alpha, rng=rng
    )
    return alpha_new, accepted


@tf.function(reduce_retracing=True)
def update_E_bar(
    posterior,
    rng: tf.random.Generator,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    phi: tf.Tensor,
    k_E_bar: tf.Tensor,
):
    """Elementwise RW-MH update for the market shock vector E_bar.

    E_bar has shape (T,) and enters utility as a market-level component:
      delta_{t,j} = alpha * delta_cl_{t,j} + E_bar[t] + njt[t,j].

    This update is "batched": the RW-MH kernel proposes a vector E_bar' and
    accepts/rejects each market component independently. This matches the way
    posterior.logpost_vec returns per-market log posterior contributions.

    Returns:
        E_bar_new: Updated vector of shape (T,).
        accepted: Boolean vector of shape (T,) indicating accepted markets.
    """

    def logp_E_bar_vec(E_bar_val: tf.Tensor) -> tf.Tensor:
        return posterior.logpost_vec(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar_val,
            njt=njt,
            gamma=gamma,
            phi=phi,
        )

    E_bar_new, accepted = rw_mh_step(
        theta0=E_bar, logp_fn=logp_E_bar_vec, k=k_E_bar, rng=rng
    )
    return E_bar_new, accepted


@tf.function(reduce_retracing=True)
def update_njt(
    posterior,
    rng: tf.random.Generator,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    alpha: tf.Tensor,
    E_bar: tf.Tensor,
    njt: tf.Tensor,
    gamma: tf.Tensor,
    phi: tf.Tensor,
    k_njt: tf.Tensor,
    ridge: tf.Tensor,
):
    """Sequential TMH sweep updating each market's n_t = (n_{1,t},...,n_{J,t}).

    njt has shape (T, J). This function updates it market-by-market:
      for t = 0..T-1:
        update the J-vector njt[t] using TMH, conditioning on everything else.

    The market loop is implemented with tf.while_loop and parallel_iterations=1
    to keep the update order deterministic with respect to the RNG stream.

    Returns:
        njt_new: Updated njt, shape (T, J).
        acc_sum: Float64 scalar count of accepted market updates in the sweep.
    """
    T_t = tf.shape(njt)[0]
    ta_n = tf.TensorArray(tf.float64, size=T_t).unstack(njt)

    def cond(t, ta_in, acc_sum):
        return t < T_t

    def body(t, ta_in, acc_sum):
        qjt_t = qjt[t]
        q0t_t = q0t[t]
        delta_cl_t = delta_cl[t]

        E_bar_t = E_bar[t]
        gamma_t = gamma[t]
        phi_t = phi[t]
        njt_t = ta_in.read(t)

        def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
            ll = posterior.market_loglik(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                delta_cl_t=delta_cl_t,
                alpha=alpha,
                E_bar_t=E_bar_t,
                njt_t=njt_t_val,
            )

            lp_1 = posterior.logprior_market_vec(
                E_bar=tf.reshape(E_bar_t, (1,)),
                njt=tf.expand_dims(njt_t_val, axis=0),
                gamma=tf.expand_dims(gamma_t, axis=0),
                phi=tf.reshape(phi_t, (1,)),
            )
            return ll + lp_1[0]

        njt_new_t, accepted = tmh_step(
            theta0=njt_t,
            logp_fn=logp_njt_t,
            ridge=ridge,
            rng=rng,
            k=k_njt,
        )

        ta_out = ta_in.write(t, njt_new_t)
        acc_sum = acc_sum + tf.cast(accepted, tf.float64)
        return t + 1, ta_out, acc_sum

    t0 = tf.constant(0, tf.int32)
    acc0 = tf.constant(0.0, tf.float64)

    _, ta_out, acc_sum = tf.while_loop(
        cond,
        body,
        loop_vars=(t0, ta_n, acc0),
        parallel_iterations=1,
    )

    return ta_out.stack(), acc_sum


@tf.function(reduce_retracing=True)
def update_gamma(
    posterior,
    rng: tf.random.Generator,
    njt: tf.Tensor,
    phi: tf.Tensor,
):
    """Gibbs update for gamma given njt and phi.

    Returns:
        gamma_new: Updated gamma, shape (T, J), float64 in {0,1}.
    """
    return gibbs_gamma(
        njt_t=njt,
        phi_t=phi[:, None],
        T0_sq=posterior.T0_sq,
        T1_sq=posterior.T1_sq,
        log_T0_sq=posterior.log_T0_sq,
        log_T1_sq=posterior.log_T1_sq,
        rng=rng,
    )


@tf.function(reduce_retracing=True)
def update_phi(
    posterior,
    rng: tf.random.Generator,
    gamma: tf.Tensor,
):
    """Gibbs update for phi given gamma.

    Returns:
        phi_new: Updated phi, shape (T,).
    """
    return gibbs_phi(
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        rng=rng,
    )
