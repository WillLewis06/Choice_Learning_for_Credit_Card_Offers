"""
Block-wise MCMC updates for the Lu shrinkage estimator.

These functions implement one update per parameter block, holding the remaining
blocks fixed. The estimator's main MCMC iteration calls these in a fixed order.

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

from market_shock_estimators.lu_kernels import (
    gibbs_gamma,
    gibbs_phi,
    rw_mh_step,
    tmh_step,
)


@tf.function(reduce_retracing=True)
def update_beta(
    posterior,
    rng: tf.random.Generator,
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
    ridge: tf.Tensor,
):
    """TMH update for the global coefficients (beta_p, beta_w).

    This block updates the mean price coefficient beta_p and the characteristic
    coefficient beta_w jointly. Joint updating matters because these parameters
    are typically correlated in the posterior.

    Target density:
      log p(beta_p, beta_w | rest) ∝
          sum_t loglik_t(beta_p, beta_w, r, E_bar, njt)
        + logprior_global(beta_p, beta_w, r)

    Returns:
        beta_p_new: Updated beta_p.
        beta_w_new: Updated beta_w.
        accepted: Boolean scalar acceptance indicator from TMH.
    """
    # Represent the two scalars as a single length-2 vector for a joint TMH move.
    beta0 = tf.stack([beta_p, beta_w], axis=0)

    def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
        """Return log posterior for theta_vec = [beta_p, beta_w]."""
        bp = theta_vec[0]
        bw = theta_vec[1]

        # Likelihood decomposes across markets; sum to get the full log-likelihood.
        ll_t = posterior.loglik_vec(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=bp,
            beta_w=bw,
            r=r,
            E_bar=E_bar,
            njt=njt,
        )
        ll = tf.reduce_sum(ll_t)

        # Global prior includes r, but r is held fixed in this conditional update.
        lp = posterior.logprior_global(beta_p=bp, beta_w=bw, r=r)
        return ll + lp

    # TMH uses local curvature of logp_beta to propose an informed joint move.
    beta_new, accepted = tmh_step(
        theta0=beta0,
        logp_fn=logp_beta,
        ridge=ridge,
        rng=rng,
        k=k_beta,
    )
    return beta_new[0], beta_new[1], accepted


@tf.function(reduce_retracing=True)
def update_r(
    posterior,
    rng: tf.random.Generator,
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
):
    """RW-MH update for r = log(sigma).

    Here sigma is the standard deviation of the random coefficient on price.
    The model parameterization uses r = log(sigma) so sigma = exp(r) is positive.

    Target density:
      log p(r | rest) ∝
          sum_t loglik_t(beta_p, beta_w, r, E_bar, njt)
        + logprior_global(beta_p, beta_w, r)

    Returns:
        r_new: Updated r.
        accepted: Boolean scalar acceptance indicator.
    """

    def logp_r(r_val: tf.Tensor) -> tf.Tensor:
        """Return log posterior for r_val."""
        ll_t = posterior.loglik_vec(
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
        ll = tf.reduce_sum(ll_t)
        lp = posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r_val)
        return ll + lp

    # Scalar RW-MH: symmetric proposal, simple acceptance ratio.
    r_new, accepted = rw_mh_step(theta0=r, logp_fn=logp_r, k=k_r, rng=rng)
    return r_new, accepted


@tf.function(reduce_retracing=True)
def update_E_bar(
    posterior,
    rng: tf.random.Generator,
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
    phi: tf.Tensor,
    k_E_bar: tf.Tensor,
):
    """Elementwise RW-MH update for the market shock vector E_bar.

    E_bar has shape (T,) and enters utility as a market-level component:
      E_{j,t} = E_bar_t + n_{j,t}.

    This update is "batched": the RW-MH kernel proposes a vector E_bar' and
    accepts/rejects each market component independently. This matches the way
    posterior.logpost_vec returns per-market log posterior contributions.

    Returns:
        E_bar_new: Updated vector of shape (T,).
        accepted: Boolean vector of shape (T,) indicating accepted markets.
    """

    def logp_E_bar_vec(E_bar_val: tf.Tensor) -> tf.Tensor:
        """Return per-market log posterior as a function of E_bar (shape (T,))."""
        return posterior.logpost_vec(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
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
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    beta_p: tf.Tensor,
    beta_w: tf.Tensor,
    r: tf.Tensor,
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

    # TensorArray enables in-graph updates of a (T,J) tensor one row at a time.
    ta_n = tf.TensorArray(tf.float64, size=T_t).unstack(njt)

    def cond(t, ta_in, acc_sum):
        """Continue until all markets have been updated."""
        return t < T_t

    def body(t, ta_in, acc_sum):
        """Update market t using a TMH step on the J-dimensional vector."""
        # Slice market-specific observed data.
        qjt_t = qjt[t]
        q0t_t = q0t[t]
        pjt_t = pjt[t]
        wjt_t = wjt[t]

        # Slice market-specific latent components and current state.
        E_bar_t = E_bar[t]
        gamma_t = gamma[t]
        phi_t = phi[t]
        njt_t = ta_in.read(t)

        def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
            """Return the scalar log posterior for market-t njt_t_val."""
            # Likelihood for a single market uses only market-t objects.
            ll = posterior.market_loglik(
                qjt_t=qjt_t,
                q0t_t=q0t_t,
                pjt_t=pjt_t,
                wjt_t=wjt_t,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar_t,
                njt_t=njt_t_val,
            )

            # Market prior includes (E_bar_t, njt_t, gamma_t, phi_t).
            # logprior_market_vec is vectorized over markets; wrap market t as size-1.
            lp_1 = posterior.logprior_market_vec(
                E_bar=tf.reshape(E_bar_t, (1,)),
                njt=tf.expand_dims(njt_t_val, axis=0),
                gamma=tf.expand_dims(gamma_t, axis=0),
                phi=tf.reshape(phi_t, (1,)),
            )
            return ll + lp_1[0]

        # TMH proposes a joint move for the J-vector njt_t.
        njt_new_t, accepted = tmh_step(
            theta0=njt_t,
            logp_fn=logp_njt_t,
            ridge=ridge,
            rng=rng,
            k=k_njt,
        )

        # Write updated market vector back and update acceptance count.
        ta_out = ta_in.write(t, njt_new_t)
        acc_sum = acc_sum + tf.cast(accepted, tf.float64)
        return t + 1, ta_out, acc_sum

    # Initialize loop state: start at t=0 with zero accepted updates.
    t0 = tf.constant(0, tf.int32)
    acc0 = tf.constant(0.0, tf.float64)

    # Sequential market loop.
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

    gamma_{t,j} is the sparsity indicator selecting spike vs slab variance for n_{t,j}.
    Conditional on njt and phi, the gamma updates are independent across products,
    so this is applied in a fully vectorized way across (T, J).

    Returns:
        gamma_new: Updated gamma, shape (T, J), float64 in {0,1}.
    """
    # Broadcast phi across products so the kernel can operate on the full matrix.
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

    phi_t is the market-level inclusion rate:
      gamma_{t,j} | phi_t ~ Bernoulli(phi_t)
      phi_t ~ Beta(a_phi, b_phi)

    Conditional on gamma, each phi_t has a conjugate Beta posterior, so the full
    vector phi is sampled in one call.

    Returns:
        phi_new: Updated phi, shape (T,).
    """
    return gibbs_phi(
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        rng=rng,
    )
