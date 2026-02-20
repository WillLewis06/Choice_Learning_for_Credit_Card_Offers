"""
Block-wise MCMC updates for the Lu shrinkage estimator.

Each function updates one parameter block conditional on the rest:
- TMH for dense blocks with strong local curvature,
- RW-MH for scalar/vector random-walk proposals,
- Gibbs for conjugate conditional distributions.

All tensors are expected to be tf.float64. Input validation is handled upstream
(on external configs); these update functions assume shapes and dtypes are
already consistent.
"""

from __future__ import annotations

import tensorflow as tf

from toolbox.mcmc_kernels import gibbs_gamma, gibbs_phi, rw_mh_step, tmh_step


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
    """TMH update for (beta_p, beta_w).

    Returns:
        beta_p_new: Scalar tf.float64.
        beta_w_new: Scalar tf.float64.
        accepted: Scalar boolean.
    """
    # Pack the two scalars into a length-2 vector for a joint proposal.
    beta0 = tf.stack([beta_p, beta_w], axis=0)

    def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
        # Unpack proposal.
        bp = theta_vec[0]
        bw = theta_vec[1]

        # Conditional log-likelihood: sum over markets.
        ll = tf.reduce_sum(
            posterior.loglik_vec(
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
        )

        # Global prior for (beta_p, beta_w, r); r is held fixed in this block update.
        lp = posterior.logprior_global(beta_p=bp, beta_w=bw, r=r)
        return ll + lp

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

    Returns:
        r_new: Scalar tf.float64.
        accepted: Scalar boolean.
    """

    def logp_r(r_val: tf.Tensor) -> tf.Tensor:
        ll = tf.reduce_sum(
            posterior.loglik_vec(
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
        )
        lp = posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r_val)
        return ll + lp

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
    """Elementwise RW-MH update for E_bar (shape (T,)).

    Returns:
        E_bar_new: tf.float64 tensor, shape (T,).
        accepted: Boolean tensor, shape (T,).
    """

    def logp_E_bar_vec(E_bar_val: tf.Tensor) -> tf.Tensor:
        # posterior.logpost_vec returns per-market log posterior contributions.
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
        theta0=E_bar,
        logp_fn=logp_E_bar_vec,
        k=k_E_bar,
        rng=rng,
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
    """Sequential TMH sweep updating njt market-by-market.

    The conditional density of njt given the rest depends on gamma (spike/slab
    selection) but not on phi. `phi` is kept in the signature for call-site
    uniformity.

    Returns:
        njt_new: tf.float64 tensor, shape (T, J).
        accepted_count: tf.float64 scalar count of accepted market updates.
    """
    # Constants for the Normal prior terms on n_{t,j}.
    two_pi = tf.constant(6.283185307179586, tf.float64)
    log_two_pi = tf.math.log(two_pi)
    log_T0_sq = tf.math.log(posterior.T0_sq)
    log_T1_sq = tf.math.log(posterior.T1_sq)

    T_t = tf.shape(njt)[0]
    ta_n = tf.TensorArray(tf.float64, size=T_t).unstack(njt)

    def cond(t, ta_in, accepted_count):
        return t < T_t

    def body(t, ta_in, accepted_count):
        # Market-specific observed data.
        qjt_t = qjt[t]
        q0t_t = q0t[t]
        pjt_t = pjt[t]
        wjt_t = wjt[t]

        # Market-specific latent state.
        E_bar_t = E_bar[t]
        gamma_t = gamma[t]
        njt_t = ta_in.read(t)

        def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
            # Market-t likelihood contribution.
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

            # Conditional prior for njt_t given gamma_t (independent across products).
            var = gamma_t * posterior.T1_sq + (1.0 - gamma_t) * posterior.T0_sq
            log_var = gamma_t * log_T1_sq + (1.0 - gamma_t) * log_T0_sq
            lp_n = tf.reduce_sum(
                -0.5 * (log_two_pi + log_var + tf.square(njt_t_val) / var)
            )

            return ll + lp_n

        # One TMH update for the J-dimensional vector njt_t.
        njt_new_t, accepted = tmh_step(
            theta0=njt_t,
            logp_fn=logp_njt_t,
            ridge=ridge,
            rng=rng,
            k=k_njt,
        )

        ta_out = ta_in.write(t, njt_new_t)
        accepted_count = accepted_count + tf.cast(accepted, tf.float64)
        return t + 1, ta_out, accepted_count

    t0 = tf.constant(0, tf.int32)
    accepted0 = tf.constant(0.0, tf.float64)

    _, ta_out, accepted_count = tf.while_loop(
        cond,
        body,
        loop_vars=(t0, ta_n, accepted0),
        parallel_iterations=1,
    )

    return ta_out.stack(), accepted_count


@tf.function(reduce_retracing=True)
def update_gamma(
    posterior,
    rng: tf.random.Generator,
    njt: tf.Tensor,
    phi: tf.Tensor,
):
    """Gibbs update for gamma (shape (T, J)) given njt and phi."""
    log_T0_sq = tf.math.log(posterior.T0_sq)
    log_T1_sq = tf.math.log(posterior.T1_sq)

    return gibbs_gamma(
        njt_t=njt,
        phi_t=phi[:, None],
        T0_sq=posterior.T0_sq,
        T1_sq=posterior.T1_sq,
        log_T0_sq=log_T0_sq,
        log_T1_sq=log_T1_sq,
        rng=rng,
    )


@tf.function(reduce_retracing=True)
def update_phi(
    posterior,
    rng: tf.random.Generator,
    gamma: tf.Tensor,
):
    """Gibbs update for phi (shape (T,)) given gamma."""
    return gibbs_phi(
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        rng=rng,
    )
