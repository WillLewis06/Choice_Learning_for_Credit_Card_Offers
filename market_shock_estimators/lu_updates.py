from __future__ import annotations

import tensorflow as tf

from market_shock_estimators.lu_kernels import (
    rw_mh_step,
    tmh_step,
    gibbs_gamma,
    gibbs_phi,
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
    """
    One TMH step for (beta_p, beta_w), with r fixed.
    Returns (beta_p_new, beta_w_new, accepted).
    """
    beta0 = tf.stack([beta_p, beta_w], axis=0)

    def logp_beta(theta_vec: tf.Tensor) -> tf.Tensor:
        bp = theta_vec[0]
        bw = theta_vec[1]
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
    """
    One RW-MH step for scalar r, with (beta_p, beta_w) fixed.
    Returns (r_new, accepted).
    """

    def logp_r(r_val: tf.Tensor) -> tf.Tensor:
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
    """
    One RW-MH step for E_bar vector (T,), batched across markets.
    logp_fn returns (T,) via posterior.logpost_vec, so acceptance is elementwise.
    Returns (E_bar_new, accepted_vec).
    """

    def logp_E_bar_vec(E_bar_val: tf.Tensor) -> tf.Tensor:
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
    """
    One sweep over markets updating each njt_t (J,) via TMH (sequential market loop).
    Returns (njt_new, acc_sum) where acc_sum is a float64 scalar count of accepted markets.
    """

    T_t = tf.shape(njt)[0]
    ta_n = tf.TensorArray(tf.float64, size=T_t).unstack(njt)

    def cond(t, ta_in, acc_sum):
        return t < T_t

    def body(t, ta_in, acc_sum):
        qjt_t = qjt[t]
        q0t_t = q0t[t]
        pjt_t = pjt[t]
        wjt_t = wjt[t]

        E_bar_t = E_bar[t]
        gamma_t = gamma[t]
        phi_t = phi[t]
        njt_t = ta_in.read(t)

        def logp_njt_t(njt_t_val: tf.Tensor) -> tf.Tensor:
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
    """
    Batched Gibbs update for gamma (T,J) given njt and phi.
    Returns gamma_new (float64 0/1).
    """
    gamma_new = gibbs_gamma(
        njt_t=njt,
        phi_t=phi[:, None],
        T0_sq=posterior.T0_sq,
        T1_sq=posterior.T1_sq,
        log_T0_sq=posterior.log_T0_sq,
        log_T1_sq=posterior.log_T1_sq,
        rng=rng,
    )
    return gamma_new


@tf.function(reduce_retracing=True)
def update_phi(
    posterior,
    rng: tf.random.Generator,
    gamma: tf.Tensor,
):
    """
    Batched Gibbs update for phi (T,) given gamma (T,J).
    Returns phi_new.
    """
    phi_new = gibbs_phi(
        gamma=gamma,
        a_phi=posterior.a_phi,
        b_phi=posterior.b_phi,
        rng=rng,
    )
    return phi_new
