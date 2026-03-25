from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class LuPosteriorConfig:
    n_draws: int
    seed: int

    dtype: tf.dtypes.DType
    eps: float

    beta_p_mean: float
    beta_p_var: float
    beta_w_mean: float
    beta_w_var: float
    r_mean: float
    r_var: float

    E_bar_mean: float
    E_bar_var: float

    T0_sq: float
    T1_sq: float

    a_phi: float
    b_phi: float


class LuPosteriorTF:
    def __init__(self, config: LuPosteriorConfig):
        self.dtype = config.dtype
        self.eps = tf.constant(config.eps, dtype=self.dtype)

        self.v_draws = tf.random.stateless_normal(
            shape=(config.n_draws,),
            seed=tf.constant([config.seed, 0], dtype=tf.int32),
            dtype=self.dtype,
        )

        self.beta_p_mean = tf.constant(config.beta_p_mean, dtype=self.dtype)
        self.beta_p_var = tf.constant(config.beta_p_var, dtype=self.dtype)

        self.beta_w_mean = tf.constant(config.beta_w_mean, dtype=self.dtype)
        self.beta_w_var = tf.constant(config.beta_w_var, dtype=self.dtype)

        self.r_mean = tf.constant(config.r_mean, dtype=self.dtype)
        self.r_var = tf.constant(config.r_var, dtype=self.dtype)

        self.E_bar_mean = tf.constant(config.E_bar_mean, dtype=self.dtype)
        self.E_bar_var = tf.constant(config.E_bar_var, dtype=self.dtype)

        self.T0_sq = tf.constant(config.T0_sq, dtype=self.dtype)
        self.T1_sq = tf.constant(config.T1_sq, dtype=self.dtype)

        self.a_phi = tf.constant(config.a_phi, dtype=self.dtype)
        self.b_phi = tf.constant(config.b_phi, dtype=self.dtype)

        self._log_two_pi = tf.math.log(
            tf.constant(2.0 * 3.141592653589793, dtype=self.dtype)
        )
        self._log_beta_ab = (
            tf.math.lgamma(self.a_phi)
            + tf.math.lgamma(self.b_phi)
            - tf.math.lgamma(self.a_phi + self.b_phi)
        )

        self._log_beta_p_var = tf.math.log(self.beta_p_var)
        self._log_beta_w_var = tf.math.log(self.beta_w_var)
        self._log_r_var = tf.math.log(self.r_var)
        self._log_E_bar_var = tf.math.log(self.E_bar_var)
        self._log_T0_sq = tf.math.log(self.T0_sq)
        self._log_T1_sq = tf.math.log(self.T1_sq)

        self._lp0_beta_p = -0.5 * (self._log_two_pi + self._log_beta_p_var)
        self._lp0_beta_w = -0.5 * (self._log_two_pi + self._log_beta_w_var)
        self._lp0_r = -0.5 * (self._log_two_pi + self._log_r_var)
        self._lp0_E_bar = -0.5 * (self._log_two_pi + self._log_E_bar_var)

    def _gaussian_logpdf(
        self,
        x: tf.Tensor,
        mean: tf.Tensor,
        var: tf.Tensor,
        log_var: tf.Tensor,
    ) -> tf.Tensor:
        return -0.5 * (self._log_two_pi + log_var + tf.square(x - mean) / var)

    def _mean_utility_jt(
        self,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        return beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t

    def _choice_probs_t(
        self,
        pjt_t: tf.Tensor,
        delta_t: tf.Tensor,
        r: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        sigma = tf.exp(r)
        util = delta_t[:, None] + pjt_t[:, None] * (sigma * self.v_draws)[None, :]

        m = tf.reduce_max(util, axis=0, keepdims=True)
        m = tf.maximum(m, tf.zeros_like(m))

        expu = tf.exp(util - m)
        exp0 = tf.exp(-m)
        denom = exp0 + tf.reduce_sum(expu, axis=0, keepdims=True)

        sjt_t = tf.reduce_mean(expu / denom, axis=1)
        s0t = tf.reduce_mean(exp0[0] / denom[0])
        return sjt_t, s0t

    def _market_loglik_impl(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        delta_t = self._mean_utility_jt(
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        sjt_t, s0t = self._choice_probs_t(
            pjt_t=pjt_t,
            delta_t=delta_t,
            r=r,
        )

        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        return tf.reduce_sum(qjt_t * tf.math.log(sjt_t)) + q0t_t * tf.math.log(s0t)

    def _logprior_E_bar_t(self, E_bar_t: tf.Tensor) -> tf.Tensor:
        return (
            self._lp0_E_bar
            - 0.5 * tf.square(E_bar_t - self.E_bar_mean) / self.E_bar_var
        )

    def _logprior_njt_t_given_gamma_t(
        self,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
    ) -> tf.Tensor:
        gamma_t = tf.cast(gamma_t, self.dtype)
        one_minus_gamma_t = 1.0 - gamma_t
        var_t = gamma_t * self.T1_sq + one_minus_gamma_t * self.T0_sq
        log_var_t = gamma_t * self._log_T1_sq + one_minus_gamma_t * self._log_T0_sq
        lp_n_t = -0.5 * (self._log_two_pi + log_var_t + tf.square(njt_t) / var_t)
        return tf.reduce_sum(lp_n_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def market_loglik(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        return self._market_loglik_impl(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik_vec(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        T = tf.shape(pjt)[0]

        def _market_ll(t: tf.Tensor) -> tf.Tensor:
            return self._market_loglik_impl(
                qjt_t=qjt[t],
                q0t_t=q0t[t],
                pjt_t=pjt[t],
                wjt_t=wjt[t],
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )

        return tf.map_fn(_market_ll, tf.range(T), fn_output_signature=self.dtype)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        return tf.reduce_sum(
            self.loglik_vec(
                qjt=qjt,
                q0t=q0t,
                pjt=pjt,
                wjt=wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar=E_bar,
                njt=njt,
            )
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_global(
        self,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
    ) -> tf.Tensor:
        lp_beta_p = (
            self._lp0_beta_p
            - 0.5 * tf.square(beta_p - self.beta_p_mean) / self.beta_p_var
        )
        lp_beta_w = (
            self._lp0_beta_w
            - 0.5 * tf.square(beta_w - self.beta_w_mean) / self.beta_w_var
        )
        lp_r = self._lp0_r - 0.5 * tf.square(r - self.r_mean) / self.r_var
        return lp_beta_p + lp_beta_w + lp_r

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_E_bar_vec(
        self,
        E_bar: tf.Tensor,
    ) -> tf.Tensor:
        return (
            self._lp0_E_bar - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_E_bar(
        self,
        E_bar: tf.Tensor,
    ) -> tf.Tensor:
        return tf.reduce_sum(self.logprior_E_bar_vec(E_bar=E_bar))

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_njt_given_gamma_vec(
        self,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        gamma = tf.cast(gamma, self.dtype)
        one_minus_gamma = 1.0 - gamma
        var = gamma * self.T1_sq + one_minus_gamma * self.T0_sq
        log_var = gamma * self._log_T1_sq + one_minus_gamma * self._log_T0_sq
        lp_n = -0.5 * (self._log_two_pi + log_var + tf.square(njt) / var)
        return tf.reduce_sum(lp_n, axis=1)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_njt_given_gamma(
        self,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        return tf.reduce_sum(self.logprior_njt_given_gamma_vec(njt=njt, gamma=gamma))

    @tf.function(jit_compile=True, reduce_retracing=True)
    def continuous_prior_vec(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        return self.logprior_E_bar_vec(E_bar=E_bar) + self.logprior_njt_given_gamma_vec(
            njt=njt, gamma=gamma
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def continuous_prior(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        return tf.reduce_sum(
            self.continuous_prior_vec(E_bar=E_bar, njt=njt, gamma=gamma)
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def collapsed_gamma_prior_vec(
        self,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        gamma = tf.cast(gamma, self.dtype)
        s = tf.reduce_sum(gamma, axis=1)
        J = tf.cast(tf.shape(gamma)[1], self.dtype)

        log_beta_post = (
            tf.math.lgamma(self.a_phi + s)
            + tf.math.lgamma(self.b_phi + (J - s))
            - tf.math.lgamma(self.a_phi + self.b_phi + J)
        )
        return log_beta_post - self._log_beta_ab

    @tf.function(jit_compile=True, reduce_retracing=True)
    def collapsed_gamma_prior(
        self,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        return tf.reduce_sum(self.collapsed_gamma_prior_vec(gamma=gamma))

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_block_logpost(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        ll = self.loglik(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar=E_bar,
            njt=njt,
        )
        lp_beta_p = (
            self._lp0_beta_p
            - 0.5 * tf.square(beta_p - self.beta_p_mean) / self.beta_p_var
        )
        lp_beta_w = (
            self._lp0_beta_w
            - 0.5 * tf.square(beta_w - self.beta_w_mean) / self.beta_w_var
        )
        return ll + lp_beta_p + lp_beta_w

    @tf.function(jit_compile=True, reduce_retracing=True)
    def r_block_logpost(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        ll = self.loglik(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar=E_bar,
            njt=njt,
        )
        lp_r = self._lp0_r - 0.5 * tf.square(r - self.r_mean) / self.r_var
        return ll + lp_r

    @tf.function(jit_compile=True, reduce_retracing=True)
    def E_bar_block_logpost(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        return self.market_loglik(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        ) + self._logprior_E_bar_t(E_bar_t=E_bar_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def njt_block_logpost(
        self,
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
    ) -> tf.Tensor:
        return self.market_loglik(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        ) + self._logprior_njt_t_given_gamma_t(njt_t=njt_t, gamma_t=gamma_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def conditional_continuous_logpost(
        self,
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
    ) -> tf.Tensor:
        return (
            self.loglik(
                qjt=qjt,
                q0t=q0t,
                pjt=pjt,
                wjt=wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar=E_bar,
                njt=njt,
            )
            + self.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)
            + self.continuous_prior(E_bar=E_bar, njt=njt, gamma=gamma)
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def joint_logpost(
        self,
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
    ) -> tf.Tensor:
        return (
            self.loglik(
                qjt=qjt,
                q0t=q0t,
                pjt=pjt,
                wjt=wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar=E_bar,
                njt=njt,
            )
            + self.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)
            + self.continuous_prior(E_bar=E_bar, njt=njt, gamma=gamma)
            + self.collapsed_gamma_prior(gamma=gamma)
        )
