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
        self._log_T0_sq = tf.math.log(self.T0_sq)
        self._log_T1_sq = tf.math.log(self.T1_sq)

        self._log_beta_ab = (
            tf.math.lgamma(self.a_phi)
            + tf.math.lgamma(self.b_phi)
            - tf.math.lgamma(self.a_phi + self.b_phi)
        )

        self._lp0_beta_p = -0.5 * (self._log_two_pi + tf.math.log(self.beta_p_var))
        self._lp0_beta_w = -0.5 * (self._log_two_pi + tf.math.log(self.beta_w_var))
        self._lp0_r = -0.5 * (self._log_two_pi + tf.math.log(self.r_var))
        self._lp0_E_bar = -0.5 * (self._log_two_pi + tf.math.log(self.E_bar_var))

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

    def _market_loglik(
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
        T = tf.shape(pjt)[0]

        def _market_ll(t: tf.Tensor) -> tf.Tensor:
            return self._market_loglik(
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

        return tf.reduce_sum(
            tf.map_fn(_market_ll, tf.range(T), fn_output_signature=self.dtype)
        )

    def global_prior(
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

    def continuous_prior(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        lp_E = (
            self._lp0_E_bar - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )

        one_minus_gamma = 1.0 - gamma
        var = gamma * self.T1_sq + one_minus_gamma * self.T0_sq
        log_var = gamma * self._log_T1_sq + one_minus_gamma * self._log_T0_sq

        lp_n = -0.5 * (self._log_two_pi + log_var + tf.square(njt) / var)

        return tf.reduce_sum(lp_E) + tf.reduce_sum(lp_n)

    def full_prior(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        lp_cont = self.continuous_prior(
            E_bar=E_bar,
            njt=njt,
            gamma=gamma,
        )

        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)
        one_minus_gamma = 1.0 - gamma

        lp_g = gamma * tf.math.log(phi[:, None]) + one_minus_gamma * tf.math.log(
            1.0 - phi[:, None]
        )
        lp_phi = (
            (self.a_phi - 1.0) * tf.math.log(phi)
            + (self.b_phi - 1.0) * tf.math.log(1.0 - phi)
            - self._log_beta_ab
        )

        return lp_cont + tf.reduce_sum(lp_g) + tf.reduce_sum(lp_phi)

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
            + self.global_prior(beta_p=beta_p, beta_w=beta_w, r=r)
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
        phi: tf.Tensor,
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
            + self.global_prior(beta_p=beta_p, beta_w=beta_w, r=r)
            + self.full_prior(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)
        )
