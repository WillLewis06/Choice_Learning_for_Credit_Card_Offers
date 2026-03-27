"""Posterior evaluation for the choice-learn shrinkage model."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ChoiceLearnPosteriorConfig:
    """Store fixed hyperparameters and numerical settings for posterior evaluation."""

    eps: float

    alpha_mean: float
    alpha_var: float

    E_bar_mean: float
    E_bar_var: float

    T0_sq: float
    T1_sq: float

    a_phi: float
    b_phi: float


class ChoiceLearnPosteriorTF:
    """Evaluate posterior terms used by the choice-learn shrinkage sampler."""

    def __init__(self, config: ChoiceLearnPosteriorConfig):
        """Cache fixed constants and prior terms used across posterior evaluations."""

        self.eps = tf.constant(config.eps, dtype=tf.float64)

        self.alpha_mean = tf.constant(config.alpha_mean, dtype=tf.float64)
        self.alpha_var = tf.constant(config.alpha_var, dtype=tf.float64)

        self.E_bar_mean = tf.constant(config.E_bar_mean, dtype=tf.float64)
        self.E_bar_var = tf.constant(config.E_bar_var, dtype=tf.float64)

        self.T0_sq = tf.constant(config.T0_sq, dtype=tf.float64)
        self.T1_sq = tf.constant(config.T1_sq, dtype=tf.float64)

        self.a_phi = tf.constant(config.a_phi, dtype=tf.float64)
        self.b_phi = tf.constant(config.b_phi, dtype=tf.float64)

        self._log_two_pi = tf.math.log(
            tf.constant(2.0 * 3.141592653589793, dtype=tf.float64)
        )
        self._log_beta_ab = (
            tf.math.lgamma(self.a_phi)
            + tf.math.lgamma(self.b_phi)
            - tf.math.lgamma(self.a_phi + self.b_phi)
        )

        self._log_alpha_var = tf.math.log(self.alpha_var)
        self._log_E_bar_var = tf.math.log(self.E_bar_var)
        self._log_T0_sq = tf.math.log(self.T0_sq)
        self._log_T1_sq = tf.math.log(self.T1_sq)

        self._lp0_alpha = -0.5 * (self._log_two_pi + self._log_alpha_var)
        self._lp0_E_bar = -0.5 * (self._log_two_pi + self._log_E_bar_var)

    def _mean_utility_t(
        self,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Return the mean utility vector for one market."""

        return alpha * delta_cl_t + E_bar_t + njt_t

    def _log_choice_probs_t(
        self,
        delta_t: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute inside-good and outside-good log probabilities for one market."""

        zeros = tf.zeros((1,), dtype=tf.float64)
        delta_aug = tf.concat([zeros, delta_t], axis=0)
        log_denom = tf.reduce_logsumexp(delta_aug)
        log_sjt_t = delta_t - log_denom
        log_s0t = -log_denom
        return log_sjt_t, log_s0t

    def _market_loglik_impl(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Return the one-market log-likelihood contribution."""

        delta_t = self._mean_utility_t(
            delta_cl_t=delta_cl_t,
            alpha=alpha,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        log_sjt_t, log_s0t = self._log_choice_probs_t(delta_t=delta_t)

        sjt_t = tf.clip_by_value(tf.exp(log_sjt_t), self.eps, 1.0)
        s0t = tf.clip_by_value(tf.exp(log_s0t), self.eps, 1.0)

        return tf.reduce_sum(qjt_t * tf.math.log(sjt_t)) + q0t_t * tf.math.log(s0t)

    def _logprior_E_bar_t(self, E_bar_t: tf.Tensor) -> tf.Tensor:
        """Return the Gaussian log prior for one market-level common shock."""

        return (
            self._lp0_E_bar
            - 0.5 * tf.square(E_bar_t - self.E_bar_mean) / self.E_bar_var
        )

    def _logprior_njt_t_given_gamma_t(
        self,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
    ) -> tf.Tensor:
        """Return the conditional log prior for one market's product shocks."""

        one_minus_gamma_t = 1.0 - gamma_t
        var_t = gamma_t * self.T1_sq + one_minus_gamma_t * self.T0_sq
        log_var_t = gamma_t * self._log_T1_sq + one_minus_gamma_t * self._log_T0_sq
        lp_n_t = -0.5 * (self._log_two_pi + log_var_t + tf.square(njt_t) / var_t)
        return tf.reduce_sum(lp_n_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total log-likelihood across markets."""

        T = tf.shape(delta_cl)[0]

        def _market_ll(t: tf.Tensor) -> tf.Tensor:
            return self._market_loglik_impl(
                qjt_t=qjt[t],
                q0t_t=q0t[t],
                delta_cl_t=delta_cl[t],
                alpha=alpha,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )

        ll_vec = tf.map_fn(_market_ll, tf.range(T), fn_output_signature=tf.float64)
        return tf.reduce_sum(ll_vec)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_global(
        self,
        alpha: tf.Tensor,
    ) -> tf.Tensor:
        """Return the joint prior for the global coefficient."""

        return (
            self._lp0_alpha - 0.5 * tf.square(alpha - self.alpha_mean) / self.alpha_var
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_E_bar_vec(
        self,
        E_bar: tf.Tensor,
    ) -> tf.Tensor:
        """Return the vector of marketwise prior terms for E_bar."""

        return (
            self._lp0_E_bar - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_njt_given_gamma_vec(
        self,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return the vector of marketwise conditional prior terms for njt."""

        one_minus_gamma = 1.0 - gamma
        var = gamma * self.T1_sq + one_minus_gamma * self.T0_sq
        log_var = gamma * self._log_T1_sq + one_minus_gamma * self._log_T0_sq
        lp_n = -0.5 * (self._log_two_pi + log_var + tf.square(njt) / var)
        return tf.reduce_sum(lp_n, axis=1)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def continuous_prior(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total prior contribution from E_bar and njt."""

        return tf.reduce_sum(
            self.logprior_E_bar_vec(E_bar=E_bar)
            + self.logprior_njt_given_gamma_vec(njt=njt, gamma=gamma)
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def collapsed_gamma_prior(
        self,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return the collapsed Beta-binomial prior contribution for gamma."""

        s = tf.reduce_sum(gamma, axis=1)
        J = tf.cast(tf.shape(gamma)[1], tf.float64)

        log_beta_post = (
            tf.math.lgamma(self.a_phi + s)
            + tf.math.lgamma(self.b_phi + (J - s))
            - tf.math.lgamma(self.a_phi + self.b_phi + J)
        )
        return tf.reduce_sum(log_beta_post - self._log_beta_ab)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def alpha_block_logpost(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for alpha."""

        return self.loglik(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
        ) + self.logprior_global(alpha=alpha)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def E_bar_block_logpost(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Return the one-market block log posterior for E_bar_t."""

        return self._market_loglik_impl(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            delta_cl_t=delta_cl_t,
            alpha=alpha,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        ) + self._logprior_E_bar_t(E_bar_t=E_bar_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def njt_block_logpost(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
        gamma_t: tf.Tensor,
    ) -> tf.Tensor:
        """Return the one-market block log posterior for njt_t."""

        return self._market_loglik_impl(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            delta_cl_t=delta_cl_t,
            alpha=alpha,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        ) + self._logprior_njt_t_given_gamma_t(njt_t=njt_t, gamma_t=gamma_t)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def joint_logpost(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return the full joint log posterior."""

        return (
            self.loglik(
                qjt=qjt,
                q0t=q0t,
                delta_cl=delta_cl,
                alpha=alpha,
                E_bar=E_bar,
                njt=njt,
            )
            + self.logprior_global(alpha=alpha)
            + self.continuous_prior(E_bar=E_bar, njt=njt, gamma=gamma)
            + self.collapsed_gamma_prior(gamma=gamma)
        )
