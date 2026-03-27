"""Posterior evaluation for the choice-learn shrinkage model."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ChoiceLearnPosteriorConfig:
    """Store fixed prior hyperparameters for posterior evaluation."""

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

        self._alpha_log_normalizer = -0.5 * (self._log_two_pi + self._log_alpha_var)
        self._E_bar_log_normalizer = -0.5 * (self._log_two_pi + self._log_E_bar_var)

    def _utilities(
        self,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Return mean utilities for one market or a batch of markets."""

        return alpha * delta_cl + E_bar[..., None] + njt

    def _log_choice_probs(
        self,
        utilities: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return inside-good and outside-good log probabilities.

        The outside option is normalized to zero utility. This helper works for both
        one-market inputs of shape ``(J,)`` and batched inputs of shape ``(T, J)``.
        """

        zero_column = tf.zeros(
            tf.concat([tf.shape(utilities)[:-1], [1]], axis=0),
            dtype=tf.float64,
        )
        augmented_utilities = tf.concat([zero_column, utilities], axis=-1)
        log_denom = tf.reduce_logsumexp(augmented_utilities, axis=-1)

        log_inside_probs = utilities - log_denom[..., None]
        log_outside_prob = -log_denom
        return log_inside_probs, log_outside_prob

    def _log_likelihood_terms(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Return per-market log-likelihood contributions.

        For one-market inputs this returns a scalar.
        For batched inputs this returns a vector of length ``T``.
        """

        utilities = self._utilities(
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
        )
        log_inside_probs, log_outside_prob = self._log_choice_probs(utilities=utilities)

        # Stay in log space throughout. There is no need for exp/clip/log roundtrips.
        inside_term = tf.reduce_sum(qjt * log_inside_probs, axis=-1)
        outside_term = q0t * log_outside_prob
        return inside_term + outside_term

    def _alpha_logprior(
        self,
        alpha: tf.Tensor,
    ) -> tf.Tensor:
        """Return the Gaussian log prior for ``alpha``."""

        return (
            self._alpha_log_normalizer
            - 0.5 * tf.square(alpha - self.alpha_mean) / self.alpha_var
        )

    def _E_bar_logprior(
        self,
        E_bar: tf.Tensor,
    ) -> tf.Tensor:
        """Return Gaussian log prior contributions for ``E_bar``."""

        return (
            self._E_bar_log_normalizer
            - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )

    def _njt_logprior_given_gamma(
        self,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return conditional Gaussian log prior contributions for ``njt``.

        This returns one contribution per market when the inputs are batched and a
        scalar when the inputs correspond to a single market.
        """

        variance = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq
        log_variance = gamma * self._log_T1_sq + (1.0 - gamma) * self._log_T0_sq

        log_density = -0.5 * (
            self._log_two_pi + log_variance + tf.square(njt) / variance
        )
        return tf.reduce_sum(log_density, axis=-1)

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

        return tf.reduce_sum(
            self._log_likelihood_terms(
                qjt=qjt,
                q0t=q0t,
                delta_cl=delta_cl,
                alpha=alpha,
                E_bar=E_bar,
                njt=njt,
            )
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def alpha_logprior(
        self,
        alpha: tf.Tensor,
    ) -> tf.Tensor:
        """Return the Gaussian log prior for ``alpha``."""

        return self._alpha_logprior(alpha=alpha)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def E_bar_logprior(
        self,
        E_bar: tf.Tensor,
    ) -> tf.Tensor:
        """Return marketwise Gaussian log prior terms for ``E_bar``."""

        return self._E_bar_logprior(E_bar=E_bar)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def njt_logprior_given_gamma(
        self,
        njt: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return marketwise conditional Gaussian log prior terms for ``njt``."""

        return self._njt_logprior_given_gamma(njt=njt, gamma=gamma)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def collapsed_gamma_prior(
        self,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        """Return the collapsed Beta-binomial prior contribution for ``gamma``.

        This integrates out the market-level inclusion probability ``phi``.
        """

        active_counts = tf.reduce_sum(gamma, axis=1)
        num_products = tf.cast(tf.shape(gamma)[1], tf.float64)

        log_beta_post = (
            tf.math.lgamma(self.a_phi + active_counts)
            + tf.math.lgamma(self.b_phi + (num_products - active_counts))
            - tf.math.lgamma(self.a_phi + self.b_phi + num_products)
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
        """Return the block log posterior for ``alpha``."""

        return self.loglik(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
        ) + self._alpha_logprior(alpha=alpha)

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
        """Return the one-market block log posterior for ``E_bar_t``."""

        return self._log_likelihood_terms(
            qjt=qjt_t,
            q0t=q0t_t,
            delta_cl=delta_cl_t,
            alpha=alpha,
            E_bar=E_bar_t,
            njt=njt_t,
        ) + self._E_bar_logprior(E_bar=E_bar_t)

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
        """Return the one-market block log posterior for ``njt_t``."""

        return self._log_likelihood_terms(
            qjt=qjt_t,
            q0t=q0t_t,
            delta_cl=delta_cl_t,
            alpha=alpha,
            E_bar=E_bar_t,
            njt=njt_t,
        ) + self._njt_logprior_given_gamma(njt=njt_t, gamma=gamma_t)

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

        log_likelihood = self.loglik(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
        )

        # Keep the full posterior decomposition explicit in one place.
        alpha_prior = self._alpha_logprior(alpha=alpha)
        E_bar_prior = tf.reduce_sum(self._E_bar_logprior(E_bar=E_bar))
        njt_prior = tf.reduce_sum(self._njt_logprior_given_gamma(njt=njt, gamma=gamma))
        gamma_prior = self.collapsed_gamma_prior(gamma=gamma)

        return log_likelihood + alpha_prior + E_bar_prior + njt_prior + gamma_prior
