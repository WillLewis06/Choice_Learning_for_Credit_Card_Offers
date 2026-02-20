"""
cl_posterior.py

TensorFlow posterior components for the choice-learn + Lu sparse market-shock model.

Systematic utility (market t, product j):
  delta[t, j] = alpha * delta_cl[t, j] + E_bar[t] + njt[t, j]

Sparse shocks (elementwise over (t, j)):
  gamma[t, j] | phi[t] ~ Bernoulli(phi[t])
  phi[t] ~ Beta(a_phi, b_phi)
  njt[t, j] | gamma[t, j] = 1 ~ Normal(0, T1_sq)   (slab)
  njt[t, j] | gamma[t, j] = 0 ~ Normal(0, T0_sq)   (spike)

Likelihood is multinomial logit with an outside option (utility normalized to 0).
The multinomial combinatorial constant is omitted (cancels in MH ratios).
"""

from __future__ import annotations

from collections.abc import Mapping

import tensorflow as tf


class LuPosteriorTF:
    """Likelihood and prior terms used by the shrinkage sampler (float64 only)."""

    _LOG_2PI = tf.constant(1.8378770664093453, tf.float64)  # log(2*pi)
    _PHI_EPS = tf.constant(1e-12, tf.float64)  # numerical guard for log(phi)

    def __init__(self, config: Mapping[str, float]):
        """Construct priors from a fully-specified hyperparameter config.

        Required keys:
          - alpha_mean, alpha_var
          - E_bar_mean, E_bar_var
          - T0_sq, T1_sq
          - a_phi, b_phi
        """
        # Global prior: alpha ~ Normal(alpha_mean, alpha_var)
        self.alpha_mean = tf.constant(config["alpha_mean"], tf.float64)
        self.alpha_var = tf.constant(config["alpha_var"], tf.float64)
        self._alpha_prec = 1.0 / self.alpha_var
        self._lp_alpha_const = -0.5 * (self._LOG_2PI + tf.math.log(self.alpha_var))

        # Market prior: E_bar[t] ~ Normal(E_bar_mean, E_bar_var), iid over t
        self.E_bar_mean = tf.constant(config["E_bar_mean"], tf.float64)
        self.E_bar_var = tf.constant(config["E_bar_var"], tf.float64)
        self._E_bar_prec = 1.0 / self.E_bar_var
        self._lp_E_bar_const = -0.5 * (self._LOG_2PI + tf.math.log(self.E_bar_var))

        # Spike-and-slab variances for njt
        self.T0_sq = tf.constant(config["T0_sq"], tf.float64)
        self.T1_sq = tf.constant(config["T1_sq"], tf.float64)
        self._log_T0_sq = tf.math.log(self.T0_sq)
        self._log_T1_sq = tf.math.log(self.T1_sq)

        # Beta prior: phi[t] ~ Beta(a_phi, b_phi), iid over t
        self.a_phi = tf.constant(config["a_phi"], tf.float64)
        self.b_phi = tf.constant(config["b_phi"], tf.float64)
        self._logB_phi = (
            tf.math.lgamma(self.a_phi)
            + tf.math.lgamma(self.b_phi)
            - tf.math.lgamma(self.a_phi + self.b_phi)
        )

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _mean_utility(
        self,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        n: tf.Tensor,
    ) -> tf.Tensor:
        """Compute mean utility for inside goods (broadcasts E_bar across products)."""
        return alpha * delta_cl + E_bar[..., None] + n

    def _log_choice_probs(self, delta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return (log_pj, log_p0) under logit with outside option."""
        zeros = tf.zeros_like(delta[..., :1])  # outside utility normalized to 0
        delta_aug = tf.concat([zeros, delta], axis=-1)
        log_denom = tf.math.reduce_logsumexp(delta_aug, axis=-1)
        log_p0 = -log_denom
        log_pj = delta - log_denom[..., None]
        return log_pj, log_p0

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def market_loglik(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Log-likelihood for one market t (scalar)."""
        delta = self._mean_utility(
            delta_cl=delta_cl_t, alpha=alpha, E_bar=E_bar_t, n=njt_t
        )
        log_pj, log_p0 = self._log_choice_probs(delta)
        return tf.reduce_sum(qjt_t * log_pj) + q0t_t * log_p0

    @tf.function(reduce_retracing=True)
    def loglik_vec(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market log-likelihood vector (shape (T,))."""
        delta = self._mean_utility(delta_cl=delta_cl, alpha=alpha, E_bar=E_bar, n=njt)
        log_pj, log_p0 = self._log_choice_probs(delta)
        return tf.reduce_sum(qjt * log_pj, axis=1) + q0t * log_p0

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def logprior_global(self, alpha: tf.Tensor) -> tf.Tensor:
        """Log prior for alpha (scalar)."""
        quad = -0.5 * tf.square(alpha - self.alpha_mean) * self._alpha_prec
        return self._lp_alpha_const + quad

    def logprior_market_vec(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market log prior vector for (E_bar, njt, gamma, phi)."""

        # E_bar prior: Normal, iid over markets
        lp_E = (
            self._lp_E_bar_const
            - 0.5 * tf.square(E_bar - self.E_bar_mean) * self._E_bar_prec
        )

        # njt prior: Normal with variance selected by gamma (spike/slab)
        var = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq
        log_var = gamma * self._log_T1_sq + (1.0 - gamma) * self._log_T0_sq
        lp_n_entry = -0.5 * (self._LOG_2PI + log_var + tf.square(njt) / var)
        lp_n = tf.reduce_sum(lp_n_entry, axis=1)

        # gamma prior: Bernoulli(phi[t]) iid over products conditional on phi[t]
        # Numerical guard against log(0) under finite precision.
        phi_safe = tf.clip_by_value(phi, self._PHI_EPS, 1.0 - self._PHI_EPS)
        log_phi = tf.math.log(phi_safe)[:, None]
        log_1mphi = tf.math.log(1.0 - phi_safe)[:, None]
        lp_g = tf.reduce_sum(gamma * log_phi + (1.0 - gamma) * log_1mphi, axis=1)

        # phi prior: Beta(a_phi, b_phi) iid over markets
        lp_phi = (
            (self.a_phi - 1.0) * tf.math.log(phi_safe)
            + (self.b_phi - 1.0) * tf.math.log(1.0 - phi_safe)
            - self._logB_phi
        )

        return lp_E + lp_n + lp_g + lp_phi

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def logpost_vec(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market log posterior vector excluding the alpha prior (shape (T,))."""
        ll_t = self.loglik_vec(
            qjt=qjt, q0t=q0t, delta_cl=delta_cl, alpha=alpha, E_bar=E_bar, njt=njt
        )
        lp_t = self.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)
        return ll_t + lp_t

    def logpost(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        alpha: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Full scalar log posterior including the alpha prior."""
        return tf.reduce_sum(
            self.logpost_vec(
                qjt=qjt,
                q0t=q0t,
                delta_cl=delta_cl,
                alpha=alpha,
                E_bar=E_bar,
                njt=njt,
                gamma=gamma,
                phi=phi,
            )
        ) + self.logprior_global(alpha)
