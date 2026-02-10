"""
TensorFlow posterior utilities for the choice-learn + Lu market-shocks estimator.

This file defines `LuPosteriorTF`, a TF-native collection of:
  - Choice probabilities for a standard logit with outside option,
  - Multinomial log-likelihood terms using observed counts (qjt, q0t),
  - Prior terms for the global scaling parameter alpha and the sparse shock structure,
  - Convenience functions that return either per-market vectors or full scalars.

Model (systematic utility):
  delta_{t,j} = alpha * delta_cl_{t,j} + E_bar[t] + njt[t,j]

Sparse market-product shocks use a spike-and-slab (continuous mixture):
  gamma | phi ~ Bernoulli(phi) elementwise
  phi ~ Beta(a_phi, b_phi) per market

  njt | gamma=1 ~ Normal(0, T1_sq)   (slab, large variance)
  njt | gamma=0 ~ Normal(0, T0_sq)   (spike, small variance)

Likelihood uses multinomial counts and drops the combinatorial constant:
  log p(q_t | s_t) = q0t * log(s0t) + sum_j qjt * log(sjt) + const

Interfaces are organized to match MCMC updates:
  - market_loglik: scalar likelihood contribution for one market
  - loglik_vec: length-T vector of per-market likelihood contributions
  - logprior_global: scalar prior for alpha
  - logprior_market_vec: length-T vector for (E_bar, njt, gamma, phi) priors
  - logpost_vec: length-T vector of (likelihood + market prior)
  - logpost: full scalar posterior (sum over markets + global prior)
"""

from __future__ import annotations

import tensorflow as tf


class LuPosteriorTF:
    """Posterior pieces for the choice-learn + Lu sparse-shock model (TF-only)."""

    def __init__(
        self,
        alpha_mean: float = 1.0,
        alpha_var: float = 1.0,
        E_bar_mean: float = 0.0,
        E_bar_var: float = 10.0,
        T0_sq: float = 1e-3,
        T1_sq: float = 1.0,
        a_phi: float = 1.0,
        b_phi: float = 1.0,
        eps: float = 1e-15,
        dtype=tf.float64,
    ):
        """Initialize hyperparameters."""
        self.dtype = dtype
        self.eps = tf.constant(eps, dtype=dtype)

        # Global prior for alpha (independent Normal).
        self.alpha_mean = tf.constant(alpha_mean, dtype=dtype)
        self.alpha_var = tf.constant(alpha_var, dtype=dtype)

        # Market-level prior for E_bar (independent Normal).
        self.E_bar_mean = tf.constant(E_bar_mean, dtype=dtype)
        self.E_bar_var = tf.constant(E_bar_var, dtype=dtype)

        # Spike-and-slab variances (and cached logs used repeatedly).
        self.T0_sq = tf.constant(T0_sq, dtype=dtype)
        self.T1_sq = tf.constant(T1_sq, dtype=dtype)
        self.log_T0_sq = tf.math.log(self.T0_sq)
        self.log_T1_sq = tf.math.log(self.T1_sq)

        # Beta prior parameters for market inclusion rates.
        self.a_phi = tf.constant(a_phi, dtype=dtype)
        self.b_phi = tf.constant(b_phi, dtype=dtype)

        self.two_pi = tf.constant(2.0 * 3.141592653589793, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _mean_utility_jt(
        self,
        delta_cl_t: tf.Tensor,
        alpha: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Compute delta_t for one market.

        For a single market t (vectors over products j=0..J-1):
          delta_t = alpha * delta_cl_t + E_bar_t + njt_t

        Returns:
            delta_t: Tensor of shape (J,).
        """
        return alpha * delta_cl_t + E_bar_t + njt_t

    def _choice_probs_t(self, delta_t: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute (sjt_t, s0t) for one market under standard logit with outside option.

        Outside option utility is normalized to 0. Uses a max-shift for numerical
        stability of exp().

        Args:
            delta_t: Mean utilities for market t, shape (J,).

        Returns:
            sjt_t: Inside shares for market t, shape (J,).
            s0t: Outside share for market t, scalar.
        """
        # Include outside option (0) in the max for a stable shift.
        m = tf.reduce_max(delta_t)
        m = tf.maximum(m, tf.zeros_like(m))

        expu = tf.exp(delta_t - m)  # (J,)
        exp0 = tf.exp(-m)  # scalar
        denom = exp0 + tf.reduce_sum(expu)  # scalar

        sjt_t = expu / denom  # (J,)
        s0t = exp0 / denom  # scalar
        return sjt_t, s0t

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
        """Compute one market's multinomial log-likelihood term.

        Returns:
          q0t_t * log(s0t) + sum_j qjt_t[j] * log(sjt_t[j])

        The multinomial combinatorial constant is omitted because it does not
        depend on parameters and cancels in MH ratios.
        """
        delta_t = self._mean_utility_jt(
            delta_cl_t=delta_cl_t,
            alpha=alpha,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        sjt_t, s0t = self._choice_probs_t(delta_t=delta_t)

        # Clip to avoid log(0).
        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        ll = tf.reduce_sum(qjt_t * tf.math.log(sjt_t))
        ll += q0t_t * tf.math.log(s0t)
        return ll

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
        """Compute per-market log-likelihood contributions (shape (T,)).

        Returns a length-T vector so callers can either:
          - sum over markets for global updates, or
          - keep the vector form for elementwise MH updates.
        """
        T = tf.shape(delta_cl)[0]

        def per_t(t):
            return self.market_loglik(
                qjt_t=qjt[t],
                q0t_t=q0t[t],
                delta_cl_t=delta_cl[t],
                alpha=alpha,
                E_bar_t=E_bar[t],
                njt_t=njt[t],
            )

        return tf.map_fn(per_t, tf.range(T), fn_output_signature=self.dtype)

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def logprior_global(self, alpha: tf.Tensor) -> tf.Tensor:
        """Compute the scalar prior for alpha.

        Independent Normal prior:
          alpha ~ Normal(alpha_mean, alpha_var)
        """
        lp_alpha = (
            -0.5 * tf.math.log(self.two_pi * self.alpha_var)
            - 0.5 * tf.square(alpha - self.alpha_mean) / self.alpha_var
        )
        return lp_alpha

    def logprior_market_vec(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Compute per-market prior contributions for (E_bar, njt, gamma, phi).

        For each market t, this includes:
          - E_bar[t] prior: Normal(E_bar_mean, E_bar_var)
          - njt[t,:] prior: Normal with variance chosen by gamma[t,:]
          - gamma[t,:] prior: Bernoulli(phi[t]) elementwise
          - phi[t] prior: Beta(a_phi, b_phi)

        Returns:
            lp_t: Tensor of shape (T,).
        """
        # E_bar prior: independent across markets.
        lp_E = (
            -0.5 * tf.math.log(self.two_pi * self.E_bar_var)
            - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )  # (T,)

        # njt prior: spike vs slab variance selected by gamma.
        var = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq  # (T,J)
        log_var = gamma * self.log_T1_sq + (1.0 - gamma) * self.log_T0_sq  # (T,J)
        lp_n_entry = -0.5 * (tf.math.log(self.two_pi) + log_var + tf.square(njt) / var)
        lp_n = tf.reduce_sum(lp_n_entry, axis=1)  # (T,)

        # gamma prior: Bernoulli(phi) independent across products.
        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)  # (T,)
        phi_b = phi[:, None]  # (T,1) broadcasts to (T,J)
        lp_g_entry = gamma * tf.math.log(phi_b) + (1.0 - gamma) * tf.math.log(
            1.0 - phi_b
        )
        lp_g = tf.reduce_sum(lp_g_entry, axis=1)  # (T,)

        # phi prior: Beta(a_phi, b_phi) independent across markets.
        a = self.a_phi
        b = self.b_phi
        logB = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
        lp_phi = (
            (a - 1.0) * tf.math.log(phi) + (b - 1.0) * tf.math.log(1.0 - phi) - logB
        )  # (T,)

        return lp_E + lp_n + lp_g + lp_phi

    # ------------------------------------------------------------------
    # Posterior (per-market factorization)
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
        """Compute per-market log posterior terms (shape (T,)).

        Returns:
          logpost_t = loglik_t + logprior_market_t

        The global prior logprior_global(alpha) is excluded so that:
          - market-level MH updates can use logpost_vec directly, and
          - global-parameter updates can add the global prior once after summing.
        """
        ll_t = self.loglik_vec(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
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
        """Compute the full scalar log posterior.

        Full posterior is:
          sum_t logpost_vec[t] + logprior_global(alpha)
        """
        lp_t = self.logpost_vec(
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            alpha=alpha,
            E_bar=E_bar,
            njt=njt,
            gamma=gamma,
            phi=phi,
        )
        return tf.reduce_sum(lp_t) + self.logprior_global(alpha=alpha)
