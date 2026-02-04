"""
TensorFlow posterior utilities for the Lu shrinkage estimator.

This file defines `LuPosteriorTF`, a TF-native collection of:
  - Choice probabilities for a random-coefficient logit (RC logit),
  - Multinomial log-likelihood terms using observed counts (qjt, q0t),
  - Prior terms for global parameters and the sparse shock structure,
  - Convenience functions that return either per-market vectors or full scalars.

The design is intentionally per-market:
  - `loglik_vec(...)`, `logprior_market_vec(...)`, and `logpost_vec(...)` return
    a length-T vector, which enables elementwise MH updates (e.g. for E_bar).
"""

from __future__ import annotations

import tensorflow as tf


class LuPosteriorTF:
    """Posterior pieces for the Lu shrinkage model (TF-only).

    Utility (price-only random coefficient variant used in this codebase):
      delta_jt = beta_p * pjt + beta_w * wjt + E_bar_t + njt

    Heterogeneity in the price coefficient is simulated with fixed draws:
      beta_p_i = beta_p + exp(r) * v_i,  v_i ~ Normal(0, 1)
      sigma = exp(r) > 0

    Sparse market-product shocks use a spike-and-slab (continuous mixture):
      gamma | phi ~ Bernoulli(phi) elementwise
      phi ~ Beta(a_phi, b_phi) per market

      njt | gamma=1 ~ Normal(0, T1_sq)   (slab, large variance)
      njt | gamma=0 ~ Normal(0, T0_sq)   (spike, small variance)

    Likelihood uses multinomial counts and drops the combinatorial constant:
      log p(q_t | s_t) = q0t * log(s0t) + sum_j qjt * log(sjt) + const

    Interfaces are organized to match the MCMC updates:
      - market_loglik: scalar likelihood contribution for one market
      - loglik_vec: length-T vector of per-market likelihood contributions
      - logprior_global: scalar prior for (beta_p, beta_w, r)
      - logprior_market_vec: length-T vector for (E_bar, njt, gamma, phi) priors
      - logpost_vec: length-T vector of (likelihood + market prior)
      - logpost: full scalar posterior (sum over markets + global prior)
    """

    def __init__(
        self,
        n_draws: int,
        seed: int,
        beta_p_mean: float = 0.0,
        beta_p_var: float = 10.0,
        beta_w_mean: float = 0.0,
        beta_w_var: float = 10.0,
        r_mean: float = 0.0,
        r_var: float = 0.5,
        E_bar_mean: float = 0.0,
        E_bar_var: float = 10.0,
        T0_sq: float = 1e-3,
        T1_sq: float = 1.0,
        a_phi: float = 1.0,
        b_phi: float = 1.0,
        eps: float = 1e-15,
        dtype=tf.float64,
    ):
        """Initialize hyperparameters and simulation draws.

        The RC logit likelihood integrates over `v_draws`. For performance and
        determinism, those draws are generated once and stored as `self.v_draws`.
        """
        self.dtype = dtype
        self.eps = tf.constant(eps, dtype=dtype)

        self.n_draws = int(n_draws)
        self.seed = int(seed)

        # Fixed simulation draws v_i ~ Normal(0,1), shape (n_draws,).
        g = tf.random.Generator.from_seed(self.seed)
        self.v_draws = tf.cast(g.normal(shape=(self.n_draws,)), dtype)

        # Prior hyperparameters stored as TF constants for graph compatibility.
        self.beta_p_mean = tf.constant(beta_p_mean, dtype=dtype)
        self.beta_p_var = tf.constant(beta_p_var, dtype=dtype)
        self.beta_w_mean = tf.constant(beta_w_mean, dtype=dtype)
        self.beta_w_var = tf.constant(beta_w_var, dtype=dtype)

        self.r_mean = tf.constant(r_mean, dtype=dtype)
        self.r_var = tf.constant(r_var, dtype=dtype)

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
    # Deterministic helpers (price-only random coefficient)
    # ------------------------------------------------------------------

    def _mean_utility_jt(
        self,
        pjt_t,
        wjt_t,
        beta_p,
        beta_w,
        E_bar_t,
        njt_t,
    ):
        """Compute delta_t for one market.

        For a single market t (vectors over products j=0..J-1):
          delta_t = beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t

        Returns:
            delta_t: Tensor of shape (J,).
        """
        return beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t

    def _choice_probs_t(self, pjt_t, delta_t, r):
        """Compute (sjt_t, s0t) for one market under price-only RC logit.

        Integration is done with fixed draws `v_draws`:
          beta_p_i = beta_p + exp(r) * v_i

        Outside option utility is normalized to 0. A max-shift is used for
        numerical stability of exp().

        Args:
            pjt_t: Prices for market t, shape (J,).
            delta_t: Mean utilities for market t, shape (J,).
            r: log(sigma), scalar.

        Returns:
            sjt_t: Inside shares for market t, shape (J,).
            s0t: Outside share for market t, scalar.
        """
        sigma = tf.exp(r)
        scaled_v = sigma * self.v_draws  # (R,)

        # Draw-specific utilities: util[j, i] = delta[j] + p[j]*sigma*v[i].
        mu = pjt_t[:, None] * scaled_v[None, :]  # (J,R)
        util = delta_t[:, None] + mu  # (J,R)

        # Stable logit including the outside option (utility 0).
        m = tf.reduce_max(util, axis=0, keepdims=True)  # (1,R)
        m = tf.maximum(m, tf.zeros_like(m))  # ensures outside=0 is included

        expu = tf.exp(util - m)  # (J,R)
        exp0 = tf.exp(-m)  # (1,R)
        denom = exp0 + tf.reduce_sum(expu, axis=0, keepdims=True)  # (1,R)

        sjt_draw = expu / denom  # (J,R)
        s0_draw = exp0 / denom  # (1,R)

        sjt_t = tf.reduce_mean(sjt_draw, axis=1)  # (J,)
        s0t = tf.reduce_mean(s0_draw[0, :])  # scalar
        return sjt_t, s0t

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def market_loglik(
        self,
        qjt_t,
        q0t_t,
        pjt_t,
        wjt_t,
        beta_p,
        beta_w,
        r,
        E_bar_t,
        njt_t,
    ) -> tf.Tensor:
        """Compute one market's multinomial log-likelihood term.

        Returns:
          q0t_t * log(s0t) + sum_j qjt_t[j] * log(sjt_t[j])

        The multinomial combinatorial constant is omitted because it does not
        depend on parameters and cancels in MH ratios.
        """
        delta_t = self._mean_utility_jt(
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        sjt_t, s0t = self._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

        # Clip to avoid log(0).
        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        ll = tf.reduce_sum(qjt_t * tf.math.log(sjt_t))
        ll += q0t_t * tf.math.log(s0t)
        return ll

    @tf.function(reduce_retracing=True)
    def loglik_vec(
        self,
        qjt,
        q0t,
        pjt,
        wjt,
        beta_p,
        beta_w,
        r,
        E_bar,
        njt,
    ) -> tf.Tensor:
        """Compute per-market log-likelihood contributions (shape (T,)).

        Returns a length-T vector so callers can either:
          - sum over markets for global updates, or
          - keep the vector form for elementwise MH updates.
        """
        T = tf.shape(pjt)[0]

        def per_t(t):
            """Return market t log-likelihood (scalar)."""
            return self.market_loglik(
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

        return tf.map_fn(per_t, tf.range(T), fn_output_signature=self.dtype)

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def logprior_global(self, beta_p, beta_w, r) -> tf.Tensor:
        """Compute the scalar prior for (beta_p, beta_w, r).

        These are independent Normal priors:
          beta_p ~ Normal(beta_p_mean, beta_p_var)
          beta_w ~ Normal(beta_w_mean, beta_w_var)
          r      ~ Normal(r_mean, r_var)
        """
        lp_beta_p = (
            -0.5 * tf.math.log(self.two_pi * self.beta_p_var)
            - 0.5 * tf.square(beta_p - self.beta_p_mean) / self.beta_p_var
        )
        lp_beta_w = (
            -0.5 * tf.math.log(self.two_pi * self.beta_w_var)
            - 0.5 * tf.square(beta_w - self.beta_w_mean) / self.beta_w_var
        )
        lp_r = (
            -0.5 * tf.math.log(self.two_pi * self.r_var)
            - 0.5 * tf.square(r - self.r_mean) / self.r_var
        )
        return lp_beta_p + lp_beta_w + lp_r

    def logprior_market_vec(
        self,
        E_bar,
        njt,
        gamma,
        phi,
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
        qjt,
        q0t,
        pjt,
        wjt,
        beta_p,
        beta_w,
        r,
        E_bar,
        njt,
        gamma,
        phi,
    ) -> tf.Tensor:
        """Compute per-market log posterior terms (shape (T,)).

        Returns:
          logpost_t = loglik_t + logprior_market_t

        The global prior logprior_global(beta_p, beta_w, r) is excluded so that:
          - market-level MH updates can use logpost_vec directly, and
          - global-parameter updates can add the global prior once after summing.
        """
        ll_t = self.loglik_vec(
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
        lp_t = self.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)
        return ll_t + lp_t

    def logpost(
        self,
        qjt,
        q0t,
        pjt,
        wjt,
        beta_p,
        beta_w,
        r,
        E_bar,
        njt,
        gamma,
        phi,
    ) -> tf.Tensor:
        """Compute the full scalar log posterior.

        Full posterior is:
          sum_t logpost_vec[t] + logprior_global(beta_p, beta_w, r)
        """
        lp_t = self.logpost_vec(
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar=E_bar,
            njt=njt,
            gamma=gamma,
            phi=phi,
        )
        return tf.reduce_sum(lp_t) + self.logprior_global(
            beta_p=beta_p, beta_w=beta_w, r=r
        )
