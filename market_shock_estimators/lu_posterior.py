from __future__ import annotations

import tensorflow as tf


class LuPosteriorTF:
    """
    Lu (2025) shrinkage model posterior pieces (TF-only), organized around a
    canonical per-market vector factorization.

    Model (Section 3/4, simulation uses price-only random coefficient):
      delta_jt = beta_p * pjt + beta_w * wjt + E_bar_t + n_jt
      beta_{p,i} = beta_p + exp(r) * v_i,  v_i ~ N(0,1)

      gamma_jt | phi_t ~ Bernoulli(phi_t)
      phi_t ~ Beta(a_phi, b_phi)

      Spike-and-slab (mixture of two normals, NOT point-mass):
        n_jt | gamma_jt=1 ~ N(0, T1_sq)
        n_jt | gamma_jt=0 ~ N(0, T0_sq)   with T0_sq << T1_sq

    Likelihood uses multinomial counts qjt, q0t:
      log p(q_t | s_t) = q0t*log s0t + sum_j qjt*log sjt + const
    (we drop the combinatorial constant).

    Canonical interfaces:
      - market_loglik(...) -> scalar          (single market)
      - loglik_vec(...)    -> (T,)            (per-market likelihood vector)
      - logprior_global(...) -> scalar        (global prior for beta_p,beta_w,r)
      - logprior_market_vec(...) -> (T,)      (per-market prior contributions)
      - logpost_vec(...)   -> (T,)            (per-market posterior contributions)
      - logpost(...)       -> scalar          (full posterior scalar)
    """

    def __init__(
        self,
        *,
        n_draws: int,
        seed: int,
        # Priors: beta_p, beta_w (independent normals)
        beta_p_mean: float = 0.0,
        beta_p_var: float = 10.0,
        beta_w_mean: float = 0.0,
        beta_w_var: float = 10.0,
        # Prior on r = log(sigma) (normal)
        r_mean: float = 0.0,
        r_var: float = 0.5,
        # Prior on E_bar_t (normal)
        E_bar_mean: float = 0.0,
        E_bar_var: float = 10.0,
        # Spike-and-slab variances for n_jt | gamma_jt
        T0_sq: float = 1e-3,
        T1_sq: float = 1.0,
        # Beta prior for phi_t
        a_phi: float = 1.0,
        b_phi: float = 1.0,
        # Numerical
        eps: float = 1e-15,
        dtype=tf.float64,
    ):
        self.dtype = dtype
        self.eps = tf.constant(eps, dtype=dtype)

        self.n_draws = int(n_draws)
        self.seed = int(seed)

        # TF RNG and fixed simulation draws v_i ~ N(0,1), shape (R,)
        g = tf.random.Generator.from_seed(self.seed)
        self.v_draws = tf.cast(g.normal(shape=(self.n_draws,)), dtype)

        # Prior hyperparameters as TF constants
        self.beta_p_mean = tf.constant(beta_p_mean, dtype=dtype)
        self.beta_p_var = tf.constant(beta_p_var, dtype=dtype)
        self.beta_w_mean = tf.constant(beta_w_mean, dtype=dtype)
        self.beta_w_var = tf.constant(beta_w_var, dtype=dtype)

        self.r_mean = tf.constant(r_mean, dtype=dtype)
        self.r_var = tf.constant(r_var, dtype=dtype)

        self.E_bar_mean = tf.constant(E_bar_mean, dtype=dtype)
        self.E_bar_var = tf.constant(E_bar_var, dtype=dtype)

        self.T0_sq = tf.constant(T0_sq, dtype=dtype)
        self.T1_sq = tf.constant(T1_sq, dtype=dtype)
        self.log_T0_sq = tf.math.log(self.T0_sq)
        self.log_T1_sq = tf.math.log(self.T1_sq)

        self.a_phi = tf.constant(a_phi, dtype=dtype)
        self.b_phi = tf.constant(b_phi, dtype=dtype)

        self.two_pi = tf.constant(2.0 * 3.141592653589793, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Deterministic helpers (price-only random coefficient)
    # ------------------------------------------------------------------

    def _mean_utility_jt(
        self,
        *,
        pjt_t,
        wjt_t,
        beta_p,
        beta_w,
        E_bar_t,
        njt_t,
    ):
        """
        delta_t (J,) = beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t
        """
        beta_p = tf.cast(beta_p, self.dtype)
        beta_w = tf.cast(beta_w, self.dtype)
        E_bar_t = tf.cast(E_bar_t, self.dtype)

        return beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t

    def _choice_probs_t(self, *, pjt_t, delta_t, r):
        """
        Price-only RC shares for one market.

        Inputs:
          pjt_t  : (J,)
          delta_t: (J,)
          r      : scalar (log sd of price coefficient)

        Returns:
          sjt_t: (J,)
          s0t : scalar
        """
        r = tf.cast(r, self.dtype)

        sigma = tf.exp(r)  # scalar
        scaled_v = sigma * self.v_draws  # (R,)
        mu = pjt_t[:, None] * scaled_v[None, :]  # (J,R)
        util = delta_t[:, None] + mu  # (J,R)

        # Stable logit with outside option normalized to 0
        m = tf.reduce_max(util, axis=0, keepdims=True)  # (1,R)
        m = tf.maximum(m, tf.zeros_like(m))  # include outside=0 in max

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
        *,
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
        """
        Per-market log likelihood contribution (up to multinomial constant):
          q0t*log s0t + sum_j qjt*log sjt
        """
        qjt_t = tf.cast(qjt_t, self.dtype)  # (J,)
        q0t_t = tf.cast(q0t_t, self.dtype)  # scalar

        delta_t = self._mean_utility_jt(
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        sjt_t, s0t = self._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        ll = tf.reduce_sum(qjt_t * tf.math.log(sjt_t))
        ll += q0t_t * tf.math.log(s0t)
        return ll

    @tf.function(reduce_retracing=True)
    def loglik_vec(
        self,
        *,
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
        """
        Per-market log-likelihood vector.

        Inputs are batched:
          qjt, pjt, wjt, njt: (T,J)
          q0t, E_bar: (T,)

        Returns:
          ll_t: (T,)
        """
        qjt = tf.convert_to_tensor(qjt, dtype=self.dtype)
        q0t = tf.convert_to_tensor(q0t, dtype=self.dtype)
        pjt = tf.convert_to_tensor(pjt, dtype=self.dtype)
        wjt = tf.convert_to_tensor(wjt, dtype=self.dtype)
        E_bar = tf.convert_to_tensor(E_bar, dtype=self.dtype)
        njt = tf.convert_to_tensor(njt, dtype=self.dtype)

        T = tf.shape(pjt)[0]

        def per_t(t):
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

    def logprior_global(self, *, beta_p, beta_w, r) -> tf.Tensor:
        """
        Global prior contribution (scalar):
          log p(beta_p) + log p(beta_w) + log p(r)
        """
        beta_p = tf.cast(beta_p, self.dtype)
        beta_w = tf.cast(beta_w, self.dtype)
        r = tf.cast(r, self.dtype)

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
        *,
        E_bar,
        njt,
        gamma,
        phi,
    ) -> tf.Tensor:
        """
        Per-market prior contributions (vector of length T):

          log p(E_bar_t)
          + sum_j log p(n_jt | gamma_jt)
          + sum_j log p(gamma_jt | phi_t)
          + log p(phi_t)

        Inputs:
          E_bar : (T,)
          njt   : (T,J)
          gamma : (T,J) entries in {0,1} (float ok)
          phi   : (T,) entries in (0,1)

        Returns:
          lp_t : (T,)
        """
        E_bar = tf.convert_to_tensor(E_bar, dtype=self.dtype)  # (T,)
        njt = tf.convert_to_tensor(njt, dtype=self.dtype)  # (T,J)
        gamma = tf.cast(gamma, self.dtype)  # (T,J)
        phi = tf.cast(phi, self.dtype)  # (T,)

        # ---- E_bar_t ~ N(E_bar_mean, E_bar_var)
        lp_E = (
            -0.5 * tf.math.log(self.two_pi * self.E_bar_var)
            - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )  # (T,)

        # ---- n_jt | gamma_jt is Normal with variance chosen by gamma_jt
        var = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq  # (T,J)
        log_var = gamma * self.log_T1_sq + (1.0 - gamma) * self.log_T0_sq  # (T,J)

        lp_n_entry = -0.5 * (tf.math.log(self.two_pi) + log_var + tf.square(njt) / var)
        lp_n = tf.reduce_sum(lp_n_entry, axis=1)  # (T,)

        # ---- gamma_jt | phi_t ~ Bernoulli(phi_t), independent across j
        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)  # (T,)
        phi_b = phi[:, None]  # (T,1) -> broadcast to (T,J)
        lp_g_entry = gamma * tf.math.log(phi_b) + (1.0 - gamma) * tf.math.log(
            1.0 - phi_b
        )
        lp_g = tf.reduce_sum(lp_g_entry, axis=1)  # (T,)

        # ---- phi_t ~ Beta(a_phi, b_phi), independent across t
        a = self.a_phi
        b = self.b_phi
        logB = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
        lp_phi = (
            (a - 1.0) * tf.math.log(phi) + (b - 1.0) * tf.math.log(1.0 - phi) - logB
        )  # (T,)

        return lp_E + lp_n + lp_g + lp_phi

    # ------------------------------------------------------------------
    # Posterior (canonical factorization)
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def logpost_vec(
        self,
        *,
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
        """
        Per-market log posterior contributions (vector length T):

          logpost_t = loglik_t + logprior_market_t

        Returns:
          lp_t : (T,)
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
        *,
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
        """
        Full log posterior (scalar):

          sum_t logpost_vec[t] + logprior_global(beta_p,beta_w,r)
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
