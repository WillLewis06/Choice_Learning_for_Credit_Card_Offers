from __future__ import annotations

import tensorflow as tf


class LuPosteriorTF:
    """
    Lu (2025) shrinkage model posterior pieces (TF-only), aligned with the paper's
    factorization and MCMC blocking.

    Model (Section 3/4, simulation uses price-only random coefficient):
      delta_jt = beta_p * pjt + beta_w * wjt + E_bar_t + eta_jt
      beta_{p,i} = beta_p + exp(r) * v_i,  v_i ~ N(0,1)

      gamma_jt | phi_t ~ Bernoulli(phi_t)
      phi_t ~ Beta(a_phi, b_phi)

      Option A (point-mass spike-and-slab):
        eta_jt | gamma_jt=1 ~ N(0, sigma_eta^2)
        eta_jt | gamma_jt=0 = 0


    Likelihood uses multinomial counts qjt, q0t:
      log p(q_t | s_t) = q0t*log s0t + sum_j qjt*log sjt + const
    (we drop the combinatorial constant).

    All methods return scalar log-density contributions
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
        # Prior on E_bar_t (normal): center at Lu Section 4 DGP mean (-1)
        E_bar_mean: float = -1.0,
        E_bar_var: float = 10.0,
        # Slab variance for eta | gamma=1 (Option A: gamma=0 implies eta=0 exactly)
        sigma_eta_sq: float = 1.0,
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

        self.sigma_eta_sq = tf.constant(sigma_eta_sq, dtype=dtype)

        self.log_sigma_eta_sq = tf.math.log(self.sigma_eta_sq)

        self.a_phi = tf.constant(a_phi, dtype=dtype)
        self.b_phi = tf.constant(b_phi, dtype=dtype)

        self.two_pi = tf.constant(2.0 * 3.141592653589793, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Likelihood helpers (price-only random coefficient)
    # ------------------------------------------------------------------

    def _mean_utility_jt(
        self,
        *,
        pjt_t,
        wjt_t,
        beta_p,
        beta_w,
        E_bar_t,
        eta_t,
    ):
        """
        delta_t (J,) = beta_p * pjt_t + beta_w * wjt_t + E_bar_t + eta_t
        """
        pjt_t = tf.convert_to_tensor(pjt_t, dtype=self.dtype)
        wjt_t = tf.convert_to_tensor(wjt_t, dtype=self.dtype)
        eta_t = tf.convert_to_tensor(eta_t, dtype=self.dtype)
        beta_p = tf.cast(beta_p, self.dtype)
        beta_w = tf.cast(beta_w, self.dtype)
        E_bar_t = tf.cast(E_bar_t, self.dtype)

        return beta_p * pjt_t + beta_w * wjt_t + E_bar_t + eta_t

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
        pjt_t = tf.convert_to_tensor(pjt_t, dtype=self.dtype)  # (J,)
        delta_t = tf.convert_to_tensor(delta_t, dtype=self.dtype)  # (J,)
        r = tf.cast(r, self.dtype)

        sigma = tf.exp(r)  # scalar
        # mu_{j,i} = p_j * (sigma * v_i)
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
    # Log likelihood
    # ------------------------------------------------------------------

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
        eta_t,
    ):
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
            eta_t=eta_t,
        )
        sjt_t, s0t = self._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

        # Numerical protection for logs
        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        ll = tf.reduce_sum(qjt_t * tf.math.log(sjt_t))
        ll += q0t_t * tf.math.log(s0t)
        return ll

    def full_loglik(
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
        eta,
    ):
        """
        Sum of market log-likelihoods over t.
        """
        qjt = tf.convert_to_tensor(qjt, dtype=self.dtype)  # (T,J)
        q0t = tf.convert_to_tensor(q0t, dtype=self.dtype)  # (T,)
        pjt = tf.convert_to_tensor(pjt, dtype=self.dtype)  # (T,J)
        wjt = tf.convert_to_tensor(wjt, dtype=self.dtype)  # (T,J)
        E_bar = tf.convert_to_tensor(E_bar, dtype=self.dtype)  # (T,)
        eta = tf.convert_to_tensor(eta, dtype=self.dtype)  # (T,J)

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
                eta_t=eta[t],
            )

        ll_t = tf.map_fn(per_t, tf.range(T), fn_output_signature=self.dtype)
        return tf.reduce_sum(ll_t)

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def logprior_beta(self, *, beta_p, beta_w):
        """
        log p(beta_p) + log p(beta_w), independent normals (up to constants).
        """

        beta_p = tf.cast(beta_p, self.dtype)
        beta_w = tf.cast(beta_w, self.dtype)

        lp_p = (
            -0.5 * tf.math.log(self.two_pi * self.beta_p_var)
            - 0.5 * tf.square(beta_p - self.beta_p_mean) / self.beta_p_var
        )
        lp_w = (
            -0.5 * tf.math.log(self.two_pi * self.beta_w_var)
            - 0.5 * tf.square(beta_w - self.beta_w_mean) / self.beta_w_var
        )
        return lp_p + lp_w

    def logprior_r(self, *, r):
        """
        log p(r), normal on r=log(sigma) (up to constants).
        """
        r = tf.cast(r, self.dtype)
        return (
            -0.5 * tf.math.log(self.two_pi * self.r_var)
            - 0.5 * tf.square(r - self.r_mean) / self.r_var
        )

    def logprior_E_bar(self, *, E_bar):
        """
        Sum_t log p(E_bar_t), iid normal (up to constants).
        """
        E_bar = tf.convert_to_tensor(E_bar, dtype=self.dtype)
        T = tf.cast(tf.size(E_bar), self.dtype)

        return -0.5 * T * tf.math.log(
            self.two_pi * self.E_bar_var
        ) - 0.5 * tf.reduce_sum(tf.square(E_bar - self.E_bar_mean) / self.E_bar_var)

    def logprior_eta(self, *, eta, gamma):
        """
        Option A (point-mass spike-and-slab):

          eta_j | gamma_j=1 ~ N(0, sigma_eta_sq)
          eta_j | gamma_j=0 = 0

        Returns -inf if any inactive coordinate has eta != 0 (within tolerance).
        Otherwise returns the slab Normal log density for active coordinates only.
        """
        eta = tf.convert_to_tensor(eta, dtype=self.dtype)
        gamma = tf.cast(gamma, self.dtype)

        # Inactive coords must be exactly zero (up to numerical tolerance)
        tol = tf.cast(1e-12, self.dtype)
        inactive = 1.0 - gamma
        violates = tf.reduce_any(tf.abs(eta) * inactive > tol)

        def lp_valid():
            var = self.sigma_eta_sq
            n_active = tf.reduce_sum(gamma)
            return -0.5 * n_active * tf.math.log(
                self.two_pi * var
            ) - 0.5 * tf.reduce_sum(gamma * tf.square(eta) / var)

        return tf.cond(
            violates,
            lambda: tf.cast(-float("inf"), self.dtype),
            lp_valid,
        )

    def logprior_gamma(self, *, gamma, phi):
        """
        log p(gamma | phi), Bernoulli per j and t.

        gamma : (T,J) or (J,)
        phi   : (T,) or scalar for market t
        """
        gamma = tf.cast(gamma, self.dtype)
        phi = tf.cast(phi, self.dtype)

        # Clip phi away from 0/1 for log
        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)

        # Support broadcasting:
        # - if gamma is (J,) and phi is scalar: fine
        # - if gamma is (T,J) and phi is (T,), we want phi[:,None]
        if gamma.shape.rank == 2 and phi.shape.rank == 1:
            phi = phi[:, None]

        return tf.reduce_sum(
            gamma * tf.math.log(phi) + (1.0 - gamma) * tf.math.log(1.0 - phi)
        )

    def logprior_phi(self, *, phi):
        """
        Sum_t log pi(phi_t), Beta(a_phi,b_phi) (up to constants).
        """
        phi = tf.cast(phi, self.dtype)
        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)

        a = self.a_phi
        b = self.b_phi

        # log Beta density: (a-1)log phi + (b-1)log(1-phi) - log B(a,b)
        logB = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
        lp = (a - 1.0) * tf.math.log(phi) + (b - 1.0) * tf.math.log(1.0 - phi) - logB
        return tf.reduce_sum(lp)

    # ------------------------------------------------------------------
    # Posterior composition (matches Lu factorization / blocking)
    # ------------------------------------------------------------------

    def market_logprior(self, *, E_bar_t, eta_t, gamma_t, phi_t):
        """
        Market-local prior contribution:
          log p(E_bar_t) + log p(eta_t | gamma_t) + log p(gamma_t | phi_t) + log pi(phi_t)
        """
        E_bar_t = tf.cast(E_bar_t, self.dtype)
        phi_t = tf.cast(phi_t, self.dtype)
        return (
            (
                -0.5 * tf.math.log(self.two_pi * self.E_bar_var)
                - 0.5 * tf.square(E_bar_t - self.E_bar_mean) / self.E_bar_var
            )
            + self.logprior_eta(eta=eta_t, gamma=gamma_t)
            + self.logprior_gamma(gamma=gamma_t, phi=phi_t)
            + self.logprior_phi(phi=tf.reshape(phi_t, (1,)))
        )

    def market_logpost(
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
        eta_t,
        gamma_t,
        phi_t,
    ):
        """
        Market-local log posterior contribution:
          log p(q_t | ...) + market_logprior(...)
        """

        # Robustness: likelihood should only “see” active shocks.
        # Prior remains strict (logprior_eta returns -inf if inactive eta != 0).
        eta_eff = tf.cast(gamma_t, self.dtype) * tf.cast(eta_t, self.dtype)

        return self.market_loglik(
            qjt_t=qjt_t,
            q0t_t=q0t_t,
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar_t=E_bar_t,
            eta_t=eta_eff,
        ) + self.market_logprior(
            E_bar_t=E_bar_t, eta_t=eta_t, gamma_t=gamma_t, phi_t=phi_t
        )

    def global_logprior(self, *, beta_p, beta_w, r):
        """
        Global prior contribution:
          log p(beta_p,beta_w) + log p(r)
        """
        return self.logprior_beta(beta_p=beta_p, beta_w=beta_w) + self.logprior_r(r=r)

    def full_logpost(
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
        eta,
        gamma,
        phi,
    ):
        """
        Full log posterior:
          sum_t market_logpost_t + global_logprior
        """
        qjt = tf.convert_to_tensor(qjt, dtype=self.dtype)  # (T,J)
        q0t = tf.convert_to_tensor(q0t, dtype=self.dtype)  # (T,)
        pjt = tf.convert_to_tensor(pjt, dtype=self.dtype)  # (T,J)
        wjt = tf.convert_to_tensor(wjt, dtype=self.dtype)  # (T,J)
        E_bar = tf.convert_to_tensor(E_bar, dtype=self.dtype)  # (T,)
        eta = tf.convert_to_tensor(eta, dtype=self.dtype)  # (T,J)
        gamma = tf.cast(gamma, self.dtype)  # (T,J)
        phi = tf.cast(phi, self.dtype)  # (T,)

        T = tf.shape(pjt)[0]

        def per_t(t):
            return self.market_logpost(
                qjt_t=qjt[t],
                q0t_t=q0t[t],
                pjt_t=pjt[t],
                wjt_t=wjt[t],
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=E_bar[t],
                eta_t=eta[t],
                gamma_t=gamma[t],
                phi_t=phi[t],
            )

        lp_t = tf.map_fn(per_t, tf.range(T), fn_output_signature=self.dtype)
        return tf.reduce_sum(lp_t) + self.global_logprior(
            beta_p=beta_p, beta_w=beta_w, r=r
        )
