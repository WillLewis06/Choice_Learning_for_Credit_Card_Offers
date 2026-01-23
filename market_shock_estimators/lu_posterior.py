# market_shock_estimators/lu_posterior.py

import numpy as np
import tensorflow as tf


class LuPosteriorTF:
    """
    Canonical TF implementation of Lu (2025) log posterior components.

    This module is deterministic:
      - no RNG
      - no sampler logic
      - no state mutation

    It provides:
      - full_logp: scalar full log posterior
      - market_block_logp: scalar market-t block log posterior in (E_bar_t[t], njt[t,:])
      - market_block_grad_hess: gradient/Hessian for the market-t block

    Random coefficients:
      - General d-dimensional RC on selected columns of x_jt via rc_indices (default d=1).
      - r has shape (d,), where sigma_k = exp(r_k).
      - draws has shape (R,d). If passed 1D, it is treated as (R,1).
    """

    def __init__(
        self,
        x_jt,
        q_jt,
        q0_t,
        draws,
        *,
        rc_indices=None,
        beta_var=10.0,
        Ebar_var=10.0,
        r_var=0.5,
        T0_sq=1e-3,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
        dtype=tf.float64,
        eps=1e-15,
    ):
        x_jt = np.asarray(x_jt, dtype=float)
        q_jt = np.asarray(q_jt, dtype=float)
        q0_t = np.asarray(q0_t, dtype=float)
        draws = np.asarray(draws, dtype=float)

        if x_jt.ndim != 3:
            raise ValueError("x_jt must have shape (T, J, K).")
        T, J, K = x_jt.shape

        if q_jt.shape != (T, J):
            raise ValueError(f"q_jt must have shape (T, J)=({T},{J}).")
        if q0_t.shape != (T,):
            raise ValueError(f"q0_t must have shape (T,)=({T},).")

        if draws.ndim == 1:
            draws = draws[:, None]
        if draws.ndim != 2 or draws.shape[0] == 0:
            raise ValueError("draws must have shape (R,d) (or 1D treated as (R,1)).")

        R, d = draws.shape

        if rc_indices is None:
            rc_indices = [0]
        rc_indices = np.asarray(rc_indices, dtype=int).ravel()
        if rc_indices.size == 0:
            raise ValueError("rc_indices must be non-empty.")
        if np.any(rc_indices < 0) or np.any(rc_indices >= K):
            raise ValueError(f"rc_indices must be within [0, {K-1}].")
        if rc_indices.size != d:
            raise ValueError(
                "rc_indices length must equal draws second dimension d. "
                f"Got len(rc_indices)={rc_indices.size}, draws has d={d}."
            )

        if T0_sq <= 0.0 or T1_sq <= 0.0:
            raise ValueError("T0_sq and T1_sq must be positive.")

        self.dtype = dtype
        self.eps = tf.constant(float(eps), dtype=self.dtype)

        # Dimensions
        self.T = int(T)
        self.J = int(J)
        self.K = int(K)
        self.R = int(R)
        self.d = int(d)

        # Hyperparameters
        self.beta_var = tf.constant(float(beta_var), dtype=self.dtype)
        self.Ebar_var = tf.constant(float(Ebar_var), dtype=self.dtype)
        self.r_var = tf.constant(float(r_var), dtype=self.dtype)

        self.T0_sq = tf.constant(float(T0_sq), dtype=self.dtype)
        self.T1_sq = tf.constant(float(T1_sq), dtype=self.dtype)
        self.log_T0_sq = tf.constant(float(np.log(T0_sq)), dtype=self.dtype)
        self.log_T1_sq = tf.constant(float(np.log(T1_sq)), dtype=self.dtype)

        self.a_phi = tf.constant(float(a_phi), dtype=self.dtype)
        self.b_phi = tf.constant(float(b_phi), dtype=self.dtype)

        # Data tensors
        self.x = tf.constant(x_jt, dtype=self.dtype)  # (T,J,K)
        self.q = tf.constant(q_jt, dtype=self.dtype)  # (T,J)
        self.q0 = tf.constant(q0_t, dtype=self.dtype)  # (T,)
        self.draws = tf.constant(draws, dtype=self.dtype)  # (R,d)

        # Random-coefficient covariates Z_jt = x_jt[:, :, rc_indices] -> (T,J,d)
        self.rc_indices = tf.constant(rc_indices.tolist(), dtype=tf.int32)
        self.Z = tf.gather(self.x, self.rc_indices, axis=2)  # (T,J,d)

    # ------------------------------------------------------------------
    # Helpers: RC logit probabilities
    # ------------------------------------------------------------------

    def _choice_probs_all(self, delta, r):
        """
        Compute simulated choice probabilities for all markets.

        delta: (T,J)
        r: (d,)
        returns: (p_jt (T,J), p0_t (T,))
        """
        # sigma_k = exp(r_k)
        sigma = tf.exp(r)  # (d,)
        scaled_draws = self.draws * sigma[None, :]  # (R,d)

        # mu_tjr = sum_d Z_tjd * scaled_draws_rd
        mu = tf.einsum("tjd,rd->tjr", self.Z, scaled_draws)  # (T,J,R)
        util = delta[:, :, None] + mu  # (T,J,R)

        m = tf.reduce_max(util, axis=1, keepdims=True)  # (T,1,R)
        expu = tf.exp(util - m)  # (T,J,R)

        outside = tf.exp(-m)  # (T,1,R)
        denom = outside + tf.reduce_sum(expu, axis=1, keepdims=True)  # (T,1,R)

        p_jt_r = expu / denom  # (T,J,R)
        p0_t_r = outside[:, 0, :] / denom[:, 0, :]  # (T,R)

        p_jt = tf.reduce_mean(p_jt_r, axis=2)  # (T,J)
        p0_t = tf.reduce_mean(p0_t_r, axis=1)  # (T,)
        return p_jt, p0_t

    def _choice_probs_market(self, t, delta_t, r):
        """
        Market-t simulated probabilities.

        delta_t: (J,)
        r: (d,)
        returns: (s_j (J,), s0 ())
        """
        Z_t = tf.gather(self.Z, t)  # (J,d)

        sigma = tf.exp(r)  # (d,)
        scaled_draws = self.draws * sigma[None, :]  # (R,d)

        mu = tf.einsum("jd,rd->jr", Z_t, scaled_draws)  # (J,R)
        util = delta_t[:, None] + mu  # (J,R)

        m = tf.reduce_max(util, axis=0, keepdims=True)  # (1,R)
        expu = tf.exp(util - m)  # (J,R)

        outside = tf.exp(-m)  # (1,R)
        denom = outside + tf.reduce_sum(expu, axis=0, keepdims=True)  # (1,R)

        s_j_r = expu / denom  # (J,R)
        s0_r = outside[0, :] / denom[0, :]  # (R,)

        s_j = tf.reduce_mean(s_j_r, axis=1)  # (J,)
        s0 = tf.reduce_mean(s0_r)  # ()
        return s_j, s0

    # ------------------------------------------------------------------
    # Full log posterior
    # ------------------------------------------------------------------

    @tf.function
    def full_logp(self, beta, r, E_bar_t, njt, gamma_jt, phi_t):
        """
        Full log posterior (scalar).

        Shapes:
          beta: (K,)
          r: (d,)
          E_bar_t: (T,)
          njt: (T,J)
          gamma_jt: (T,J) {0,1}
          phi_t: (T,) in (0,1)
        """
        beta = tf.convert_to_tensor(beta, dtype=self.dtype)
        r = tf.convert_to_tensor(r, dtype=self.dtype)
        E_bar_t = tf.convert_to_tensor(E_bar_t, dtype=self.dtype)
        njt = tf.convert_to_tensor(njt, dtype=self.dtype)
        gamma_jt = tf.cast(gamma_jt, self.dtype)
        phi_t = tf.convert_to_tensor(phi_t, dtype=self.dtype)

        # delta = x*beta + E_bar + n
        xb = tf.einsum("tjk,k->tj", self.x, beta)  # (T,J)
        delta = xb + E_bar_t[:, None] + njt  # (T,J)

        p_jt, p0_t = self._choice_probs_all(delta, r)

        ll = tf.reduce_sum(self.q * tf.math.log(p_jt + self.eps))
        ll += tf.reduce_sum(self.q0 * tf.math.log(p0_t + self.eps))

        # Global priors
        lp = ll
        lp += -0.5 * tf.reduce_sum(beta * beta) / self.beta_var
        lp += -0.5 * tf.reduce_sum(r * r) / self.r_var
        lp += -0.5 * tf.reduce_sum(E_bar_t * E_bar_t) / self.Ebar_var

        # Spike-and-slab on njt | gamma
        var = gamma_jt * self.T1_sq + (1.0 - gamma_jt) * self.T0_sq
        logvar = gamma_jt * self.log_T1_sq + (1.0 - gamma_jt) * self.log_T0_sq
        lp += -0.5 * tf.reduce_sum((njt * njt) / var)
        lp += -0.5 * tf.reduce_sum(logvar)

        # Discrete: gamma | phi
        phi = phi_t[:, None]  # (T,1)
        lp += tf.reduce_sum(gamma_jt * tf.math.log(phi + self.eps))
        lp += tf.reduce_sum((1.0 - gamma_jt) * tf.math.log(1.0 - phi + self.eps))

        # Beta prior on phi (Lu paper alignment), up to constant
        lp += tf.reduce_sum((self.a_phi - 1.0) * tf.math.log(phi_t + self.eps))
        lp += tf.reduce_sum((self.b_phi - 1.0) * tf.math.log(1.0 - phi_t + self.eps))

        return lp

    # ------------------------------------------------------------------
    # Market block log posterior and derivatives
    # ------------------------------------------------------------------

    @tf.function
    def market_block_logp(self, t, theta_block, beta, r, gamma_t):
        """
        Market-t block log posterior as a function of:
          theta_block = [E_bar_t[t], njt[t,:]]  shape (1+J,)

        Includes only terms that depend on theta_block:
          - market-t likelihood
          - prior for E_bar_t[t]
          - prior for njt[t,:] conditional on gamma_t (including log var)

        Omits constants independent of theta_block.
        """
        t = tf.cast(t, tf.int32)
        theta_block = tf.convert_to_tensor(theta_block, dtype=self.dtype)
        beta = tf.convert_to_tensor(beta, dtype=self.dtype)
        r = tf.convert_to_tensor(r, dtype=self.dtype)
        gamma_t = tf.cast(gamma_t, self.dtype)  # (J,)

        x_t = tf.gather(self.x, t)  # (J,K)
        q_t = tf.gather(self.q, t)  # (J,)
        q0_t = tf.gather(self.q0, t)  # ()

        Ebar = theta_block[0]  # ()
        nj = theta_block[1:]  # (J,)

        delta_t = tf.linalg.matvec(x_t, beta) + Ebar + nj  # (J,)

        s_j, s0 = self._choice_probs_market(t, delta_t, r)

        ll = tf.reduce_sum(q_t * tf.math.log(s_j + self.eps)) + q0_t * tf.math.log(
            s0 + self.eps
        )

        var = gamma_t * self.T1_sq + (1.0 - gamma_t) * self.T0_sq
        logvar = gamma_t * self.log_T1_sq + (1.0 - gamma_t) * self.log_T0_sq

        lp = ll
        lp += -0.5 * (Ebar * Ebar) / self.Ebar_var
        lp += -0.5 * tf.reduce_sum((nj * nj) / var)
        lp += -0.5 * tf.reduce_sum(logvar)

        return lp

    def market_block_grad_hess(self, t, theta_block, beta, r, gamma_t):
        """
        Gradient and Hessian of market_block_logp w.r.t theta_block.

        Returns:
          grad: (1+J,) tensor
          hess: (1+J,1+J) tensor
        """
        t_tf = tf.constant(int(t), dtype=tf.int32)
        theta = tf.convert_to_tensor(theta_block, dtype=self.dtype)

        with tf.GradientTape(persistent=True) as outer:
            outer.watch(theta)
            with tf.GradientTape() as inner:
                inner.watch(theta)
                lp = self.market_block_logp(t_tf, theta, beta, r, gamma_t)
            grad = inner.gradient(lp, theta)
        hess = outer.jacobian(grad, theta)
        return grad, hess
