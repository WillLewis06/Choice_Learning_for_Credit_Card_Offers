# market_shock_estimators/lu_posterior.py

import numpy as np
import tensorflow as tf


class LuPosteriorTF:
    """
    Market-local posterior for Lu (2025) shrinkage estimator.

    This class is:
      - stateless (no sampler state)
      - market-local (no t index anywhere)
      - deterministic (no RNG)

    It exposes:
      - market_logp(theta, ...)
      - market_grad_hess(theta, ...)

    where theta = [E_bar_t, eta_{1t}, ..., eta_{Jt}].
    """

    def __init__(
        self,
        *,
        beta_var=10.0,
        Ebar_var=10.0,
        r_var=0.5,
        T0_sq=1e-3,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
        draws,
        dtype=tf.float64,
        eps=1e-15,
    ):
        draws = np.asarray(draws, dtype=float)
        if draws.ndim == 1:
            draws = draws[:, None]
        if draws.ndim != 2:
            raise ValueError("draws must have shape (R,d) or (R,)")

        self.draws = tf.constant(draws, dtype=dtype)
        self.R, self.d = self.draws.shape

        self.dtype = dtype
        self.eps = tf.constant(eps, dtype=dtype)

        # Priors / hyperparameters
        self.beta_var = tf.constant(beta_var, dtype=dtype)
        self.Ebar_var = tf.constant(Ebar_var, dtype=dtype)
        self.r_var = tf.constant(r_var, dtype=dtype)

        self.T0_sq = tf.constant(T0_sq, dtype=dtype)
        self.T1_sq = tf.constant(T1_sq, dtype=dtype)
        self.log_T0_sq = tf.constant(np.log(T0_sq), dtype=dtype)
        self.log_T1_sq = tf.constant(np.log(T1_sq), dtype=dtype)

        self.a_phi = tf.constant(a_phi, dtype=dtype)
        self.b_phi = tf.constant(b_phi, dtype=dtype)

    # ------------------------------------------------------------------
    # Choice probabilities (market-local)
    # ------------------------------------------------------------------

    def _choice_probs(self, Z, delta, r):
        """
        Z:     (J,d)
        delta: (J,)
        r:     (d,)

        returns:
          s_j: (J,)
          s0:  ()
        """
        sigma = tf.exp(r)  # (d,)
        scaled = self.draws * sigma[None, :]  # (R,d)

        mu = tf.einsum("jd,rd->jr", Z, scaled)  # (J,R)
        util = delta[:, None] + mu  # (J,R)

        m = tf.reduce_max(util, axis=0, keepdims=True)
        expu = tf.exp(util - m)

        outside = tf.exp(-m)
        denom = outside + tf.reduce_sum(expu, axis=0, keepdims=True)

        s_j = tf.reduce_mean(expu / denom, axis=1)
        s0 = tf.reduce_mean(outside[0] / denom[0])

        return s_j, s0

    # ------------------------------------------------------------------
    # Market log posterior
    # ------------------------------------------------------------------

    def market_logp(
        self,
        *,
        theta,
        q,
        q0,
        x,
        Z,
        beta,
        r,
        gamma,
    ):
        """
        Market log posterior.

        Inputs (market-local):
          theta : (1+J,)  [E_bar, eta_1,...,eta_J]
          q     : (J,)
          q0    : ()
          x     : (J,K)
          Z     : (J,d)
          beta  : (K,)
          r     : (d,)
          gamma : (J,) in {0,1}

        Returns:
          scalar log posterior contribution
        """
        # Tensorize + cast once
        theta = tf.convert_to_tensor(theta, self.dtype)
        q = tf.convert_to_tensor(q, self.dtype)
        q0 = tf.convert_to_tensor(q0, self.dtype)
        x = tf.convert_to_tensor(x, self.dtype)
        Z = tf.convert_to_tensor(Z, self.dtype)
        beta = tf.convert_to_tensor(beta, self.dtype)
        r = tf.convert_to_tensor(r, self.dtype)
        gamma = tf.cast(gamma, self.dtype)

        Ebar = theta[0]
        eta = theta[1:]

        delta = tf.linalg.matvec(x, beta) + Ebar + eta
        s_j, s0 = self._choice_probs(Z, delta, r)

        ll = tf.reduce_sum(q * tf.math.log(s_j + self.eps)) + q0 * tf.math.log(
            s0 + self.eps
        )

        var = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq
        logvar = gamma * self.log_T1_sq + (1.0 - gamma) * self.log_T0_sq

        lp = ll
        lp += -0.5 * (Ebar * Ebar) / self.Ebar_var
        lp += -0.5 * tf.reduce_sum((eta * eta) / var)
        lp += -0.5 * tf.reduce_sum(logvar)

        return lp

    # ------------------------------------------------------------------
    # Gradient + Hessian (single entry point)
    # ------------------------------------------------------------------

    def market_grad_hess(
        self,
        *,
        theta,
        q,
        q0,
        x,
        Z,
        beta,
        r,
        gamma,
    ):
        """
        Gradient and Hessian of market_logp w.r.t theta.

        Returns:
          grad : (1+J,)
          hess : (1+J,1+J)
        """

        # print("[lu_posterior] market_grad_hess start")
        theta = tf.convert_to_tensor(theta, self.dtype)

        with tf.GradientTape(persistent=True) as outer:
            outer.watch(theta)
            with tf.GradientTape() as inner:
                inner.watch(theta)
                lp = self.market_logp(
                    theta=theta,
                    q=q,
                    q0=q0,
                    x=x,
                    Z=Z,
                    beta=beta,
                    r=r,
                    gamma=gamma,
                )
            grad = inner.gradient(lp, theta)

        # print("[lu_posterior] gradient computed")
        hess = outer.jacobian(grad, theta)
        # print("[lu_posterior] hessian computed")
        del outer

        return lp, grad, hess
