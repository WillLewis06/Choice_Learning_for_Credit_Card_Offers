# market_shock_estimators/lu_posterior.py

from __future__ import annotations

import tensorflow as tf


class LuPosteriorTF:
    """
    Market-local posterior components for Lu (2025) shrinkage estimator.

    Design principles:
      - Pure TensorFlow (no NumPy)
      - Stateless and deterministic
      - No internal casting or tensorization
      - No gradient/Hessian computation here
      - Explicit separation of likelihood and priors

    This class provides scalar log-density contributions only.
    """

    def __init__(
        self,
        *,
        draws,
        beta_var=10.0,
        Ebar_var=10.0,
        r_var=0.5,
        T0_sq=1e-3,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
        eps=1e-15,
        dtype=tf.float64,
    ):
        # Simulation draws for random coefficients
        self.draws = tf.convert_to_tensor(draws, dtype=dtype)
        if self.draws.ndim == 1:
            self.draws = self.draws[:, None]
        if self.draws.ndim != 2:
            raise ValueError("draws must have shape (R,d) or (R,)")

        self.R = tf.shape(self.draws)[0]
        self.d = tf.shape(self.draws)[1]

        self.dtype = dtype
        self.eps = tf.constant(eps, dtype=dtype)

        # Prior hyperparameters (all TF constants)
        self.beta_var = tf.constant(beta_var, dtype=dtype)
        self.Ebar_var = tf.constant(Ebar_var, dtype=dtype)
        self.r_var = tf.constant(r_var, dtype=dtype)

        self.T0_sq = tf.constant(T0_sq, dtype=dtype)
        self.T1_sq = tf.constant(T1_sq, dtype=dtype)
        self.log_T0_sq = tf.math.log(self.T0_sq)
        self.log_T1_sq = tf.math.log(self.T1_sq)

        self.a_phi = tf.constant(a_phi, dtype=dtype)
        self.b_phi = tf.constant(b_phi, dtype=dtype)

    # ------------------------------------------------------------------
    # Choice probabilities (market-local)
    # ------------------------------------------------------------------

    def _choice_probs(self, Z, delta, r):
        """
        Compute market choice probabilities.

        Inputs:
          Z     : (J,d)
          delta : (J,)
          r     : (d,)

        Returns:
          s_j : (J,)
          s0  : scalar
        """
        sigma = tf.exp(r)  # (d,)
        scaled_draws = self.draws * sigma  # (R,d)

        mu = tf.einsum("jd,rd->jr", Z, scaled_draws)  # (J,R)
        util = delta[:, None] + mu  # (J,R)

        m = tf.reduce_max(util, axis=0, keepdims=True)
        expu = tf.exp(util - m)

        outside = tf.exp(-m)
        denom = outside + tf.reduce_sum(expu, axis=0, keepdims=True)

        s_j = tf.reduce_mean(expu / denom, axis=1)
        s0 = tf.reduce_mean(outside[0] / denom[0])

        return s_j, s0

    # ------------------------------------------------------------------
    # Log likelihood
    # ------------------------------------------------------------------

    def market_loglik(
        self,
        *,
        Ebar,
        eta,
        q,
        q0,
        x,
        Z,
        beta,
        r,
    ):
        """
        Market log likelihood contribution.

        Inputs:
          Ebar : scalar
          eta  : (J,)
          q    : (J,)
          q0   : scalar
          x    : (J,K)
          Z    : (J,d)
          beta : (K,)
          r    : (d,)

        Returns:
          scalar log likelihood
        """
        delta = tf.linalg.matvec(x, beta) + Ebar + eta
        s_j, s0 = self._choice_probs(Z, delta, r)

        ll = tf.reduce_sum(q * tf.math.log(s_j + self.eps))
        ll += q0 * tf.math.log(s0 + self.eps)

        return ll

    # ------------------------------------------------------------------
    # Log priors
    # ------------------------------------------------------------------

    def logprior_Ebar(self, Ebar):
        """
        Gaussian prior for Ebar.
        """
        return -0.5 * (Ebar * Ebar) / self.Ebar_var

    def logprior_eta(self, eta, gamma):
        """
        Spike-and-slab prior for eta | gamma.

        Inputs:
          eta   : (J,)
          gamma : (J,) in {0,1}

        Returns:
          scalar log prior
        """
        gamma = tf.cast(gamma, self.dtype)

        var = gamma * self.T1_sq + (1.0 - gamma) * self.T0_sq
        logvar = gamma * self.log_T1_sq + (1.0 - gamma) * self.log_T0_sq

        lp = -0.5 * tf.reduce_sum((eta * eta) / var)
        lp += -0.5 * tf.reduce_sum(logvar)

        return lp

    # ------------------------------------------------------------------
    # Convenience: full market log posterior
    # ------------------------------------------------------------------

    def market_logp(
        self,
        *,
        Ebar,
        eta,
        q,
        q0,
        x,
        Z,
        beta,
        r,
        gamma,
    ):
        """
        Full market log posterior contribution.

        This is a thin wrapper combining likelihood and priors.
        """
        return (
            self.market_loglik(
                Ebar=Ebar,
                eta=eta,
                q=q,
                q0=q0,
                x=x,
                Z=Z,
                beta=beta,
                r=r,
            )
            + self.logprior_Ebar(Ebar)
            + self.logprior_eta(eta, gamma)
        )
