"""
TensorFlow posterior utilities for the Lu shrinkage estimator.

This module provides:
- RC-logit choice probabilities with fixed Monte Carlo draws,
- multinomial log-likelihood terms using observed counts,
- prior terms for global parameters and sparse market×product shocks,
- posterior composition in both per-market-vector and full-scalar forms.

All hyperparameters and numerical settings are required and must be supplied
via a validated config. This module does not implement input validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class LuPosteriorConfig:
    """Configuration container for LuPosteriorTF.

    All fields are required and should be validated upstream (e.g., config
    validation). This class does not apply defaults or type coercions.
    """

    # Monte Carlo integration settings
    n_draws: int
    seed: int

    # Numeric settings
    dtype: tf.dtypes.DType
    eps: float

    # Global priors: Normal(mean, var)
    beta_p_mean: float
    beta_p_var: float
    beta_w_mean: float
    beta_w_var: float
    r_mean: float
    r_var: float

    # Market common shock prior: Normal(mean, var)
    E_bar_mean: float
    E_bar_var: float

    # Spike-and-slab variances for njt
    T0_sq: float
    T1_sq: float

    # Beta prior for phi
    a_phi: float
    b_phi: float


class LuPosteriorTF:
    """Posterior building blocks for the Lu shrinkage model (TensorFlow).

    Utility in market t for product j:
        delta_jt = beta_p * pjt + beta_w * wjt + E_bar_t + njt

    Price heterogeneity is integrated via fixed draws:
        beta_p_i = beta_p + exp(r) * v_i,   v_i ~ Normal(0, 1)

    Sparse shocks use a continuous spike-and-slab mixture:
        gamma_tj | phi_t ~ Bernoulli(phi_t)
        njt_tj | gamma_tj = 1 ~ Normal(0, T1_sq)   (slab)
        njt_tj | gamma_tj = 0 ~ Normal(0, T0_sq)   (spike)
        phi_t ~ Beta(a_phi, b_phi)

    Likelihood uses multinomial counts and omits the combinatorial constant.
    """

    def __init__(self, config: LuPosteriorConfig):
        """Initialize constants and fixed Monte Carlo draws."""
        self.dtype = config.dtype
        self.eps = tf.constant(config.eps, dtype=self.dtype)

        # Fixed Monte Carlo draws v_i ~ Normal(0, 1), reused across all calls.
        g = tf.random.Generator.from_seed(config.seed)
        self.v_draws = g.normal(shape=(config.n_draws,), dtype=self.dtype)

        # Shared constants.
        two_pi = tf.constant(2.0 * 3.141592653589793, dtype=self.dtype)
        self._log_two_pi = tf.math.log(two_pi)

        # Prior hyperparameters as TF constants (graph-friendly).
        self.beta_p_mean = tf.constant(config.beta_p_mean, dtype=self.dtype)
        self.beta_p_var = tf.constant(config.beta_p_var, dtype=self.dtype)
        self.beta_w_mean = tf.constant(config.beta_w_mean, dtype=self.dtype)
        self.beta_w_var = tf.constant(config.beta_w_var, dtype=self.dtype)
        self.r_mean = tf.constant(config.r_mean, dtype=self.dtype)
        self.r_var = tf.constant(config.r_var, dtype=self.dtype)

        self.E_bar_mean = tf.constant(config.E_bar_mean, dtype=self.dtype)
        self.E_bar_var = tf.constant(config.E_bar_var, dtype=self.dtype)

        self.T0_sq = tf.constant(config.T0_sq, dtype=self.dtype)
        self.T1_sq = tf.constant(config.T1_sq, dtype=self.dtype)
        self._log_T0_sq = tf.math.log(self.T0_sq)
        self._log_T1_sq = tf.math.log(self.T1_sq)

        self.a_phi = tf.constant(config.a_phi, dtype=self.dtype)
        self.b_phi = tf.constant(config.b_phi, dtype=self.dtype)
        self._log_beta_ab = (
            tf.math.lgamma(self.a_phi)
            + tf.math.lgamma(self.b_phi)
            - tf.math.lgamma(self.a_phi + self.b_phi)
        )

        # Cached Normal log-density constants: -0.5 * log(2*pi*var).
        self._lp0_beta_p = -0.5 * (self._log_two_pi + tf.math.log(self.beta_p_var))
        self._lp0_beta_w = -0.5 * (self._log_two_pi + tf.math.log(self.beta_w_var))
        self._lp0_r = -0.5 * (self._log_two_pi + tf.math.log(self.r_var))
        self._lp0_E_bar = -0.5 * (self._log_two_pi + tf.math.log(self.E_bar_var))

    # ------------------------------------------------------------------
    # Choice probabilities (price-only random coefficient)
    # ------------------------------------------------------------------

    def _mean_utility_jt(
        self,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Mean utilities for one market (shape (J,))."""
        return beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t

    def _choice_probs_t(
        self, pjt_t: tf.Tensor, delta_t: tf.Tensor, r: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """RC-logit inside shares sjt (J,) and outside share s0 (scalar) for one market."""
        sigma = tf.exp(r)
        scaled_v = sigma * self.v_draws  # (R,)

        # Draw-specific utility: util[j, i] = delta[j] + p[j] * sigma * v[i].
        util = delta_t[:, None] + pjt_t[:, None] * scaled_v[None, :]  # (J, R)

        # Stabilize exp() using a max-shift that also accounts for outside option u0 = 0.
        m = tf.reduce_max(util, axis=0, keepdims=True)  # (1, R)
        m = tf.maximum(m, tf.zeros_like(m))  # ensures shift >= 0 so outside is included

        expu = tf.exp(util - m)  # (J, R)
        exp0 = tf.exp(-m)  # (1, R)
        denom = exp0 + tf.reduce_sum(expu, axis=0, keepdims=True)  # (1, R)

        sjt_draw = expu / denom  # (J, R)
        s0_draw = exp0 / denom  # (1, R)

        sjt_t = tf.reduce_mean(sjt_draw, axis=1)  # (J,)
        s0t = tf.reduce_mean(s0_draw[0, :])  # scalar
        return sjt_t, s0t

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def market_loglik(
        self,
        qjt_t: tf.Tensor,
        q0t_t: tf.Tensor,
        pjt_t: tf.Tensor,
        wjt_t: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar_t: tf.Tensor,
        njt_t: tf.Tensor,
    ) -> tf.Tensor:
        """Multinomial log-likelihood contribution for one market (scalar)."""
        delta_t = self._mean_utility_jt(
            pjt_t=pjt_t,
            wjt_t=wjt_t,
            beta_p=beta_p,
            beta_w=beta_w,
            E_bar_t=E_bar_t,
            njt_t=njt_t,
        )
        sjt_t, s0t = self._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

        # Numerical stability: prevent log(0) from underflowing shares to exactly 0.
        sjt_t = tf.clip_by_value(sjt_t, self.eps, 1.0)
        s0t = tf.clip_by_value(s0t, self.eps, 1.0)

        return tf.reduce_sum(qjt_t * tf.math.log(sjt_t)) + q0t_t * tf.math.log(s0t)

    @tf.function(reduce_retracing=True)
    def loglik_vec(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market log-likelihood vector (shape (T,))."""
        T = tf.shape(pjt)[0]

        def _market_ll(t: tf.Tensor) -> tf.Tensor:
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

        return tf.map_fn(_market_ll, tf.range(T), fn_output_signature=self.dtype)

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def logprior_global(
        self, beta_p: tf.Tensor, beta_w: tf.Tensor, r: tf.Tensor
    ) -> tf.Tensor:
        """Global prior for (beta_p, beta_w, r), scalar."""
        lp_beta_p = (
            self._lp0_beta_p
            - 0.5 * tf.square(beta_p - self.beta_p_mean) / self.beta_p_var
        )
        lp_beta_w = (
            self._lp0_beta_w
            - 0.5 * tf.square(beta_w - self.beta_w_mean) / self.beta_w_var
        )
        lp_r = self._lp0_r - 0.5 * tf.square(r - self.r_mean) / self.r_var
        return lp_beta_p + lp_beta_w + lp_r

    def logprior_market_vec(
        self,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market prior vector for (E_bar, njt, gamma, phi), shape (T,)."""
        # E_bar[t] ~ Normal(E_bar_mean, E_bar_var)
        lp_E = (
            self._lp0_E_bar - 0.5 * tf.square(E_bar - self.E_bar_mean) / self.E_bar_var
        )  # (T,)

        # njt[t, j] ~ Normal(0, var_tj) where var_tj selects spike vs slab via gamma.
        one_minus_gamma = 1.0 - gamma
        var = gamma * self.T1_sq + one_minus_gamma * self.T0_sq  # (T, J)
        log_var = gamma * self._log_T1_sq + one_minus_gamma * self._log_T0_sq  # (T, J)
        lp_n_entry = -0.5 * (
            self._log_two_pi + log_var + tf.square(njt) / var
        )  # (T, J)
        lp_n = tf.reduce_sum(lp_n_entry, axis=1)  # (T,)

        # gamma[t, j] | phi[t] ~ Bernoulli(phi[t]) (independent across j).
        phi = tf.clip_by_value(phi, self.eps, 1.0 - self.eps)  # numerical stability
        phi_b = phi[:, None]  # (T, 1) -> broadcast to (T, J)
        lp_g_entry = gamma * tf.math.log(phi_b) + one_minus_gamma * tf.math.log(
            1.0 - phi_b
        )  # (T, J)
        lp_g = tf.reduce_sum(lp_g_entry, axis=1)  # (T,)

        # phi[t] ~ Beta(a_phi, b_phi)
        lp_phi = (
            (self.a_phi - 1.0) * tf.math.log(phi)
            + (self.b_phi - 1.0) * tf.math.log(1.0 - phi)
            - self._log_beta_ab
        )  # (T,)

        return lp_E + lp_n + lp_g + lp_phi

    # ------------------------------------------------------------------
    # Posterior composition
    # ------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def logpost_vec(
        self,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Per-market posterior terms (likelihood + market priors), shape (T,)."""
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
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        beta_p: tf.Tensor,
        beta_w: tf.Tensor,
        r: tf.Tensor,
        E_bar: tf.Tensor,
        njt: tf.Tensor,
        gamma: tf.Tensor,
        phi: tf.Tensor,
    ) -> tf.Tensor:
        """Full scalar log posterior: sum_t logpost_vec[t] + global prior."""
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
