"""Posterior terms for the Bonus Q2 model.

This module caches fixed observed tensors and deterministic state tensors, then
provides compiled likelihood and posterior kernels for blockwise MCMC updates.
Input validation is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import tensorflow as tf

from bonus2 import bonus2_model as model


class Bonus2PosteriorInputs(TypedDict):
    """Fixed tensors required by the Bonus Q2 posterior."""

    y_mit: tf.Tensor
    delta_mj: tf.Tensor
    is_weekend_t: tf.Tensor
    season_sin_kt: tf.Tensor
    season_cos_kt: tf.Tensor
    h_mntj: tf.Tensor
    p_mntj: tf.Tensor


@dataclass(frozen=True)
class Bonus2PosteriorConfig:
    """Store fixed prior scales for posterior evaluation."""

    sigma_z_beta_intercept_j: float
    sigma_z_beta_habit_j: float
    sigma_z_beta_peer_j: float
    sigma_z_beta_weekend_jw: float
    sigma_z_a_m: float
    sigma_z_b_m: float


class Bonus2PosteriorTF:
    """Evaluate likelihood and posterior terms for the Bonus Q2 sampler."""

    def __init__(
        self,
        config: Bonus2PosteriorConfig,
        inputs: Bonus2PosteriorInputs,
    ):
        """Cache fixed tensors and prior constants."""
        self.y_mit = inputs["y_mit"]
        self.delta_mj = inputs["delta_mj"]
        self.is_weekend_t = inputs["is_weekend_t"]
        self.season_sin_kt = inputs["season_sin_kt"]
        self.season_cos_kt = inputs["season_cos_kt"]
        self.h_mntj = inputs["h_mntj"]
        self.p_mntj = inputs["p_mntj"]

        self.inv_var_z_beta_intercept_j = tf.constant(
            1.0 / (config.sigma_z_beta_intercept_j**2),
            dtype=tf.float64,
        )
        self.inv_var_z_beta_habit_j = tf.constant(
            1.0 / (config.sigma_z_beta_habit_j**2),
            dtype=tf.float64,
        )
        self.inv_var_z_beta_peer_j = tf.constant(
            1.0 / (config.sigma_z_beta_peer_j**2),
            dtype=tf.float64,
        )
        self.inv_var_z_beta_weekend_jw = tf.constant(
            1.0 / (config.sigma_z_beta_weekend_jw**2),
            dtype=tf.float64,
        )
        self.inv_var_z_a_m = tf.constant(
            1.0 / (config.sigma_z_a_m**2),
            dtype=tf.float64,
        )
        self.inv_var_z_b_m = tf.constant(
            1.0 / (config.sigma_z_b_m**2),
            dtype=tf.float64,
        )

    def _logprior_quadratic_sum(
        self,
        z_block: tf.Tensor,
        inv_var: tf.Tensor,
    ) -> tf.Tensor:
        """Return the Gaussian quadratic prior contribution."""
        return -0.5 * tf.reduce_sum(tf.square(z_block) * inv_var)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik_mnt(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return per-(market, consumer, time) log-likelihood contributions."""
        theta = {
            "beta_intercept_j": z_beta_intercept_j,
            "beta_habit_j": z_beta_habit_j,
            "beta_peer_j": z_beta_peer_j,
            "beta_weekend_jw": z_beta_weekend_jw,
            "a_m": z_a_m,
            "b_m": z_b_m,
        }

        return model.loglik_mnt_from_theta(
            theta=theta,
            y_mit=self.y_mit,
            delta_mj=self.delta_mj,
            is_weekend_t=self.is_weekend_t,
            season_sin_kt=self.season_sin_kt,
            season_cos_kt=self.season_cos_kt,
            h_mntj=self.h_mntj,
            p_mntj=self.p_mntj,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total log-likelihood."""
        return tf.reduce_sum(
            self.loglik_mnt(
                z_beta_intercept_j=z_beta_intercept_j,
                z_beta_habit_j=z_beta_habit_j,
                z_beta_peer_j=z_beta_peer_j,
                z_beta_weekend_jw=z_beta_weekend_jw,
                z_a_m=z_a_m,
                z_b_m=z_b_m,
            )
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_intercept(
        self,
        z_beta_intercept_j: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_intercept_j."""
        return self._logprior_quadratic_sum(
            z_block=z_beta_intercept_j,
            inv_var=self.inv_var_z_beta_intercept_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_habit(
        self,
        z_beta_habit_j: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_habit_j."""
        return self._logprior_quadratic_sum(
            z_block=z_beta_habit_j,
            inv_var=self.inv_var_z_beta_habit_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_peer(
        self,
        z_beta_peer_j: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_peer_j."""
        return self._logprior_quadratic_sum(
            z_block=z_beta_peer_j,
            inv_var=self.inv_var_z_beta_peer_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_weekend(
        self,
        z_beta_weekend_jw: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_weekend_jw."""
        return self._logprior_quadratic_sum(
            z_block=z_beta_weekend_jw,
            inv_var=self.inv_var_z_beta_weekend_jw,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_a(
        self,
        z_a_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_a_m."""
        return self._logprior_quadratic_sum(
            z_block=z_a_m,
            inv_var=self.inv_var_z_a_m,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_b(
        self,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_b_m."""
        return self._logprior_quadratic_sum(
            z_block=z_b_m,
            inv_var=self.inv_var_z_b_m,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_all(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the joint prior over all parameter blocks."""
        return (
            self.logprior_beta_intercept(z_beta_intercept_j=z_beta_intercept_j)
            + self.logprior_beta_habit(z_beta_habit_j=z_beta_habit_j)
            + self.logprior_beta_peer(z_beta_peer_j=z_beta_peer_j)
            + self.logprior_beta_weekend(z_beta_weekend_jw=z_beta_weekend_jw)
            + self.logprior_a(z_a_m=z_a_m)
            + self.logprior_b(z_b_m=z_b_m)
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_intercept_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_beta_intercept_j."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_beta_intercept(z_beta_intercept_j=z_beta_intercept_j)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_habit_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_beta_habit_j."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_beta_habit(z_beta_habit_j=z_beta_habit_j)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_peer_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_beta_peer_j."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_beta_peer(z_beta_peer_j=z_beta_peer_j)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_weekend_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_beta_weekend_jw."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_beta_weekend(z_beta_weekend_jw=z_beta_weekend_jw)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def a_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_a_m."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_a(z_a_m=z_a_m)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def b_block_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the block log posterior for z_b_m."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_b(z_b_m=z_b_m)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def joint_logpost(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the full joint log posterior."""
        return self.loglik(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        ) + self.logprior_all(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )
