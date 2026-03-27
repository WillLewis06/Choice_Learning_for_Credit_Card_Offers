"""
bonus2_posterior.py

Posterior evaluation for the Bonus Q2 model.

Design
- Fixed observed tensors and precomputed deterministic states are stored once on the
  posterior object.
- Repeated posterior evaluations consume only the current unconstrained parameter state.
- Priors are independent elementwise Normal(0, sigma^2) on each z block.
- Explicit compiled block log-posteriors are exposed for TFP-compatible block updates.

This module performs no input validation.
All tensors are assumed to have already been validated and normalized upstream.
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


class Bonus2State(TypedDict):
    """Unconstrained sampler state for the Bonus Q2 model."""

    z_beta_intercept_j: tf.Tensor
    z_beta_habit_j: tf.Tensor
    z_beta_peer_j: tf.Tensor
    z_beta_weekend_jw: tf.Tensor
    z_a_m: tf.Tensor
    z_b_m: tf.Tensor


Z_KEYS: tuple[str, ...] = (
    "z_beta_intercept_j",
    "z_beta_habit_j",
    "z_beta_peer_j",
    "z_beta_weekend_jw",
    "z_a_m",
    "z_b_m",
)


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
    """Evaluate posterior terms used by the Bonus Q2 sampler."""

    def __init__(
        self,
        config: Bonus2PosteriorConfig,
        inputs: Bonus2PosteriorInputs,
    ):
        """Cache fixed observed tensors and repeated prior constants."""

        # Fixed observed data and deterministic state tensors.
        self.y_mit = tf.convert_to_tensor(inputs["y_mit"], dtype=tf.int32)
        self.delta_mj = tf.convert_to_tensor(inputs["delta_mj"], dtype=tf.float64)
        self.is_weekend_t = tf.convert_to_tensor(inputs["is_weekend_t"], dtype=tf.int32)
        self.season_sin_kt = tf.convert_to_tensor(
            inputs["season_sin_kt"], dtype=tf.float64
        )
        self.season_cos_kt = tf.convert_to_tensor(
            inputs["season_cos_kt"], dtype=tf.float64
        )
        self.h_mntj = tf.convert_to_tensor(inputs["h_mntj"], dtype=tf.float64)
        self.p_mntj = tf.convert_to_tensor(inputs["p_mntj"], dtype=tf.float64)

        # Shared Normal prior constants.
        self._log_two_pi = tf.math.log(
            tf.constant(2.0 * 3.141592653589793, dtype=tf.float64)
        )

        # Prior scales and cached derived constants for each block.
        self.sigma_z_beta_intercept_j = tf.constant(
            config.sigma_z_beta_intercept_j, dtype=tf.float64
        )
        self.sigma_z_beta_habit_j = tf.constant(
            config.sigma_z_beta_habit_j, dtype=tf.float64
        )
        self.sigma_z_beta_peer_j = tf.constant(
            config.sigma_z_beta_peer_j, dtype=tf.float64
        )
        self.sigma_z_beta_weekend_jw = tf.constant(
            config.sigma_z_beta_weekend_jw, dtype=tf.float64
        )
        self.sigma_z_a_m = tf.constant(config.sigma_z_a_m, dtype=tf.float64)
        self.sigma_z_b_m = tf.constant(config.sigma_z_b_m, dtype=tf.float64)

        self.var_z_beta_intercept_j = tf.square(self.sigma_z_beta_intercept_j)
        self.var_z_beta_habit_j = tf.square(self.sigma_z_beta_habit_j)
        self.var_z_beta_peer_j = tf.square(self.sigma_z_beta_peer_j)
        self.var_z_beta_weekend_jw = tf.square(self.sigma_z_beta_weekend_jw)
        self.var_z_a_m = tf.square(self.sigma_z_a_m)
        self.var_z_b_m = tf.square(self.sigma_z_b_m)

        self.log_var_z_beta_intercept_j = tf.math.log(self.var_z_beta_intercept_j)
        self.log_var_z_beta_habit_j = tf.math.log(self.var_z_beta_habit_j)
        self.log_var_z_beta_peer_j = tf.math.log(self.var_z_beta_peer_j)
        self.log_var_z_beta_weekend_jw = tf.math.log(self.var_z_beta_weekend_jw)
        self.log_var_z_a_m = tf.math.log(self.var_z_a_m)
        self.log_var_z_b_m = tf.math.log(self.var_z_b_m)

        self.lp0_z_beta_intercept_j = -0.5 * (
            self._log_two_pi + self.log_var_z_beta_intercept_j
        )
        self.lp0_z_beta_habit_j = -0.5 * (
            self._log_two_pi + self.log_var_z_beta_habit_j
        )
        self.lp0_z_beta_peer_j = -0.5 * (self._log_two_pi + self.log_var_z_beta_peer_j)
        self.lp0_z_beta_weekend_jw = -0.5 * (
            self._log_two_pi + self.log_var_z_beta_weekend_jw
        )
        self.lp0_z_a_m = -0.5 * (self._log_two_pi + self.log_var_z_a_m)
        self.lp0_z_b_m = -0.5 * (self._log_two_pi + self.log_var_z_b_m)

    def _state_as_dict(
        self,
        z_beta_intercept_j: tf.Tensor,
        z_beta_habit_j: tf.Tensor,
        z_beta_peer_j: tf.Tensor,
        z_beta_weekend_jw: tf.Tensor,
        z_a_m: tf.Tensor,
        z_b_m: tf.Tensor,
    ) -> Bonus2State:
        """Package the unconstrained state into the canonical z dictionary."""
        return {
            "z_beta_intercept_j": z_beta_intercept_j,
            "z_beta_habit_j": z_beta_habit_j,
            "z_beta_peer_j": z_beta_peer_j,
            "z_beta_weekend_jw": z_beta_weekend_jw,
            "z_a_m": z_a_m,
            "z_b_m": z_b_m,
        }

    def _logprior_normal_sum(
        self,
        z_block: tf.Tensor,
        var: tf.Tensor,
        lp0: tf.Tensor,
    ) -> tf.Tensor:
        """Return the summed elementwise Normal(0, var) log prior."""
        return tf.reduce_sum(lp0 - 0.5 * tf.square(z_block) / var)

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
        z = self._state_as_dict(
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
        )
        theta = model.unconstrained_to_theta(z)

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
        return self._logprior_normal_sum(
            z_block=z_beta_intercept_j,
            var=self.var_z_beta_intercept_j,
            lp0=self.lp0_z_beta_intercept_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_habit(
        self,
        z_beta_habit_j: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_habit_j."""
        return self._logprior_normal_sum(
            z_block=z_beta_habit_j,
            var=self.var_z_beta_habit_j,
            lp0=self.lp0_z_beta_habit_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_peer(
        self,
        z_beta_peer_j: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_peer_j."""
        return self._logprior_normal_sum(
            z_block=z_beta_peer_j,
            var=self.var_z_beta_peer_j,
            lp0=self.lp0_z_beta_peer_j,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta_weekend(
        self,
        z_beta_weekend_jw: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_beta_weekend_jw."""
        return self._logprior_normal_sum(
            z_block=z_beta_weekend_jw,
            var=self.var_z_beta_weekend_jw,
            lp0=self.lp0_z_beta_weekend_jw,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_a(
        self,
        z_a_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_a_m."""
        return self._logprior_normal_sum(
            z_block=z_a_m,
            var=self.var_z_a_m,
            lp0=self.lp0_z_a_m,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_b(
        self,
        z_b_m: tf.Tensor,
    ) -> tf.Tensor:
        """Return the prior contribution for z_b_m."""
        return self._logprior_normal_sum(
            z_block=z_b_m,
            var=self.var_z_b_m,
            lp0=self.lp0_z_b_m,
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
        """Return the full joint prior over all six parameter blocks."""
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
