"""
bonus2_estimator.py

RW-MH estimator for Bonus Q2 using TensorFlow.

Known model inputs (validated upstream):
  y_mit          (M,N,T) int32    choices; 0=outside, c=j+1 for inside product j
  delta_mj       (M,J)   float64  Phase-1 baseline utilities (fixed)
  is_weekend_t   (T,)    int32    indicator in {0,1}
  season_sin_kt  (K,T)   float64  seasonal basis
  season_cos_kt  (K,T)   float64  seasonal basis
  neighbors_m    neighbors_m[m][i] -> list[int] (within-market)
  lookback       scalar  int32    peer lookback window length L (known)
  decay          scalar  float64  habit decay in (0,1) (known)

Unconstrained parameter blocks z (float64):
  z_beta_intercept_j  (J,)
  z_beta_habit_j      (J,)
  z_beta_peer_j       (J,)
  z_beta_weekend_jw   (J,2)
  z_a_m               (M,K)
  z_b_m               (M,K)

Config-driven inputs (validated upstream):
  init_theta:
    beta_intercept, beta_habit, beta_peer, beta_weekend_weekday, beta_weekend_weekend, a_m, b_m
  sigmas:
    per-block prior scales keyed by z keys
  step_size_z:
    per-block RW proposal scales keyed by z keys
  seed:
    RNG seed (int)

This file performs no input validation. All validation must occur in bonus2_input_validation.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from bonus2 import bonus2_model as model
from bonus2.bonus2_diagnostics import report_iteration_progress
from bonus2.bonus2_input_validation import (
    validate_bonus2_estimator_fit_inputs,
    validate_bonus2_estimator_init_inputs,
)
from bonus2.bonus2_posterior import PosteriorInputs
from bonus2.bonus2_updates import update_z_block

Z_KEYS: tuple[str, ...] = (
    "z_beta_intercept_j",
    "z_beta_habit_j",
    "z_beta_peer_j",
    "z_beta_weekend_jw",
    "z_a_m",
    "z_b_m",
)


class Bonus2Estimator:
    """Estimate Bonus Q2 parameters using random-walk Metropolis-Hastings."""

    def __init__(
        self,
        y_mit: Any,
        delta_mj: Any,
        is_weekend_t: Any,
        season_sin_kt: Any,
        season_cos_kt: Any,
        neighbors_m: Any,
        lookback: int,
        decay: float,
        init_theta: dict[str, float],
        sigmas: dict[str, float],
        step_size_z: dict[str, float],
        seed: int,
    ) -> None:
        validate_bonus2_estimator_init_inputs(
            y_mit=y_mit,
            delta_mj=delta_mj,
            is_weekend_t=is_weekend_t,
            season_sin_kt=season_sin_kt,
            season_cos_kt=season_cos_kt,
            neighbors_m=neighbors_m,
            lookback=lookback,
            decay=decay,
            init_theta=init_theta,
            sigmas=sigmas,
            step_size_z=step_size_z,
            seed=seed,
        )

        y_np = np.asarray(y_mit, dtype=np.int32)
        delta_np = np.asarray(delta_mj, dtype=np.float64)
        w_np = np.asarray(is_weekend_t, dtype=np.int32)
        sin_np = np.asarray(season_sin_kt, dtype=np.float64)
        cos_np = np.asarray(season_cos_kt, dtype=np.float64)

        self.M, self.N, self.T = (int(x) for x in y_np.shape)
        self.J = int(delta_np.shape[1])
        self.K = int(sin_np.shape[0])

        self.y_mit = tf.convert_to_tensor(y_np, dtype=tf.int32)  # (M,N,T)
        self.delta_mj = tf.convert_to_tensor(delta_np, dtype=tf.float64)  # (M,J)
        self.is_weekend_t = tf.convert_to_tensor(w_np, dtype=tf.int32)  # (T,)
        self.season_sin_kt = tf.convert_to_tensor(sin_np, dtype=tf.float64)  # (K,T)
        self.season_cos_kt = tf.convert_to_tensor(cos_np, dtype=tf.float64)  # (K,T)

        self.lookback = tf.convert_to_tensor(int(lookback), dtype=tf.int32)
        self.decay = tf.convert_to_tensor(float(decay), dtype=tf.float64)

        self.peer_adj_m = model.build_peer_adjacency(
            neighbors_m=neighbors_m,
            n_consumers=self.N,
        )

        self.inputs: PosteriorInputs = {
            "y_mit": self.y_mit,
            "delta_mj": self.delta_mj,
            "is_weekend_t": self.is_weekend_t,
            "season_sin_kt": self.season_sin_kt,
            "season_cos_kt": self.season_cos_kt,
            "peer_adj_m": self.peer_adj_m,
            "lookback": self.lookback,
            "decay": self.decay,
        }

        self.sigma_z: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(sigmas[k]), dtype=tf.float64) for k in Z_KEYS
        }
        self.step_size_z: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(step_size_z[k]), dtype=tf.float64)
            for k in Z_KEYS
        }

        self.rng = tf.random.Generator.from_seed(int(seed))

        self.z: dict[str, tf.Variable] = self._init_z_from_theta(init_theta)
        self.theta_init = self._theta_from_z(self._current_z_dict(), as_numpy=True)

        self.accept: dict[str, tf.Variable] = {
            k: tf.Variable(0, dtype=tf.int32, trainable=False) for k in Z_KEYS
        }
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)

    def fit(self, n_iter: int) -> None:
        """Run MCMC for n_iter sweeps."""
        validate_bonus2_estimator_fit_inputs(n_iter=n_iter)

        self._reset_chain_state()

        for it in range(int(n_iter)):
            it_tf = tf.constant(it, dtype=tf.int32)
            self._mcmc_iteration_step(it=it_tf)
            report_iteration_progress(z=self._current_z_dict(), it=it_tf)

    def get_results(self) -> dict[str, object]:
        """Return theta_init, theta_hat (last draw), n_saved, accept rates per z block."""
        theta_last = self._theta_from_z(self._current_z_dict(), as_numpy=False)
        theta_hat = {k: v.numpy() for k, v in theta_last.items()}

        n_saved = int(self.saved.numpy())
        counts = {k: int(v.numpy()) for k, v in self.accept.items()}
        denom = max(1, n_saved)
        rates = {k: counts[k] / denom for k in counts.keys()}

        return {
            "theta_init": self.theta_init,
            "theta_hat": theta_hat,
            "n_saved": n_saved,
            "accept": rates,
        }

    def _init_z_from_theta(
        self, init_theta: dict[str, float]
    ) -> dict[str, tf.Variable]:
        """Initialize unconstrained blocks z by scalar fills from init_theta."""
        beta_intercept0 = tf.convert_to_tensor(
            float(init_theta["beta_intercept"]), tf.float64
        )
        beta_habit0 = tf.convert_to_tensor(float(init_theta["beta_habit"]), tf.float64)
        beta_peer0 = tf.convert_to_tensor(float(init_theta["beta_peer"]), tf.float64)

        beta_weekday0 = tf.convert_to_tensor(
            float(init_theta["beta_weekend_weekday"]), tf.float64
        )
        beta_weekend0 = tf.convert_to_tensor(
            float(init_theta["beta_weekend_weekend"]), tf.float64
        )

        a_m0 = tf.convert_to_tensor(float(init_theta["a_m"]), tf.float64)
        b_m0 = tf.convert_to_tensor(float(init_theta["b_m"]), tf.float64)

        z_beta_intercept_j0 = tf.fill((self.J,), beta_intercept0)
        z_beta_habit_j0 = tf.fill((self.J,), beta_habit0)
        z_beta_peer_j0 = tf.fill((self.J,), beta_peer0)

        z_beta_weekend_jw0 = tf.stack(
            [
                tf.fill((self.J,), beta_weekday0),
                tf.fill((self.J,), beta_weekend0),
            ],
            axis=1,
        )  # (J,2)

        z_a_m0 = tf.fill((self.M, self.K), a_m0)
        z_b_m0 = tf.fill((self.M, self.K), b_m0)

        return {
            "z_beta_intercept_j": tf.Variable(
                z_beta_intercept_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_habit_j": tf.Variable(
                z_beta_habit_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_peer_j": tf.Variable(
                z_beta_peer_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_weekend_jw": tf.Variable(
                z_beta_weekend_jw0, trainable=False, dtype=tf.float64
            ),
            "z_a_m": tf.Variable(z_a_m0, trainable=False, dtype=tf.float64),
            "z_b_m": tf.Variable(z_b_m0, trainable=False, dtype=tf.float64),
        }

    def _current_z_dict(self) -> dict[str, tf.Tensor]:
        """Return the current unconstrained blocks as tensors."""
        return {k: self.z[k] for k in Z_KEYS}

    def _theta_from_z(
        self, z_dict: dict[str, tf.Tensor], as_numpy: bool
    ) -> dict[str, Any]:
        """Map unconstrained z blocks to structural theta tensors."""
        theta = model.unconstrained_to_theta(z_dict)
        if as_numpy:
            return {k: v.numpy() for k, v in theta.items()}
        return theta

    def _reset_chain_state(self) -> None:
        """Reset acceptance counters and saved sweep counter."""
        for v in self.accept.values():
            v.assign(0)
        self.saved.assign(0)

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it: tf.Tensor) -> None:
        """Run one full MH sweep over all z blocks (compiled graph)."""
        for z_key in Z_KEYS:
            z_dict = self._current_z_dict()
            z_block_next, accepted = update_z_block(
                z=z_dict,
                z_key=z_key,
                inputs=self.inputs,
                step_size_z=self.step_size_z,
                sigma_z=self.sigma_z,
                rng=self.rng,
            )
            self.z[z_key].assign(z_block_next)
            self.accept[z_key].assign_add(accepted)

        self.saved.assign_add(1)
