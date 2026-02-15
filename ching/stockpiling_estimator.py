"""
ching/stockpiling_estimator.py

MCMC estimator for the Phase-3 stockpiling (Ching-style) model.

Observed inputs:
  a_mnjt:        (M, N, J, T)  purchase indicator a ∈ {0,1}
  p_state_mjt:   (M, J, T)     observed price state s ∈ {0,...,S-1}
  u_mj:          (M, J)        fixed Phase-1/2 intercept u_mj = δ_j + Ē_m + n_mj
  lambda_mn:     (M, N)        known consumption probability (passed, not inferred)

Known price process:
  P_price_mj:    (M, J, S, S)  Markov transition matrix over price states
  price_vals_mj: (M, J, S)     price level for each state
  pi_I0:         (I_max+1,)    initial inventory distribution (shared across m,n,j)

Estimated parameters θ:
  beta:      (1,)   discount factor in (0,1), shared across markets/products
  alpha:     (J,)   price sensitivity > 0
  v:         (J,)   stockout penalty > 0
  fc:        (J,)   fixed purchase cost > 0
  u_scale:   (M,)   nuisance scale on u_mj during estimation > 0

Unconstrained parameters z (RW-MH on z-space):
  z_beta:    (1,)
  z_alpha:   (J,)
  z_v:       (J,)
  z_fc:      (J,)
  z_u_scale: (M,)

Notes:
  - lambda_mn is treated as known input data and is not estimated.
  - u_scale can be "frozen" for testing by setting k["u_scale"] = 0.0 (skips the update).
  - `self.inputs` holds only fixed data tensors (no current parameter state).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from ching import stockpiling_model as model
from ching.stockpiling_diagnostics import report_iteration_progress
from ching.stockpiling_posterior import StockpilingInputs
from ching.stockpiling_updates import (
    update_z_alpha_j,
    update_z_beta_scalar,
    update_z_fc_j,
    update_z_u_scale_m,
    update_z_v_j,
)


class StockpilingEstimator:
    """TensorFlow implementation of an MCMC estimator for the stockpiling model."""

    def __init__(
        self,
        a_mnjt: np.ndarray,
        p_state_mjt: np.ndarray,
        u_mj: np.ndarray,
        lambda_mn: np.ndarray,
        P_price_mj: np.ndarray,
        price_vals_mj: np.ndarray,
        pi_I0: np.ndarray,
        I_max: int,
        waste_cost: float,
        sigmas: Mapping[str, float],
        tol: float = 1e-8,
        max_iter: int = 10_000,
        rng_seed: int = 0,
    ) -> None:
        """
        Args:
          a_mnjt: Purchase indicators, shape (M,N,J,T).
          p_state_mjt: Price states, shape (M,J,T), integer in {0,...,S-1}.
          u_mj: Fixed intercept from Phase 1–2, shape (M,J).
          lambda_mn: Known consumption probability, shape (M,N), in (0,1).
          P_price_mj: Price Markov transitions, shape (M,J,S,S).
          price_vals_mj: Price levels for each state, shape (M,J,S).
          pi_I0: Initial inventory distribution, shape (I_max+1,).
          I_max: Maximum inventory units.
          waste_cost: Waste penalty coefficient in the buy utility.
          sigmas: Prior scales on z-space blocks. Required keys:
            {"z_beta","z_alpha","z_v","z_fc","z_u_scale"}.
          tol: Convergence tolerance for the CCP solver.
          max_iter: Maximum iterations for the CCP solver.
          rng_seed: RNG seed for MCMC proposals.
        """
        a_mnjt = np.asarray(a_mnjt, dtype=np.int32)
        p_state_mjt = np.asarray(p_state_mjt, dtype=np.int32)
        u_mj = np.asarray(u_mj, dtype=np.float64)
        lambda_mn = np.asarray(lambda_mn, dtype=np.float64)
        P_price_mj = np.asarray(P_price_mj, dtype=np.float64)
        price_vals_mj = np.asarray(price_vals_mj, dtype=np.float64)
        pi_I0 = np.asarray(pi_I0, dtype=np.float64)

        M, N, J, T = a_mnjt.shape
        S = int(price_vals_mj.shape[2])

        self.M, self.N, self.J, self.T, self.S = int(M), int(N), int(J), int(T), int(S)
        self.I_max = int(I_max)

        self.rng = tf.random.Generator.from_seed(int(rng_seed))
        inventory_maps = model.build_inventory_maps(self.I_max)

        # Fixed inputs passed into the posterior/likelihood (dict-like mapping).
        # Do NOT attach any current parameter state (no "z" field).
        self.inputs: StockpilingInputs = {
            "a_mnjt": tf.convert_to_tensor(a_mnjt, dtype=tf.int32),
            "s_mjt": tf.convert_to_tensor(p_state_mjt, dtype=tf.int32),
            "u_mj": tf.convert_to_tensor(u_mj, dtype=tf.float64),
            "P_price_mj": tf.convert_to_tensor(P_price_mj, dtype=tf.float64),
            "price_vals_mj": tf.convert_to_tensor(price_vals_mj, dtype=tf.float64),
            "lambda_mn": tf.convert_to_tensor(lambda_mn, dtype=tf.float64),
            "waste_cost": tf.constant(float(waste_cost), dtype=tf.float64),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "init_I_dist": tf.convert_to_tensor(pi_I0, dtype=tf.float64),
            "inventory_maps": inventory_maps,
        }

        self.sigma_z: dict[str, tf.Tensor] = {
            "z_beta": tf.constant(float(sigmas["z_beta"]), dtype=tf.float64),
            "z_alpha": tf.constant(float(sigmas["z_alpha"]), dtype=tf.float64),
            "z_v": tf.constant(float(sigmas["z_v"]), dtype=tf.float64),
            "z_fc": tf.constant(float(sigmas["z_fc"]), dtype=tf.float64),
            "z_u_scale": tf.constant(float(sigmas["z_u_scale"]), dtype=tf.float64),
        }

        self.z: dict[str, tf.Variable] = {}
        self.k: dict[str, tf.Tensor] = {}
        self.accept: dict[str, tf.Variable] = {}
        self.sum_theta: dict[str, tf.Variable] = {}
        self.saved = tf.Variable(0, dtype=tf.int32)

    def _as_vec(self, x: Any, length: int, name: str, min_value: float) -> tf.Tensor:
        """
        Convert x to a float64 tensor of shape (length,).

        If x is scalar, broadcasts to (length,).
        If x is shape (length,), uses it directly.
        """
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 0:
            a = np.full((length,), float(a), dtype=np.float64)
        if a.shape != (length,):
            raise ValueError(
                f"{name}: expected scalar or shape ({length},), got {a.shape}"
            )
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: expected finite values, got non-finite")
        if (a <= min_value).any():
            mn = float(a.min()) if a.size else float("nan")
            raise ValueError(f"{name}: expected all > {min_value}, got min={mn}")
        return tf.convert_to_tensor(a, dtype=tf.float64)

    def _reset_chain_state(
        self, init_theta: Mapping[str, Any], k: Mapping[str, float]
    ) -> None:
        """Initialize z, proposal scales, acceptance counters, and running sums."""
        required_k = {"beta", "alpha", "v", "fc", "u_scale"}
        missing_k = required_k - set(k.keys())
        if missing_k:
            raise ValueError(f"k: missing keys {sorted(missing_k)}")

        self.k = {
            "beta": tf.constant(float(k["beta"]), dtype=tf.float64),
            "alpha": tf.constant(float(k["alpha"]), dtype=tf.float64),
            "v": tf.constant(float(k["v"]), dtype=tf.float64),
            "fc": tf.constant(float(k["fc"]), dtype=tf.float64),
            "u_scale": tf.constant(float(k["u_scale"]), dtype=tf.float64),
        }

        beta0 = float(init_theta["beta"])
        if not np.isfinite(beta0) or beta0 <= 0.0 or beta0 >= 1.0:
            raise ValueError(f"init_theta['beta']: expected in (0,1), got {beta0}")

        alpha0 = self._as_vec(init_theta["alpha"], self.J, "init_theta['alpha']", 0.0)
        v0 = self._as_vec(init_theta["v"], self.J, "init_theta['v']", 0.0)
        fc0 = self._as_vec(init_theta["fc"], self.J, "init_theta['fc']", 0.0)
        u_scale0 = self._as_vec(
            init_theta["u_scale"], self.M, "init_theta['u_scale']", 0.0
        )

        beta0_tf = tf.constant(beta0, dtype=tf.float64)
        z_beta0 = tf.reshape(tf.math.log(beta0_tf / (1.0 - beta0_tf)), (1,))
        z_alpha0 = tf.math.log(alpha0)
        z_v0 = tf.math.log(v0)
        z_fc0 = tf.math.log(fc0)
        z_u_scale0 = tf.math.log(u_scale0)

        self.z = {
            "z_beta": tf.Variable(z_beta0, dtype=tf.float64),
            "z_alpha": tf.Variable(z_alpha0, dtype=tf.float64),
            "z_v": tf.Variable(z_v0, dtype=tf.float64),
            "z_fc": tf.Variable(z_fc0, dtype=tf.float64),
            "z_u_scale": tf.Variable(z_u_scale0, dtype=tf.float64),
        }

        self.accept = {
            "beta": tf.Variable(0, dtype=tf.int32),
            "alpha": tf.Variable(0, dtype=tf.int32),
            "v": tf.Variable(0, dtype=tf.int32),
            "fc": tf.Variable(0, dtype=tf.int32),
            "u_scale": tf.Variable(0, dtype=tf.int32),
        }

        self.sum_theta = {
            "beta": tf.Variable(tf.zeros((1,), dtype=tf.float64)),
            "alpha": tf.Variable(tf.zeros((self.J,), dtype=tf.float64)),
            "v": tf.Variable(tf.zeros((self.J,), dtype=tf.float64)),
            "fc": tf.Variable(tf.zeros((self.J,), dtype=tf.float64)),
            "u_scale": tf.Variable(tf.zeros((self.M,), dtype=tf.float64)),
        }
        self.saved.assign(0)

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it: tf.Tensor) -> None:
        """Run one MCMC iteration (sequential block updates + running sums)."""
        z_beta_new, acc_beta = update_z_beta_scalar(
            self.z["z_beta"],
            self.z["z_alpha"],
            self.z["z_v"],
            self.z["z_fc"],
            self.z["z_u_scale"],
            self.inputs,
            self.sigma_z["z_beta"],
            self.k["beta"],
            self.rng,
        )
        self.z["z_beta"].assign(z_beta_new)
        self.accept["beta"].assign_add(tf.reduce_sum(tf.cast(acc_beta, tf.int32)))

        z_alpha_new, acc_alpha = update_z_alpha_j(
            self.z["z_beta"],
            self.z["z_alpha"],
            self.z["z_v"],
            self.z["z_fc"],
            self.z["z_u_scale"],
            self.inputs,
            self.sigma_z["z_alpha"],
            self.k["alpha"],
            self.rng,
        )
        self.z["z_alpha"].assign(z_alpha_new)
        self.accept["alpha"].assign_add(tf.reduce_sum(tf.cast(acc_alpha, tf.int32)))

        z_v_new, acc_v = update_z_v_j(
            self.z["z_beta"],
            self.z["z_alpha"],
            self.z["z_v"],
            self.z["z_fc"],
            self.z["z_u_scale"],
            self.inputs,
            self.sigma_z["z_v"],
            self.k["v"],
            self.rng,
        )
        self.z["z_v"].assign(z_v_new)
        self.accept["v"].assign_add(tf.reduce_sum(tf.cast(acc_v, tf.int32)))

        z_fc_new, acc_fc = update_z_fc_j(
            self.z["z_beta"],
            self.z["z_alpha"],
            self.z["z_v"],
            self.z["z_fc"],
            self.z["z_u_scale"],
            self.inputs,
            self.sigma_z["z_fc"],
            self.k["fc"],
            self.rng,
        )
        self.z["z_fc"].assign(z_fc_new)
        self.accept["fc"].assign_add(tf.reduce_sum(tf.cast(acc_fc, tf.int32)))

        def _do_u_scale_update() -> tuple[tf.Tensor, tf.Tensor]:
            return update_z_u_scale_m(
                self.z["z_beta"],
                self.z["z_alpha"],
                self.z["z_v"],
                self.z["z_fc"],
                self.z["z_u_scale"],
                self.inputs,
                self.sigma_z["z_u_scale"],
                self.k["u_scale"],
                self.rng,
            )

        def _skip_u_scale_update() -> tuple[tf.Tensor, tf.Tensor]:
            acc = tf.zeros_like(self.z["z_u_scale"], dtype=tf.bool)
            return self.z["z_u_scale"], acc

        z_u_new, acc_u = tf.cond(
            self.k["u_scale"] > 0.0, _do_u_scale_update, _skip_u_scale_update
        )
        self.z["z_u_scale"].assign(z_u_new)
        self.accept["u_scale"].assign_add(tf.reduce_sum(tf.cast(acc_u, tf.int32)))

        theta = model.unconstrained_to_theta(self.z)
        self.sum_theta["beta"].assign_add(theta["beta"])
        self.sum_theta["alpha"].assign_add(theta["alpha"])
        self.sum_theta["v"].assign_add(theta["v"])
        self.sum_theta["fc"].assign_add(theta["fc"])
        self.sum_theta["u_scale"].assign_add(theta["u_scale"])
        self.saved.assign_add(1)

        report_iteration_progress(self.z, it)

    def fit(
        self,
        n_iter: int,
        init_theta: Mapping[str, Any],
        k: Mapping[str, float],
    ) -> dict[str, Any]:
        """
        Run MCMC and return posterior means and per-block acceptance rates.

        Args:
          n_iter: Number of MCMC iterations (positive int).
          init_theta: Dict with keys {"beta","alpha","v","fc","u_scale"}.
            - beta: scalar in (0,1)
            - alpha, v, fc: scalar or shape (J,)
            - u_scale: scalar or shape (M,)
          k: Proposal step sizes with required keys {"beta","alpha","v","fc","u_scale"}.
            Setting k["u_scale"]=0.0 freezes the u_scale block.

        Returns:
          Dict with:
            - "theta_mean": posterior means on theta-space
            - "accept": per-block acceptance rates
            - "n_saved": number of saved draws
            - "z_last": final z-space state
        """
        if int(n_iter) < 1:
            raise ValueError(f"n_iter: expected >= 1, got {n_iter}")

        self._reset_chain_state(init_theta=init_theta, k=k)

        for it in range(int(n_iter)):
            self._mcmc_iteration_step(tf.constant(it, dtype=tf.int32))

        saved_f = tf.cast(tf.maximum(self.saved, 1), tf.float64)
        theta_mean = {key: val / saved_f for key, val in self.sum_theta.items()}

        block_sizes = {
            "beta": tf.cast(tf.size(self.z["z_beta"]), tf.float64),
            "alpha": tf.cast(tf.size(self.z["z_alpha"]), tf.float64),
            "v": tf.cast(tf.size(self.z["z_v"]), tf.float64),
            "fc": tf.cast(tf.size(self.z["z_fc"]), tf.float64),
            "u_scale": tf.cast(tf.size(self.z["z_u_scale"]), tf.float64),
        }
        accept = {
            key: tf.cast(self.accept[key], tf.float64) / (saved_f * block_sizes[key])
            for key in self.accept
        }

        return {
            "theta_mean": {k: v.numpy() for k, v in theta_mean.items()},
            "accept": {k: v.numpy() for k, v in accept.items()},
            "n_saved": int(self.saved.numpy()),
            "z_last": {k: v.numpy() for k, v in self.z.items()},
        }
