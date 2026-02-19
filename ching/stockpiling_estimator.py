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
from ching.stockpiling_input_validation import (
    normalize_stockpiling_estimator_fit_inputs,
    normalize_stockpiling_estimator_init_inputs,
    normalize_stockpiling_estimator_init_theta,
)
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
        norm = normalize_stockpiling_estimator_init_inputs(
            a_mnjt=a_mnjt,
            p_state_mjt=p_state_mjt,
            u_mj=u_mj,
            price_vals_mj=price_vals_mj,
            P_price_mj=P_price_mj,
            lambda_mn=lambda_mn,
            I_max=I_max,
            pi_I0=pi_I0,
            waste_cost=waste_cost,
            tol=tol,
            max_iter=max_iter,
            sigmas=dict(sigmas),
            rng_seed=rng_seed,
        )

        self.M = int(norm["M"])
        self.N = int(norm["N"])
        self.J = int(norm["J"])
        self.T = int(norm["T"])
        self.S = int(norm["S"])
        self.I_max = int(norm["I_max"])

        self.rng = tf.random.Generator.from_seed(int(norm["rng_seed"]))
        inventory_maps = model.build_inventory_maps(self.I_max)

        # Fixed inputs passed into the posterior/likelihood (dict-like mapping).
        # Do NOT attach any current parameter state (no "z" field).
        self.inputs: StockpilingInputs = {
            "a_mnjt": tf.convert_to_tensor(norm["a_mnjt"], dtype=tf.int32),
            "s_mjt": tf.convert_to_tensor(norm["p_state_mjt"], dtype=tf.int32),
            "u_mj": tf.convert_to_tensor(norm["u_mj"], dtype=tf.float64),
            "P_price_mj": tf.convert_to_tensor(norm["P_price_mj"], dtype=tf.float64),
            "price_vals_mj": tf.convert_to_tensor(
                norm["price_vals_mj"], dtype=tf.float64
            ),
            "lambda_mn": tf.convert_to_tensor(norm["lambda_mn"], dtype=tf.float64),
            "waste_cost": tf.constant(float(norm["waste_cost"]), dtype=tf.float64),
            "tol": float(norm["tol"]),
            "max_iter": int(norm["max_iter"]),
            "init_I_dist": tf.convert_to_tensor(norm["pi_I0"], dtype=tf.float64),
            "inventory_maps": inventory_maps,
        }

        sig = norm["sigmas"]
        self.sigma_z: dict[str, tf.Tensor] = {
            "z_beta": tf.constant(float(sig["z_beta"]), dtype=tf.float64),
            "z_alpha": tf.constant(float(sig["z_alpha"]), dtype=tf.float64),
            "z_v": tf.constant(float(sig["z_v"]), dtype=tf.float64),
            "z_fc": tf.constant(float(sig["z_fc"]), dtype=tf.float64),
            "z_u_scale": tf.constant(float(sig["z_u_scale"]), dtype=tf.float64),
        }

        self.z: dict[str, tf.Variable] = {}
        self.k: dict[str, tf.Tensor] = {}
        self.accept: dict[str, tf.Variable] = {}
        self.sum_theta: dict[str, tf.Variable] = {}
        self.saved = tf.Variable(0, dtype=tf.int32)

    def _reset_chain_state(
        self, init_theta: Mapping[str, Any], k: Mapping[str, float]
    ) -> None:
        """Initialize z, proposal scales, acceptance counters, and running sums."""
        self.k = {
            "beta": tf.constant(float(k["beta"]), dtype=tf.float64),
            "alpha": tf.constant(float(k["alpha"]), dtype=tf.float64),
            "v": tf.constant(float(k["v"]), dtype=tf.float64),
            "fc": tf.constant(float(k["fc"]), dtype=tf.float64),
            "u_scale": tf.constant(float(k["u_scale"]), dtype=tf.float64),
        }

        beta0 = float(init_theta["beta"])
        alpha0 = tf.convert_to_tensor(init_theta["alpha"], dtype=tf.float64)
        v0 = tf.convert_to_tensor(init_theta["v"], dtype=tf.float64)
        fc0 = tf.convert_to_tensor(init_theta["fc"], dtype=tf.float64)
        u_scale0 = tf.convert_to_tensor(init_theta["u_scale"], dtype=tf.float64)

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
        fit_norm = normalize_stockpiling_estimator_fit_inputs(n_iter=n_iter, k=dict(k))
        n_iter_n = int(fit_norm["n_iter"])
        k_norm = fit_norm["k"]

        init_theta_norm = normalize_stockpiling_estimator_init_theta(
            init_theta=dict(init_theta),
            M=self.M,
            J=self.J,
        )

        self._reset_chain_state(init_theta=init_theta_norm, k=k_norm)

        for it in range(n_iter_n):
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
