"""
ching/stockpiling_estimator.py

Phase-3 (Ching-style) stockpiling estimator for the multi-product model.

This estimator:
  - assumes all input validation is handled upstream (validation module / caller)
  - runs elementwise RW-MH on unconstrained blocks z_* and accumulates posterior means
  - delegates all constraint transforms (z -> theta) to ching.stockpiling_model

Observed inputs:
  a_mnjt        (M,N,J,T)  actions {0,1}
  p_state_mjt   (M,J,T)    price states in {0,...,S-1}
  u_mj          (M,J)      fixed utilities from Phase 1–2
  price_vals_mj (M,J,S)    price levels by state
  P_price_mj    (M,J,S,S)  price transitions
  pi_I0         (I,)       initial inventory prior

Unconstrained blocks z:
  z_beta, z_alpha, z_v, z_fc : (M,J)
  z_lambda                   : (M,N)
  z_u_scale                  : (M,)

Public API:
  - fit(n_iter, k): k is a dict of RW step sizes keyed by:
      {"beta","alpha","v","fc","lambda","u_scale"}
  - get_results(): returns theta_hat, n_saved, accept (rates only)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.dtypes import int32

from ching import stockpiling_model as model

from ching.stockpiling_updates import (
    update_z_beta_alpha_fc_mj,
    update_z_lambda_mn,
    update_z_u_scale_m,
    update_z_v_mj,
)

from ching.stockpiling_diagnostics import report_iteration_progress


class StockpilingEstimator:
    """
    Estimate Phase-3 stockpiling parameters with elementwise RW-MH on z-parameters.

    Block shapes:
      - Market-product blocks (M,J): z_beta, z_alpha, z_v, z_fc
      - Market-consumer block (M,N): z_lambda
      - Market block (M,):          z_u_scale
    """

    def __init__(
        self,
        a_mnjt: Any,
        p_state_mjt: Any,
        u_mj: Any,
        price_vals_mj: Any,
        P_price_mj: Any,
        I_max: int,
        pi_I0: Any,
        waste_cost: float,
        eps: float,
        tol: float,
        max_iter: int,
        init_theta: dict[str, float],
        sigmas: dict[str, float],
        seed: int,
    ) -> None:
        # ---- infer dimensions (assumes inputs are consistent) ----
        a_np = np.asarray(a_mnjt)
        self.M, self.N, self.J, self.T = (int(x) for x in a_np.shape)

        pv_np = np.asarray(price_vals_mj, dtype=np.float64)
        self.S = int(pv_np.shape[2])

        # ---- convert to TF tensors (posterior expects these dtypes) ----
        self.a_mnjt = tf.convert_to_tensor(a_np, dtype=tf.int32)  # (M,N,J,T)

        self.p_state_mjt = tf.convert_to_tensor(
            np.asarray(p_state_mjt), dtype=tf.int32
        )  # (M,J,T)

        self.u_mj = tf.convert_to_tensor(
            np.asarray(u_mj, dtype=np.float64), dtype=tf.float64
        )  # (M,J)

        self.price_vals_mj = tf.convert_to_tensor(pv_np, dtype=tf.float64)  # (M,J,S)

        self.P_price_mj = tf.convert_to_tensor(
            np.asarray(P_price_mj, dtype=np.float64), dtype=tf.float64
        )  # (M,J,S,S)

        self.pi_I0 = tf.convert_to_tensor(np.asarray(pi_I0), dtype=tf.float64)  # (I,)

        self.waste_cost = tf.convert_to_tensor(float(waste_cost), dtype=tf.float64)
        self.eps = tf.convert_to_tensor(float(eps), dtype=tf.float64)
        self.tol = tf.convert_to_tensor(float(tol), dtype=tf.float64)
        self.max_iter = tf.convert_to_tensor(int(max_iter), dtype=tf.int32)

        # ---- prior scales over z (scalar float64 tensors) ----
        # Assumes upstream provides the needed keys:
        #   z_beta, z_alpha, z_v, z_fc, z_lambda, z_u_scale
        self.sigma_z: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(v), dtype=tf.float64)
            for k, v in sigmas.items()
        }

        # ---- inventory maps (precomputed once) ----
        I_max_tf = tf.convert_to_tensor(int(I_max), dtype=tf.int32)
        self.maps = model.build_inventory_maps(I_max_tf)

        # ---- pack constant model inputs for posterior views ----
        self.inputs: dict[str, tf.Tensor] = {
            "a_mnjt": self.a_mnjt,
            "p_state_mjt": self.p_state_mjt,
            "u_mj": self.u_mj,
            "price_vals_mj": self.price_vals_mj,
            "P_price_mj": self.P_price_mj,
            "pi_I0": self.pi_I0,
            "waste_cost": self.waste_cost,
            "eps": self.eps,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "maps": self.maps,
        }

        # ---- RNG (single stream) ----
        self.rng = tf.random.Generator.from_seed(int(seed))

        # ---- initialize z explicitly from init_theta ----

        beta0 = tf.convert_to_tensor(float(init_theta["beta"]), dtype=tf.float64)
        alpha0 = tf.convert_to_tensor(float(init_theta["alpha"]), dtype=tf.float64)
        v0 = tf.convert_to_tensor(float(init_theta["v"]), dtype=tf.float64)
        fc0 = tf.convert_to_tensor(float(init_theta["fc"]), dtype=tf.float64)
        lam0 = tf.convert_to_tensor(float(init_theta["lambda"]), dtype=tf.float64)
        us0 = tf.convert_to_tensor(float(init_theta["u_scale"]), dtype=tf.float64)

        beta0_mj = tf.fill((self.M, self.J), beta0)
        alpha0_mj = tf.fill((self.M, self.J), alpha0)
        v0_mj = tf.fill((self.M, self.J), v0)
        fc0_mj = tf.fill((self.M, self.J), fc0)

        lam0_mn = tf.fill((self.M, self.N), lam0)
        us0_m = tf.fill((self.M,), us0)

        # logit(x) = log(x) - log(1-x) for beta, lambda; log(x) for positive blocks
        z_beta0 = tf.math.log(beta0_mj) - tf.math.log1p(-beta0_mj)
        z_alpha0 = tf.math.log(alpha0_mj)
        z_v0 = tf.math.log(v0_mj)
        z_fc0 = tf.math.log(fc0_mj)

        z_lambda0 = tf.math.log(lam0_mn) - tf.math.log1p(-lam0_mn)
        z_u_scale0 = tf.math.log(us0_m)

        self.z: dict[str, tf.Variable] = {
            "z_beta": tf.Variable(z_beta0, trainable=False, dtype=tf.float64),
            "z_alpha": tf.Variable(z_alpha0, trainable=False, dtype=tf.float64),
            "z_v": tf.Variable(z_v0, trainable=False, dtype=tf.float64),
            "z_fc": tf.Variable(z_fc0, trainable=False, dtype=tf.float64),
            "z_lambda": tf.Variable(z_lambda0, trainable=False, dtype=tf.float64),
            "z_u_scale": tf.Variable(z_u_scale0, trainable=False, dtype=tf.float64),
        }

        # ---- step sizes (set in fit) ----
        self.k: dict[str, tf.Variable] = {
            "beta": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "alpha": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "v": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "fc": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "lambda": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "u_scale": tf.Variable(0.0, dtype=tf.float64, trainable=False),
        }

        # ---- acceptance counters (counts accepted entries per block) ----
        self.accept: dict[str, tf.Variable] = {
            "beta": tf.Variable(0, dtype=tf.int32, trainable=False),
            "alpha": tf.Variable(0, dtype=tf.int32, trainable=False),
            "v": tf.Variable(0, dtype=tf.int32, trainable=False),
            "fc": tf.Variable(0, dtype=tf.int32, trainable=False),
            "lambda": tf.Variable(0, dtype=tf.int32, trainable=False),
            "u_scale": tf.Variable(0, dtype=tf.int32, trainable=False),
        }

        # ---- running sums for posterior means ----
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.sums: dict[str, tf.Variable] = {
            "beta": tf.Variable(
                tf.zeros([self.M, self.J], tf.float64), trainable=False
            ),
            "alpha": tf.Variable(
                tf.zeros([self.M, self.J], tf.float64), trainable=False
            ),
            "v": tf.Variable(tf.zeros([self.M, self.J], tf.float64), trainable=False),
            "fc": tf.Variable(tf.zeros([self.M, self.J], tf.float64), trainable=False),
            "lambda": tf.Variable(
                tf.zeros([self.M, self.N], tf.float64), trainable=False
            ),
            "u_scale": tf.Variable(tf.zeros([self.M], tf.float64), trainable=False),
        }

        # ---- block sizes for acceptance-rate denominators ----
        self._block_sizes: dict[str, int] = {
            "beta": self.M * self.J,
            "alpha": self.M * self.J,
            "v": self.M * self.J,
            "fc": self.M * self.J,
            "lambda": self.M * self.N,
            "u_scale": self.M,
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, n_iter: int, k: dict[str, float]) -> None:
        """
        Run MCMC for n_iter iterations (Python loop), accumulating posterior means.

        Args:
          n_iter: number of sweeps
          k: RW step sizes keyed by:
               {"beta","alpha","v","fc","lambda","u_scale"}
        """
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0")

        self.k["beta"].assign(float(k["beta"]))
        self.k["alpha"].assign(float(k["alpha"]))
        self.k["v"].assign(float(k["v"]))
        self.k["fc"].assign(float(k["fc"]))
        self.k["lambda"].assign(float(k["lambda"]))
        self.k["u_scale"].assign(float(k["u_scale"]))

        self._reset_chain_state()

        for it in range(n_iter):
            it_t = tf.constant(it, dtype=tf.int32)
            self._mcmc_iteration_step(it=it_t)

    def get_results(self) -> dict[str, object]:
        """
        Return posterior means and acceptance rates as Python/numpy objects.

        Returns:
          {
            "theta_hat": {"beta","alpha","v","fc","lambda","u_scale"},
            "n_saved": int,
            "accept": {"beta","alpha","v","fc","lambda","u_scale"}  # rates only
          }
        """
        n_saved = int(self.saved.numpy())
        if n_saved <= 0:
            raise RuntimeError("No saved iterations (n_saved == 0). Call fit().")

        saved_f = tf.cast(self.saved, tf.float64)

        theta_hat = {
            "beta": (self.sums["beta"] / saved_f).numpy(),
            "alpha": (self.sums["alpha"] / saved_f).numpy(),
            "v": (self.sums["v"] / saved_f).numpy(),
            "fc": (self.sums["fc"] / saved_f).numpy(),
            "lambda": (self.sums["lambda"] / saved_f).numpy(),
            "u_scale": (self.sums["u_scale"] / saved_f).numpy(),
        }

        counts = {k: int(v.numpy()) for k, v in self.accept.items()}
        rates = {
            k: counts[k] / max(1, n_saved * self._block_sizes[k]) for k in counts.keys()
        }

        return {"theta_hat": theta_hat, "n_saved": n_saved, "accept": rates}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _current_z_dict(self) -> dict[str, tf.Tensor]:
        """Return the current unconstrained state z as a dict for model/posterior calls."""
        return {
            "z_beta": self.z["z_beta"],
            "z_alpha": self.z["z_alpha"],
            "z_v": self.z["z_v"],
            "z_fc": self.z["z_fc"],
            "z_lambda": self.z["z_lambda"],
            "z_u_scale": self.z["z_u_scale"],
        }

    def _reset_chain_state(self) -> None:
        """Reset acceptance counters and posterior-mean accumulators."""
        for v in self.accept.values():
            v.assign(0)

        self.saved.assign(0)
        for v in self.sums.values():
            v.assign(tf.zeros_like(v))

    # -------------------------------------------------------------------------
    # Only compiled method
    # -------------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it: tf.Tensor) -> None:
        """One compiled MCMC sweep: update each z_* block with elementwise RW-MH."""
        inputs = self.inputs
        sigma_z = self.sigma_z

        z_beta = self.z["z_beta"]
        z_alpha = self.z["z_alpha"]
        z_v = self.z["z_v"]
        z_fc = self.z["z_fc"]
        z_lambda = self.z["z_lambda"]
        z_u_scale = self.z["z_u_scale"]

        # ---- Parameter-block updates (RW-MH, elementwise acceptance) ----

        # ---- Joint RW-MH update for (z_beta, z_alpha, z_fc) with one accept per (m,j) ----

        z_beta_new, z_alpha_new, z_fc_new, accepted = update_z_beta_alpha_fc_mj(
            rng=self.rng,
            k_beta=self.k["beta"],
            k_alpha=self.k["alpha"],
            k_fc=self.k["fc"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_lambda=z_lambda,
            z_u_scale=z_u_scale,
        )

        self.z["z_beta"].assign(z_beta_new)
        self.z["z_alpha"].assign(z_alpha_new)
        self.z["z_fc"].assign(z_fc_new)

        acc = tf.reduce_sum(tf.cast(accepted, tf.int32))
        self.accept["beta"].assign_add(acc)
        self.accept["alpha"].assign_add(acc)
        self.accept["fc"].assign_add(acc)

        # -----------------------------------------------------------------------------
        # Legacy single-block updates for beta/alpha/fc (commented out for easy revert)
        # -----------------------------------------------------------------------------
        # z_new, accepted = update_z_beta_mj(
        #     rng=self.rng,
        #     k=self.k["beta"],
        #     inputs=inputs,
        #     sigma_z=sigma_z,
        #     z_beta=z_beta,
        #     z_alpha=z_alpha,
        #     z_v=z_v,
        #     z_fc=z_fc,
        #     z_lambda=z_lambda,
        #     z_u_scale=z_u_scale,
        # )
        # self.z["z_beta"].assign(z_new)
        # self.accept["beta"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))
        #
        # z_new, accepted = update_z_alpha_mj(
        #     rng=self.rng,
        #     k=self.k["alpha"],
        #     inputs=inputs,
        #     sigma_z=sigma_z,
        #     z_beta=z_beta,
        #     z_alpha=z_alpha,
        #     z_v=z_v,
        #     z_fc=z_fc,
        #     z_lambda=z_lambda,
        #     z_u_scale=z_u_scale,
        # )
        # self.z["z_alpha"].assign(z_new)
        # self.accept["alpha"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))
        #
        # z_new, accepted = update_z_fc_mj(
        #     rng=self.rng,
        #     k=self.k["fc"],
        #     inputs=inputs,
        #     sigma_z=sigma_z,
        #     z_beta=z_beta,
        #     z_alpha=z_alpha,
        #     z_v=z_v,
        #     z_fc=z_fc,
        #     z_lambda=z_lambda,
        #     z_u_scale=z_u_scale,
        # )
        # self.z["z_fc"].assign(z_new)
        # self.accept["fc"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_v_mj(
            rng=self.rng,
            k=self.k["v"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_lambda=z_lambda,
            z_u_scale=z_u_scale,
        )
        self.z["z_v"].assign(z_new)
        self.accept["v"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_lambda_mn(
            rng=self.rng,
            k=self.k["lambda"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_lambda=z_lambda,
            z_u_scale=z_u_scale,
        )
        self.z["z_lambda"].assign(z_new)
        self.accept["lambda"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_u_scale_m(
            rng=self.rng,
            k=self.k["u_scale"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_lambda=z_lambda,
            z_u_scale=z_u_scale,
        )
        self.z["z_u_scale"].assign(z_new)
        self.accept["u_scale"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        # ---- Accumulate posterior means (no burn-in/thinning) ----
        self.saved.assign_add(1)

        z_dict = {
            "z_beta": self.z["z_beta"],
            "z_alpha": self.z["z_alpha"],
            "z_v": self.z["z_v"],
            "z_fc": self.z["z_fc"],
            "z_lambda": self.z["z_lambda"],
            "z_u_scale": self.z["z_u_scale"],
        }

        theta_curr = model.unconstrained_to_theta(z_dict)

        self.sums["beta"].assign_add(theta_curr["beta"])
        self.sums["alpha"].assign_add(theta_curr["alpha"])
        self.sums["v"].assign_add(theta_curr["v"])
        self.sums["fc"].assign_add(theta_curr["fc"])
        self.sums["lambda"].assign_add(theta_curr["lambda"])
        self.sums["u_scale"].assign_add(theta_curr["u_scale"])

        report_iteration_progress(z=z_dict, it=it)
