# ching/stockpiling_estimator.py
#
# Phase-3 (Ching-style) stockpiling estimator.
#
# Public API (called by orchestration) is pure Python:
#   - __init__(...) accepts Python / numpy inputs and converts internally
#   - fit(...) accepts Python scalars and runs a Python loop
#   - get_results() returns Python ints/floats and numpy arrays
#
# Only the per-iteration MCMC sweep is tf.function compiled:
#   - _mcmc_iteration_step(...)
#   - all logp closures / helpers used by MH are defined inside that step
#
# Sampling:
#   - Elementwise Random-Walk Metropolis–Hastings via toolbox.mcmc_kernels.rw_mh_step
#   - Separate updates for each z_* block:
#       z_beta, z_alpha, z_v, z_fc, z_lambda : shape (M, N)
#       z_u_scale                            : shape (M,)
#
# Accumulation:
#   - No burn-in / thinning: all iterations contribute to posterior mean.
#
# Reporting:
#   - All printing is delegated to ching.stockpiling_diagnostics.

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from toolbox.mcmc_kernels import rw_mh_step

from ching.stockpiling_diagnostics import report_iteration_progress
from ching.stockpiling_input_validation import (
    validate_stockpiling_estimator_fit_inputs,
    validate_stockpiling_estimator_init_inputs,
)
from ching.stockpiling_posterior import (
    build_inventory_maps,
    logpost_u_scale_m,
    logpost_z_alpha_mn,
    logpost_z_beta_mn,
    logpost_z_fc_mn,
    logpost_z_lambda_mn,
    logpost_z_v_mn,
)


class StockpilingEstimator:
    """
    Estimate Phase-3 stockpiling parameters with elementwise RW-MH on z-parameters.

    Shapes:
      - Consumer-specific blocks (per market, per consumer): (M, N)
          z_beta, z_alpha, z_v, z_fc, z_lambda
      - Market-specific u_scale: (M,)
          z_u_scale
    """

    # -------------------------------------------------------------------------
    # Public API (pure Python)
    # -------------------------------------------------------------------------

    def __init__(
        self,
        a_imt: Any,  # (M,N,T) actions 0/1
        p_state_mt: Any,  # (M,T) price states (int)
        u_m: Any,  # (M,) fixed utilities
        price_vals: Any,  # (S,) prices by state
        P_price: Any,  # (S,S) Markov transition for price states
        I_max: int,  # scalar
        pi_I0: Any,  # (I,) prior over initial inventory
        waste_cost: float,
        eps: float,
        tol: float,
        max_iter: int,
        sigmas: dict[str, float],
        seed: int,
    ) -> None:
        # Basic dimensions from actions.
        a_np = np.asarray(a_imt)
        self.M = int(a_np.shape[0])
        self.N = int(a_np.shape[1])

        # Minimal init validation (no theta_init scaffolding).
        validate_stockpiling_estimator_init_inputs(
            a_imt=a_np,
            p_state_mt=p_state_mt,
            u_m=u_m,
            price_vals=price_vals,
            P_price=P_price,
            I_max=I_max,
            pi_I0=pi_I0,
            waste_cost=waste_cost,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            sigmas=sigmas,
        )

        # -------------------------
        # Convert inputs to TF
        # -------------------------
        p_np = np.asarray(p_state_mt)

        # Required by posterior (no boundary casting there):
        #   actions/states: int32
        #   all continuous: float64
        self.a_imt = tf.convert_to_tensor(a_np, dtype=tf.int32)  # (M,N,T)
        self.p_state_mt = tf.convert_to_tensor(p_np, dtype=tf.int32)  # (M,T)

        self.u_m = tf.convert_to_tensor(np.asarray(u_m), dtype=tf.float64)  # (M,)
        self.price_vals = tf.convert_to_tensor(
            np.asarray(price_vals), dtype=tf.float64
        )  # (S,)
        self.P_price = tf.convert_to_tensor(
            np.asarray(P_price), dtype=tf.float64
        )  # (S,S)
        self.pi_I0 = tf.convert_to_tensor(np.asarray(pi_I0), dtype=tf.float64)  # (I,)

        self.waste_cost = tf.convert_to_tensor(float(waste_cost), dtype=tf.float64)
        self.eps = tf.convert_to_tensor(float(eps), dtype=tf.float64)
        self.tol = tf.convert_to_tensor(float(tol), dtype=tf.float64)
        self.max_iter = tf.convert_to_tensor(int(max_iter), dtype=tf.int32)

        # Prior scales (scalar float64 tensors).
        self.sigmas: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(v), dtype=tf.float64)
            for k, v in sigmas.items()
        }

        # Precompute inventory maps once; posterior views take maps (not I_max).
        I_max_tf = tf.convert_to_tensor(int(I_max), dtype=tf.int32)
        self.maps = build_inventory_maps(I_max_tf)

        # RNG (single stream).
        self.rng = tf.random.Generator.from_seed(int(seed))

        # -------------------------
        # Initialize z at prior mode: all zeros.
        # -------------------------
        z_mn = tf.zeros((self.M, self.N), dtype=tf.float64)
        z_m = tf.zeros((self.M,), dtype=tf.float64)

        self.z: dict[str, tf.Variable] = {
            "z_beta": tf.Variable(z_mn, trainable=False, dtype=tf.float64),
            "z_alpha": tf.Variable(z_mn, trainable=False, dtype=tf.float64),
            "z_v": tf.Variable(z_mn, trainable=False, dtype=tf.float64),
            "z_fc": tf.Variable(z_mn, trainable=False, dtype=tf.float64),
            "z_lambda": tf.Variable(z_mn, trainable=False, dtype=tf.float64),
            "z_u_scale": tf.Variable(z_m, trainable=False, dtype=tf.float64),
        }

        # Elementwise acceptance counters (counts accepted entries per block).
        self.accept: dict[str, tf.Variable] = {
            "beta": tf.Variable(0, dtype=tf.int32, trainable=False),
            "alpha": tf.Variable(0, dtype=tf.int32, trainable=False),
            "v": tf.Variable(0, dtype=tf.int32, trainable=False),
            "fc": tf.Variable(0, dtype=tf.int32, trainable=False),
            "lambda_c": tf.Variable(0, dtype=tf.int32, trainable=False),
            "u_scale": tf.Variable(0, dtype=tf.int32, trainable=False),
        }

        # Running sums for posterior means (accumulated inside the compiled step).
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.sums: dict[str, tf.Variable] = {
            "beta": tf.Variable(
                tf.zeros([self.M, self.N], tf.float64), trainable=False
            ),
            "alpha": tf.Variable(
                tf.zeros([self.M, self.N], tf.float64), trainable=False
            ),
            "v": tf.Variable(tf.zeros([self.M, self.N], tf.float64), trainable=False),
            "fc": tf.Variable(tf.zeros([self.M, self.N], tf.float64), trainable=False),
            "lambda_c": tf.Variable(
                tf.zeros([self.M, self.N], tf.float64), trainable=False
            ),
            "u_scale": tf.Variable(tf.zeros([self.M], tf.float64), trainable=False),
        }

    def fit(
        self,
        n_iter: int,
        k_beta: float,
        k_alpha: float,
        k_v: float,
        k_fc: float,
        k_lambda: float,
        k_u_scale: float,
    ) -> None:
        """Run MCMC for n_iter iterations (Python loop), accumulating posterior means."""
        validate_stockpiling_estimator_fit_inputs(
            n_iter=n_iter,
            k_beta=k_beta,
            k_alpha=k_alpha,
            k_v=k_v,
            k_fc=k_fc,
            k_lambda=k_lambda,
            k_u_scale=k_u_scale,
        )
        n_iter = int(n_iter)

        k_beta_tf = tf.convert_to_tensor(float(k_beta), dtype=tf.float64)
        k_alpha_tf = tf.convert_to_tensor(float(k_alpha), dtype=tf.float64)
        k_v_tf = tf.convert_to_tensor(float(k_v), dtype=tf.float64)
        k_fc_tf = tf.convert_to_tensor(float(k_fc), dtype=tf.float64)
        k_lambda_tf = tf.convert_to_tensor(float(k_lambda), dtype=tf.float64)
        k_u_scale_tf = tf.convert_to_tensor(float(k_u_scale), dtype=tf.float64)

        # Reset acceptance counters.
        for v in self.accept.values():
            v.assign(0)

        # Reset running sums.
        self.saved.assign(0)
        for v in self.sums.values():
            v.assign(tf.zeros_like(v))

        self._run_mcmc_loop(
            n_iter=n_iter,
            k_beta=k_beta_tf,
            k_alpha=k_alpha_tf,
            k_v=k_v_tf,
            k_fc=k_fc_tf,
            k_lambda=k_lambda_tf,
            k_u_scale=k_u_scale_tf,
        )

    def _run_mcmc_loop(
        self,
        n_iter: int,
        k_beta: tf.Tensor,
        k_alpha: tf.Tensor,
        k_v: tf.Tensor,
        k_fc: tf.Tensor,
        k_lambda: tf.Tensor,
        k_u_scale: tf.Tensor,
    ) -> None:
        """Run the Python-owned iteration loop and mutate sampler state."""
        for it in range(n_iter):
            # Keep 'it' as a TF scalar to avoid retracing on Python ints.
            it_t = tf.constant(it, dtype=tf.int32)
            self._mcmc_iteration_step(
                it=it_t,
                k_beta=k_beta,
                k_alpha=k_alpha,
                k_v=k_v,
                k_fc=k_fc,
                k_lambda=k_lambda,
                k_u_scale=k_u_scale,
            )

    def get_results(self) -> dict[str, object]:
        """Return posterior means and acceptance summaries as Python/numpy objects."""
        n_saved = int(self.saved.numpy())
        if n_saved <= 0:
            raise RuntimeError(
                "No saved iterations (n_saved == 0). Call fit() with n_iter > 0."
            )

        saved_f = tf.cast(self.saved, tf.float64)

        theta_hat = {
            "beta": (self.sums["beta"] / saved_f).numpy(),
            "alpha": (self.sums["alpha"] / saved_f).numpy(),
            "v": (self.sums["v"] / saved_f).numpy(),
            "fc": (self.sums["fc"] / saved_f).numpy(),
            "lambda_c": (self.sums["lambda_c"] / saved_f).numpy(),
            "u_scale": (self.sums["u_scale"] / saved_f).numpy(),
        }

        counts = {k: int(v.numpy()) for k, v in self.accept.items()}

        denom_mn = max(1, n_saved * self.M * self.N)
        denom_m = max(1, n_saved * self.M)

        rates = {
            "beta": counts["beta"] / denom_mn,
            "alpha": counts["alpha"] / denom_mn,
            "v": counts["v"] / denom_mn,
            "fc": counts["fc"] / denom_mn,
            "lambda_c": counts["lambda_c"] / denom_mn,
            "u_scale": counts["u_scale"] / denom_m,
        }

        return {
            "theta_hat": theta_hat,
            "n_saved": n_saved,
            "accept": {"counts": counts, "rates": rates},
        }

    # -------------------------------------------------------------------------
    # Only compiled method
    # -------------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(
        self,
        it: tf.Tensor,
        k_beta: tf.Tensor,
        k_alpha: tf.Tensor,
        k_v: tf.Tensor,
        k_fc: tf.Tensor,
        k_lambda: tf.Tensor,
        k_u_scale: tf.Tensor,
    ) -> None:
        """One compiled MCMC sweep: update each z_* block with elementwise RW-MH."""

        # Bind shared tensors once (reduces noise in logp closures).
        a_imt = self.a_imt
        p_state_mt = self.p_state_mt
        u_m = self.u_m
        price_vals = self.price_vals
        P_price = self.P_price
        pi_I0 = self.pi_I0
        waste_cost = self.waste_cost
        eps = self.eps
        tol = self.tol
        max_iter = self.max_iter
        sigmas = self.sigmas
        maps = self.maps

        # Read current blocks (variables).
        z_beta = self.z["z_beta"]
        z_alpha = self.z["z_alpha"]
        z_v = self.z["z_v"]
        z_fc = self.z["z_fc"]
        z_lambda = self.z["z_lambda"]
        z_u_scale = self.z["z_u_scale"]

        def z_from_parts(
            z_beta_t: tf.Tensor,
            z_alpha_t: tf.Tensor,
            z_v_t: tf.Tensor,
            z_fc_t: tf.Tensor,
            z_lambda_t: tf.Tensor,
            z_u_scale_t: tf.Tensor,
        ) -> dict[str, tf.Tensor]:
            return {
                "z_beta": z_beta_t,
                "z_alpha": z_alpha_t,
                "z_v": z_v_t,
                "z_fc": z_fc_t,
                "z_lambda": z_lambda_t,
                "z_u_scale": z_u_scale_t,
            }

        def call_view(view_fn, z_dict: dict[str, tf.Tensor]) -> tf.Tensor:
            return view_fn(
                z=z_dict,
                a_imt=a_imt,
                p_state_mt=p_state_mt,
                u_m=u_m,
                price_vals=price_vals,
                P_price=P_price,
                pi_I0=pi_I0,
                waste_cost=waste_cost,
                eps=eps,
                tol=tol,
                max_iter=max_iter,
                maps=maps,
                sigmas=sigmas,
            )

        # ---- logp closures (return shape matching the updated block) ----

        def logp_beta(z_beta_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta_t, z_alpha, z_v, z_fc, z_lambda, z_u_scale)
            return call_view(logpost_z_beta_mn, z)

        def logp_alpha(z_alpha_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta, z_alpha_t, z_v, z_fc, z_lambda, z_u_scale)
            return call_view(logpost_z_alpha_mn, z)

        def logp_v(z_v_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta, z_alpha, z_v_t, z_fc, z_lambda, z_u_scale)
            return call_view(logpost_z_v_mn, z)

        def logp_fc(z_fc_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta, z_alpha, z_v, z_fc_t, z_lambda, z_u_scale)
            return call_view(logpost_z_fc_mn, z)

        def logp_lambda(z_lambda_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta, z_alpha, z_v, z_fc, z_lambda_t, z_u_scale)
            return call_view(logpost_z_lambda_mn, z)

        def logp_u_scale(z_u_scale_t: tf.Tensor) -> tf.Tensor:
            z = z_from_parts(z_beta, z_alpha, z_v, z_fc, z_lambda, z_u_scale_t)
            return call_view(logpost_u_scale_m, z)

        # ---- RW-MH updates (elementwise acceptance) ----

        z_new, accepted = rw_mh_step(z_beta, logp_beta, k_beta, self.rng)
        self.z["z_beta"].assign(z_new)
        self.accept["beta"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = rw_mh_step(z_alpha, logp_alpha, k_alpha, self.rng)
        self.z["z_alpha"].assign(z_new)
        self.accept["alpha"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = rw_mh_step(z_v, logp_v, k_v, self.rng)
        self.z["z_v"].assign(z_new)
        self.accept["v"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = rw_mh_step(z_fc, logp_fc, k_fc, self.rng)
        self.z["z_fc"].assign(z_new)
        self.accept["fc"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = rw_mh_step(z_lambda, logp_lambda, k_lambda, self.rng)
        self.z["z_lambda"].assign(z_new)
        self.accept["lambda_c"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = rw_mh_step(z_u_scale, logp_u_scale, k_u_scale, self.rng)
        self.z["z_u_scale"].assign(z_new)
        self.accept["u_scale"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        # ---- Accumulate posterior means (no burn-in/thinning) ----

        self.saved.assign_add(1)

        z_curr = self.z
        beta = tf.math.sigmoid(z_curr["z_beta"])
        alpha = tf.exp(z_curr["z_alpha"])
        v = tf.exp(z_curr["z_v"])
        fc = tf.exp(z_curr["z_fc"])
        lambda_c = tf.math.sigmoid(z_curr["z_lambda"])
        u_scale = tf.exp(z_curr["z_u_scale"])

        self.sums["beta"].assign_add(beta)
        self.sums["alpha"].assign_add(alpha)
        self.sums["v"].assign_add(v)
        self.sums["fc"].assign_add(fc)
        self.sums["lambda_c"].assign_add(lambda_c)
        self.sums["u_scale"].assign_add(u_scale)

        report_iteration_progress(self, it)
