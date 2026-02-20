"""
ching/stockpiling_estimator.py

MCMC estimator for the Phase-3 stockpiling model (multi-product).

This module provides a small, examiner-readable wrapper around:
  - the structural model/CCP solver (ching.stockpiling_model)
  - the likelihood/prediction layer with inventory filtering (ching.stockpiling_posterior)
  - RW-MH update kernels for unconstrained parameters (ching.stockpiling_updates)
  - iteration diagnostics (ching.stockpiling_diagnostics)

Design:
  - No input validation is performed here. All validation/normalization is delegated
    to ching.stockpiling_input_validation.
  - No default argument values. All settings must be supplied by the caller/config.
  - u_scale is treated as a standard model component (per-market utility scale).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from ching.stockpiling_diagnostics import report_iteration_progress
from ching.stockpiling_input_validation import (
    normalize_stockpiling_estimator_fit_inputs,
    normalize_stockpiling_estimator_init_inputs,
    normalize_stockpiling_estimator_init_theta,
)
from ching.stockpiling_model import build_inventory_maps, unconstrained_to_theta
from ching.stockpiling_posterior import (
    StockpilingInputs,
    predict_p_buy_mnjt_from_theta,
)
from ching.stockpiling_updates import (
    update_z_alpha_j,
    update_z_beta_scalar,
    update_z_fc_j,
    update_z_u_scale_m,
    update_z_v_j,
)

ONE_F64 = tf.constant(1.0, dtype=tf.float64)


def _logit(p: tf.Tensor) -> tf.Tensor:
    """Compute log(p/(1-p)) elementwise (expects p in (0,1))."""
    return tf.math.log(p) - tf.math.log(ONE_F64 - p)


def _pack_z(
    z_beta: tf.Tensor,
    z_alpha: tf.Tensor,
    z_v: tf.Tensor,
    z_fc: tf.Tensor,
    z_u_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Pack unconstrained state into the z-dict expected by diagnostics/transforms."""
    return {
        "z_beta": z_beta,
        "z_alpha": z_alpha,
        "z_v": z_v,
        "z_fc": z_fc,
        "z_u_scale": z_u_scale,
    }


def _z_from_init_theta(
    init_theta: Mapping[str, Any], M: int, J: int
) -> dict[str, tf.Tensor]:
    """Convert constrained init_theta to unconstrained z, after normalization."""
    init = normalize_stockpiling_estimator_init_theta(
        init_theta=init_theta, M=int(M), J=int(J)
    )

    beta = tf.constant(init["beta"], dtype=tf.float64)  # scalar
    alpha = tf.constant(
        np.asarray(init["alpha"], dtype=np.float64), dtype=tf.float64
    )  # (J,)
    v = tf.constant(np.asarray(init["v"], dtype=np.float64), dtype=tf.float64)  # (J,)
    fc = tf.constant(np.asarray(init["fc"], dtype=np.float64), dtype=tf.float64)  # (J,)
    u_scale = tf.constant(
        np.asarray(init["u_scale"], dtype=np.float64), dtype=tf.float64
    )  # (M,)

    z_beta = tf.reshape(_logit(beta), (1,))  # store scalar beta as shape (1,)
    z_alpha = tf.math.log(alpha)
    z_v = tf.math.log(v)
    z_fc = tf.math.log(fc)
    z_u_scale = tf.math.log(u_scale)

    return _pack_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)


class StockpilingEstimator:
    """MCMC sampler for the Phase-3 stockpiling model.

    Fixed data (known to the estimator):
      a_mnjt        (M,N,J,T)  purchases in {0,1}
      s_mjt         (M,J,T)    price states in {0,...,S-1}
      u_mj          (M,J)      fixed intercepts from Phase 1–2
      price_vals_mj (M,J,S)    price levels by state
      P_price_mj    (M,J,S,S)  price-state transitions
      lambda_mn     (M,N)      consumption probability in (0,1)
      pi_I0         (I_max+1,) initial inventory distribution
      waste_cost    scalar

    Sampler settings:
      tol, max_iter   DP/CCP solver settings passed through the posterior
      sigmas          prior scales on unconstrained z-blocks
      rng_seed        seed for tf.random.Generator
    """

    def __init__(
        self,
        a_mnjt: Any,
        s_mjt: Any,
        u_mj: Any,
        price_vals_mj: Any,
        P_price_mj: Any,
        lambda_mn: Any,
        I_max: int,
        pi_I0: Any,
        waste_cost: float,
        tol: float,
        max_iter: int,
        sigmas: dict[str, Any],
        rng_seed: int,
    ) -> None:
        norm = normalize_stockpiling_estimator_init_inputs(
            a_mnjt=a_mnjt,
            s_mjt=s_mjt,
            u_mj=u_mj,
            price_vals_mj=price_vals_mj,
            P_price_mj=P_price_mj,
            lambda_mn=lambda_mn,
            I_max=I_max,
            pi_I0=pi_I0,
            waste_cost=waste_cost,
            tol=tol,
            max_iter=max_iter,
            sigmas=sigmas,
            rng_seed=rng_seed,
        )

        self.M = int(norm["M"])
        self.N = int(norm["N"])
        self.J = int(norm["J"])
        self.T = int(norm["T"])
        self.S = int(norm["S"])
        self.I_max = int(norm["I_max"])

        inventory_maps = build_inventory_maps(self.I_max)

        self.inputs: StockpilingInputs = {
            "a_mnjt": tf.convert_to_tensor(norm["a_mnjt"], dtype=tf.int32),
            "s_mjt": tf.convert_to_tensor(norm["s_mjt"], dtype=tf.int32),
            "u_mj": tf.convert_to_tensor(norm["u_mj"], dtype=tf.float64),
            "P_price_mj": tf.convert_to_tensor(norm["P_price_mj"], dtype=tf.float64),
            "price_vals_mj": tf.convert_to_tensor(
                norm["price_vals_mj"], dtype=tf.float64
            ),
            "lambda_mn": tf.convert_to_tensor(norm["lambda_mn"], dtype=tf.float64),
            "waste_cost": tf.convert_to_tensor(norm["waste_cost"], dtype=tf.float64),
            "inventory_maps": inventory_maps,
            "tol": float(norm["tol"]),
            "max_iter": int(norm["max_iter"]),
            "pi_I0": tf.convert_to_tensor(norm["pi_I0"], dtype=tf.float64),
        }

        sig = norm["sigmas"]
        self.sigma_z_beta = tf.convert_to_tensor(sig["z_beta"], dtype=tf.float64)
        self.sigma_z_alpha = tf.convert_to_tensor(sig["z_alpha"], dtype=tf.float64)
        self.sigma_z_v = tf.convert_to_tensor(sig["z_v"], dtype=tf.float64)
        self.sigma_z_fc = tf.convert_to_tensor(sig["z_fc"], dtype=tf.float64)
        self.sigma_z_u_scale = tf.convert_to_tensor(sig["z_u_scale"], dtype=tf.float64)

        self.rng = tf.random.Generator.from_seed(int(norm["rng_seed"]))

        self.z: dict[str, tf.Tensor] | None = None
        self.theta_mean: dict[str, tf.Tensor] | None = None
        self.accept: dict[str, float] | None = None
        self.n_saved: int | None = None

    def fit(
        self,
        n_iter: int,
        k: dict[str, Any],
        init_theta: Mapping[str, Any],
        print_every: int,
    ) -> dict[str, Any]:
        """Run MCMC and return posterior means and acceptance rates.

        Args:
          n_iter: number of MCMC iterations
          k: proposal scales dict with keys {"beta","alpha","v","fc","u_scale"}
          init_theta: constrained initial state dict with keys {"beta","alpha","v","fc","u_scale"}
          print_every: print diagnostics every this many iterations

        Returns:
          dict with keys:
            - theta_mean: constrained posterior mean tensors
            - acceptance: acceptance rates by block
            - z_last: final unconstrained state (dict of tensors)
        """
        fit_norm = normalize_stockpiling_estimator_fit_inputs(n_iter=n_iter, k=k)
        n_iter_i = int(fit_norm["n_iter"])
        k_use = fit_norm["k"]

        z = _z_from_init_theta(init_theta=init_theta, M=self.M, J=self.J)

        k_beta = tf.convert_to_tensor(k_use["beta"], dtype=tf.float64)
        k_alpha = tf.convert_to_tensor(k_use["alpha"], dtype=tf.float64)
        k_v = tf.convert_to_tensor(k_use["v"], dtype=tf.float64)
        k_fc = tf.convert_to_tensor(k_use["fc"], dtype=tf.float64)
        k_u_scale = tf.convert_to_tensor(k_use["u_scale"], dtype=tf.float64)

        theta_sum = {
            "beta": tf.zeros((), dtype=tf.float64),
            "alpha": tf.zeros((self.J,), dtype=tf.float64),
            "v": tf.zeros((self.J,), dtype=tf.float64),
            "fc": tf.zeros((self.J,), dtype=tf.float64),
            "u_scale": tf.zeros((self.M,), dtype=tf.float64),
        }

        acc_sum = {
            "beta": tf.zeros((), dtype=tf.float64),
            "alpha": tf.zeros((), dtype=tf.float64),
            "v": tf.zeros((), dtype=tf.float64),
            "fc": tf.zeros((), dtype=tf.float64),
            "u_scale": tf.zeros((), dtype=tf.float64),
        }

        for it in range(n_iter_i):
            z_beta_new, acc_beta = update_z_beta_scalar(
                z_beta=z["z_beta"],
                z_alpha=z["z_alpha"],
                z_v=z["z_v"],
                z_fc=z["z_fc"],
                z_u_scale=z["z_u_scale"],
                inputs=self.inputs,
                sigma_z_beta=self.sigma_z_beta,
                k_beta=k_beta,
                rng=self.rng,
            )
            z["z_beta"] = z_beta_new

            z_alpha_new, acc_alpha = update_z_alpha_j(
                z_beta=z["z_beta"],
                z_alpha=z["z_alpha"],
                z_v=z["z_v"],
                z_fc=z["z_fc"],
                z_u_scale=z["z_u_scale"],
                inputs=self.inputs,
                sigma_z_alpha=self.sigma_z_alpha,
                k_alpha=k_alpha,
                rng=self.rng,
            )
            z["z_alpha"] = z_alpha_new

            z_v_new, acc_v = update_z_v_j(
                z_beta=z["z_beta"],
                z_alpha=z["z_alpha"],
                z_v=z["z_v"],
                z_fc=z["z_fc"],
                z_u_scale=z["z_u_scale"],
                inputs=self.inputs,
                sigma_z_v=self.sigma_z_v,
                k_v=k_v,
                rng=self.rng,
            )
            z["z_v"] = z_v_new

            z_fc_new, acc_fc = update_z_fc_j(
                z_beta=z["z_beta"],
                z_alpha=z["z_alpha"],
                z_v=z["z_v"],
                z_fc=z["z_fc"],
                z_u_scale=z["z_u_scale"],
                inputs=self.inputs,
                sigma_z_fc=self.sigma_z_fc,
                k_fc=k_fc,
                rng=self.rng,
            )
            z["z_fc"] = z_fc_new

            z_u_scale_new, acc_u_scale = update_z_u_scale_m(
                z_beta=z["z_beta"],
                z_alpha=z["z_alpha"],
                z_v=z["z_v"],
                z_fc=z["z_fc"],
                z_u_scale=z["z_u_scale"],
                inputs=self.inputs,
                sigma_z_u_scale=self.sigma_z_u_scale,
                k_u_scale=k_u_scale,
                rng=self.rng,
            )
            z["z_u_scale"] = z_u_scale_new

            theta = unconstrained_to_theta(z)
            theta_sum["beta"] = theta_sum["beta"] + tf.reshape(theta["beta"], ())
            theta_sum["alpha"] = theta_sum["alpha"] + theta["alpha"]
            theta_sum["v"] = theta_sum["v"] + theta["v"]
            theta_sum["fc"] = theta_sum["fc"] + theta["fc"]
            theta_sum["u_scale"] = theta_sum["u_scale"] + theta["u_scale"]

            acc_sum["beta"] = acc_sum["beta"] + tf.reduce_mean(
                tf.cast(acc_beta, tf.float64)
            )
            acc_sum["alpha"] = acc_sum["alpha"] + tf.reduce_mean(
                tf.cast(acc_alpha, tf.float64)
            )
            acc_sum["v"] = acc_sum["v"] + tf.reduce_mean(tf.cast(acc_v, tf.float64))
            acc_sum["fc"] = acc_sum["fc"] + tf.reduce_mean(tf.cast(acc_fc, tf.float64))
            acc_sum["u_scale"] = acc_sum["u_scale"] + tf.reduce_mean(
                tf.cast(acc_u_scale, tf.float64)
            )

            if print_every > 0 and (it % int(print_every) == 0):
                report_iteration_progress(z=z, it=tf.constant(it, dtype=tf.int32))

        denom = tf.cast(n_iter_i, tf.float64)
        theta_mean = {
            "beta": theta_sum["beta"] / denom,
            "alpha": theta_sum["alpha"] / denom,
            "v": theta_sum["v"] / denom,
            "fc": theta_sum["fc"] / denom,
            "u_scale": theta_sum["u_scale"] / denom,
        }

        accept = {
            "beta": float((acc_sum["beta"] / denom).numpy()),
            "alpha": float((acc_sum["alpha"] / denom).numpy()),
            "v": float((acc_sum["v"] / denom).numpy()),
            "fc": float((acc_sum["fc"] / denom).numpy()),
            "u_scale": float((acc_sum["u_scale"] / denom).numpy()),
        }

        n_saved = int(n_iter_i)

        self.z = z
        self.theta_mean = theta_mean
        self.accept = accept
        self.n_saved = n_saved

        return {
            "theta_mean": theta_mean,
            "accept": accept,
            "n_saved": n_saved,
            "z_last": z,
        }

    def predict_probabilities(self, theta: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """Predict buy probabilities p_buy_mnjt under the supplied constrained theta.

        Returns:
          p_buy_mnjt: (M,N,J,T) float64
        """
        theta_use = {
            "beta": tf.convert_to_tensor(theta["beta"], dtype=tf.float64),
            "alpha": tf.convert_to_tensor(theta["alpha"], dtype=tf.float64),
            "v": tf.convert_to_tensor(theta["v"], dtype=tf.float64),
            "fc": tf.convert_to_tensor(theta["fc"], dtype=tf.float64),
            "u_scale": tf.convert_to_tensor(theta["u_scale"], dtype=tf.float64),
        }
        return predict_p_buy_mnjt_from_theta(theta=theta_use, inputs=self.inputs)
