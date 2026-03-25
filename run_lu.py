"""
End-to-end simulation harness for comparing BLP vs Lu shrinkage on synthetic markets.

This script:
  1) Generates synthetic market data under multiple DGP variants.
  2) Fits two BLP estimators (strong-IV and weak-IV).
  3) Runs the Lu shrinkage sampler using tfp.mcmc.
  4) Compares recovered shocks and sigma against ground truth.

All estimator configuration is set in this orchestration layer and passed down as
config objects. The shrinkage sampler is treated as a pure run_chain(...) +
summarize_samples(...) pipeline.
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from datasets.lu_dgp import (
    BasicLuChoiceModel,
    generate_market,
    generate_market_conditions,
)
from lu.blp.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from lu.blp.blp_input_validation import validate_blp_config
from lu.shrinkage.lu_posterior import LuPosteriorConfig
from lu.shrinkage.lu_shrinkage import (
    LuShrinkageConfig,
    run_chain,
    summarize_samples,
)
from toolbox.assess_estimator import print_assessment


def _to_numpy_results(results: dict) -> dict:
    converted = {}
    for key, value in results.items():
        if isinstance(value, tf.Tensor):
            converted[key] = value.numpy()
        else:
            converted[key] = value
    return converted


def main() -> None:
    """Run synthetic Lu-style markets and compare estimators across DGP variants."""
    # -------------------------------------------------------------------------
    # Experiment dimensions and true structural parameters used in the simulator
    # -------------------------------------------------------------------------
    T, J, N = 25, 15, 1000
    beta_p_true, beta_w_true, sigma_true = -1.0, 0.5, 1.5
    seed = 123

    # Number of simulation draws used inside estimators for RC integration.
    # (Aligned with Lu(25) Section 4 setting R0=200.)
    n_draws = 200

    # -------------------------------------------------------------------------
    # BLP configuration
    # -------------------------------------------------------------------------
    blp_config_raw = {
        "n_draws": n_draws,
        "seed": seed,
        "damping": 0.5,
        "tol": 1e-12,
        "share_tol": 1e-12,
        "max_iter": 2000,
        "fail_penalty": 1e8,
        "sigma_lower": 1e-3,
        "sigma_upper": 5.0,
        "sigma_grid_points": 25,
        "nelder_mead_maxiter": 500,
        "nelder_mead_xatol": 1e-6,
        "nelder_mead_fatol": 1e-6,
    }
    blp_config = validate_blp_config(blp_config_raw)

    # -------------------------------------------------------------------------
    # Lu shrinkage posterior configuration
    # -------------------------------------------------------------------------
    posterior_config = LuPosteriorConfig(
        n_draws=n_draws,
        seed=seed,
        dtype=tf.float64,
        eps=1e-15,
        beta_p_mean=0.0,
        beta_p_var=10.0,
        beta_w_mean=0.0,
        beta_w_var=10.0,
        r_mean=0.0,
        r_var=0.5,
        E_bar_mean=0.0,
        E_bar_var=10.0,
        T0_sq=1e-3,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
    )

    # -------------------------------------------------------------------------
    # Lu shrinkage chain configuration
    # -------------------------------------------------------------------------
    shrinkage_config = LuShrinkageConfig(
        num_results=500,
        num_burnin_steps=0,
        rw_scale=0.1,
    )

    # -------------------------------------------------------------------------
    # Loop over DGP variants. Each variant changes sparsity and endogeneity.
    # -------------------------------------------------------------------------
    for dgp_type in (1, 2, 3, 4):
        print(f"=== DGP {dgp_type} ===")

        # ---------------------------------------------------------------------
        # Step 1: Generate market-level primitives and construct prices.
        # ---------------------------------------------------------------------
        wjt, Ejt, ujt, alpha, E_bar_t, njt, support_true = generate_market_conditions(
            T=T,
            J=J,
            dgp_type=dgp_type,
            seed=seed,
        )
        pjt = alpha + 0.3 * wjt + ujt

        int_true = float(np.mean(E_bar_t))
        E_true = njt

        # ---------------------------------------------------------------------
        # Step 2: Simulate utilities and aggregate to market outcomes.
        # ---------------------------------------------------------------------
        model = BasicLuChoiceModel(
            N=N,
            beta_p=beta_p_true,
            beta_w=beta_w_true,
            sigma=sigma_true,
            seed=seed,
        )
        print("=== Linear model built ===")

        uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)
        sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=seed)
        print("=== Market generated ===")

        # ---------------------------------------------------------------------
        # Step 3: Fit BLP under strong and weak instrument sets.
        # ---------------------------------------------------------------------
        Zjt_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
        blp_strong = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_strong,
            config=blp_config,
        )
        """
        print("=== Strong IVs and Estimator built ===")
        blp_strong.fit()
        res_strong = blp_strong.get_results()
        print("=== Strong BLP Estimator fitted ===")

        Zjt_weak = build_weak_IVs(wjt=wjt)
        blp_weak = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_weak,
            config=blp_config,
        )
        print("=== Weak IVs and Estimator built ===")
        blp_weak.fit()
        res_weak = blp_weak.get_results()
        print("=== Weak BLP Estimator fitted ===")
        """
        # ---------------------------------------------------------------------
        # Step 4: Run the Lu shrinkage sampler.
        #
        # The sampler uses:
        #   - a built-in TFP kernel for the continuous block
        #   - Gibbs updates for gamma and phi
        #   - tfp.mcmc.sample_chain as the top-level driver
        # ---------------------------------------------------------------------
        shrinkage_samples = run_chain(
            pjt=tf.convert_to_tensor(pjt, dtype=tf.float64),
            wjt=tf.convert_to_tensor(wjt, dtype=tf.float64),
            qjt=tf.convert_to_tensor(qjt, dtype=tf.float64),
            q0t=tf.convert_to_tensor(q0t, dtype=tf.float64),
            posterior_config=posterior_config,
            shrinkage_config=shrinkage_config,
            seed=seed,
        )
        res_shrink = _to_numpy_results(summarize_samples(shrinkage_samples))
        print("=== Shrinkage sampler run ===")
        """
        # ---------------------------------------------------------------------
        # Step 5: Compare estimates to ground truth.
        # ---------------------------------------------------------------------
        print("=== Strong BLP Estimator Results ===")
        print_assessment(
            results=res_strong,
            int_true=int_true,
            E_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )

        print("=== Weak BLP Estimator Results ===")
        print_assessment(
            results=res_weak,
            int_true=int_true,
            E_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )
        """
        print("=== Shrinkage Estimator Results ===")
        print_assessment(
            results=res_shrink,
            int_true=int_true,
            E_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )


if __name__ == "__main__":
    main()
