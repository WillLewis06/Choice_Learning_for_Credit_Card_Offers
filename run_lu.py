"""
End-to-end simulation harness for comparing BLP vs Lu shrinkage on synthetic markets.

This script:
  1) Generates synthetic market data under multiple DGP variants.
  2) Fits two BLP estimators (strong-IV and weak-IV).
  3) Fits the Lu shrinkage estimator (Bayesian sparse market-product shocks).
  4) Compares recovered shocks and sigma against ground truth.

All estimator configuration is set in this orchestration layer and passed down as
validated config objects (no defaults, no fallbacks).
"""

from __future__ import annotations

import os

# Reduce TensorFlow logging noise in terminal output.
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
from lu.shrinkage.lu_shrinkage import LuShrinkageEstimator, LuShrinkageFitConfig
from lu.shrinkage.lu_validate_input import posterior_validate_input
from toolbox.assess_estimator import print_assessment


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
    # BLP configuration (validated in orchestration; no defaults)
    # -------------------------------------------------------------------------
    blp_config_raw = {
        "n_draws": n_draws,
        "seed": seed,
        # Contraction mapping / inversion controls
        "damping": 0.5,
        "tol": 1e-12,
        "share_tol": 1e-12,
        "max_iter": 2000,
        # Objective penalty used when inversion fails
        "fail_penalty": 1e8,
        # Sigma search region and grid
        "sigma_lower": 1e-3,
        "sigma_upper": 5.0,
        "sigma_grid_points": 25,
        # Nelder-Mead controls
        "nelder_mead_maxiter": 500,
        "nelder_mead_xatol": 1e-6,
        "nelder_mead_fatol": 1e-6,
    }
    blp_config = validate_blp_config(blp_config_raw)

    # -------------------------------------------------------------------------
    # Lu shrinkage posterior/prior configuration (validated; no defaults)
    # -------------------------------------------------------------------------
    posterior_config = LuPosteriorConfig(
        # Monte Carlo integration settings
        n_draws=n_draws,
        seed=seed,
        # Numeric settings
        dtype=tf.float64,
        eps=1e-15,
        # Global priors: Normal(mean, var)
        beta_p_mean=0.0,
        beta_p_var=10.0,
        beta_w_mean=0.0,
        beta_w_var=10.0,
        r_mean=0.0,
        r_var=0.5,
        # Market common shock prior: Normal(mean, var)
        E_bar_mean=0.0,
        E_bar_var=10.0,
        # Spike-and-slab variances
        T0_sq=1e-3,
        T1_sq=1.0,
        # Beta prior for phi
        a_phi=1.0,
        b_phi=1.0,
    )
    posterior_validate_input(posterior_config)

    # -------------------------------------------------------------------------
    # Lu shrinkage tuning + sampling configuration (no burn-in/thinning)
    # -------------------------------------------------------------------------
    fit_config = LuShrinkageFitConfig(
        n_iter=500,
        pilot_length=20,
        ridge=1e-6,
        target_low=0.3,
        target_high=0.5,
        max_rounds=100,
        factor_rw=1.3,
        factor_tmh=1.05,
    )

    # -------------------------------------------------------------------------
    # Loop over DGP variants. Each variant changes sparsity and endogeneity.
    # -------------------------------------------------------------------------
    for dgp_type in (1, 2, 3, 4):
        print(f"=== DGP {dgp_type} ===")

        # ---------------------------------------------------------------------
        # Step 1: Generate market-level primitives and construct prices.
        #
        # generate_market_conditions returns:
        #   wjt         : exogenous characteristic
        #   Ejt         : total demand shock (E_bar_t[:, None] + njt)
        #   ujt         : cost shock entering pricing
        #   alpha       : endogeneity shifter (0 for exogenous-price DGPs)
        #   E_bar_t     : common market component (Lu table "Int")
        #   njt         : market-product deviations (Lu table "xi")
        #   support_true: nonzero mask for njt (DGP1/2 only; used for "Prob.")
        #
        # Pricing rule in this codebase:
        #   pjt = alpha + 0.3 * wjt + ujt
        # ---------------------------------------------------------------------
        wjt, Ejt, ujt, alpha, E_bar_t, njt, support_true = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=seed
        )
        pjt = alpha + 0.3 * wjt + ujt

        # Paper-style truth for reporting:
        #   - int_true corresponds to E_bar_t (scalar in Section 4; constant across t).
        #   - E_true corresponds to the market-product deviations njt (Lu table "xi").
        int_true = float(np.mean(E_bar_t))
        E_true = njt

        # ---------------------------------------------------------------------
        # Step 2: Simulate individual utilities and aggregate to market outcomes.
        #
        # BasicLuChoiceModel simulates a random-coefficient logit where:
        #   beta_{p,i} ~ Normal(beta_p_true, sigma_true)
        #
        # utilities(...) returns u_{i,j,t}. generate_market converts these to:
        #   sjt, s0t : expected shares (inside and outside) for BLP
        #   qjt, q0t : multinomial draws (counts) used by the shrinkage estimator
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
        # Step 3: Fit BLP under two instrument sets (with cost IV vs without).
        #
        # Both BLP estimators use demand regressors Xjt = (1, pjt, wjt) and differ
        # only in instruments Zjt, aligned with Lu(25) Section 4:
        #
        #   - BLP (with cost IV):    Zjt = (1, wjt, wjt^2, ujt, ujt^2)
        #   - BLP (without cost IV): Zjt = (1, wjt, wjt^2, wjt^3, wjt^4)
        #
        # The sigma search region and optimization controls are provided via the
        # validated BLP config.
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
        print("=== Strong IVs and Estimator built ===")
        blp_strong.fit()
        res_strong = blp_strong.get_results()
        print("=== Strong Estimator fitted ===")

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
        print("=== Weak Estimator fitted ===")

        # ---------------------------------------------------------------------
        # Step 4: Fit the Lu shrinkage estimator.
        #
        # The estimator samples from the posterior over:
        #   - beta_p, beta_w, sigma (via r = log(sigma))
        #   - market shocks E_bar_t and market-product shocks njt[t,j]
        #   - sparsity indicators gamma[t,j] and inclusion rates phi[t]
        #
        # Proposal scales are tuned once using short pilot runs, then frozen.
        # ---------------------------------------------------------------------
        shrink = LuShrinkageEstimator(
            pjt=pjt,
            wjt=wjt,
            qjt=qjt,
            q0t=q0t,
            posterior_config=posterior_config,
        )
        print("=== Shrinkage Estimator built ===")
        shrink.fit(config=fit_config)
        res_shrink = shrink.get_results()
        print("=== Shrinkage Estimator fitted ===")

        # ---------------------------------------------------------------------
        # Step 5: Compare estimates to ground truth.
        # ---------------------------------------------------------------------
        print("=== Strong BLP Estimator Results ===")
        print_assessment(
            results=res_strong,
            int_true=int_true,
            xi_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )

        print("=== Weak BLP Estimator Results ===")
        print_assessment(
            results=res_weak,
            int_true=int_true,
            xi_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )

        print("=== Shrinkage Estimator Results ===")
        print_assessment(
            results=res_shrink,
            int_true=int_true,
            xi_true=E_true,
            sigma_true=sigma_true,
            support_true=support_true,
        )


if __name__ == "__main__":
    main()
