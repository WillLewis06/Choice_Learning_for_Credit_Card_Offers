"""
End-to-end simulation harness for comparing BLP vs Lu shrinkage on synthetic markets.

This script:
  1) Generates synthetic market data under multiple DGP variants.
  2) Fits two BLP estimators (strong-IV and weak-IV).
  3) Fits the Lu shrinkage estimator (Bayesian sparse market-product shocks).
  4) Compares recovered shocks and sigma against ground truth.
"""

from __future__ import annotations

import os

# Reduce TensorFlow logging noise in terminal output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

from datasets.lu_dgp import (
    BasicLuChoiceModel,
    generate_market,
    generate_market_conditions,
)
from lu.blp.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from lu.shrinkage.lu_shrinkage import LuShrinkageEstimator
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
    n_draws = 20

    # -------------------------------------------------------------------------
    # BLP fitting configuration (shared across strong-IV and weak-IV runs)
    # -------------------------------------------------------------------------
    blp_sigma_init = 1.0
    blp_sigma_min = 1e-3
    blp_sigma_max = 5.0
    blp_grid_step = 25

    # -------------------------------------------------------------------------
    # Lu shrinkage configuration: MCMC length plus proposal-scale tuning controls
    # -------------------------------------------------------------------------
    shrink_n_iter = 10

    # Tuning targets and multiplicative adjustments for RW-MH vs TMH blocks.
    shrink_target_low = 0.3
    shrink_target_high = 0.5
    shrink_max_rounds = 100
    shrink_factor_rw = 1.3
    shrink_factor_tmh = 1.05

    # Pilot chain length used to tune each proposal scale once.
    shrink_pilot_length = 20

    # Small ridge used inside TMH to stabilize linear algebra.
    shrink_ridge = 1e-6

    # -------------------------------------------------------------------------
    # Loop over DGP variants. Each variant changes sparsity and endogeneity.
    # -------------------------------------------------------------------------
    for dgp_type in (1, 2, 3, 4):
        print(f"=== DGP {dgp_type} ===")

        # ---------------------------------------------------------------------
        # Step 1: Generate market-level primitives and construct prices.
        #
        # generate_market_conditions returns:
        #   wjt   : exogenous characteristic
        #   Ejt   : demand shock (E_bar_t + n_jt)
        #   ujt   : cost shock entering pricing
        #   alpha : endogeneity shifter (0 for exogenous-price DGPs)
        #
        # Pricing rule in this codebase:
        #   pjt = alpha + 0.3 * wjt + ujt
        # ---------------------------------------------------------------------
        wjt, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=seed
        )
        pjt = alpha + 0.3 * wjt + ujt

        # ---------------------------------------------------------------------
        # Step 2: Simulate individual utilities and aggregate to market outcomes.
        #
        # BasicLuChoiceModel simulates a random-coefficient logit where:
        #   beta_{p,i} ~ Normal(beta_p_true, sigma_true)
        #
        # utilities(...) returns u_{i,j,t}. generate_market converts these to:
        #   sjt, s0t : expected shares (inside and outside)
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
        # Step 3: Fit BLP under two instrument sets.
        #
        # Both BLP estimators use the same demand regressors (pjt, wjt) and differ
        # only in Zjt construction:
        #   - strong IVs: uses wjt and ujt (designed to be informative in the DGP)
        #   - weak IVs  : uses only wjt (intentionally less informative)
        #
        # Each estimator internally performs RC integration with n_draws draws.
        # ---------------------------------------------------------------------
        Zjt_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
        blp_strong = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_strong,
            n_draws=n_draws,
            seed=seed,
        )
        print("=== Strong IVs and Estimator built ===")
        blp_strong.fit(
            sigma_init=blp_sigma_init,
            sigma_min=blp_sigma_min,
            sigma_max=blp_sigma_max,
            grid_step=blp_grid_step,
        )
        res_strong = blp_strong.get_results()
        print("=== Strong Estimator fitted ===")

        Zjt_weak = build_weak_IVs(wjt=wjt)
        blp_weak = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_weak,
            n_draws=n_draws,
            seed=seed,
        )
        print("=== Weak IVs and Estimator built ===")
        blp_weak.fit(
            sigma_init=blp_sigma_init,
            sigma_min=blp_sigma_min,
            sigma_max=blp_sigma_max,
            grid_step=blp_grid_step,
        )
        res_weak = blp_weak.get_results()
        print("=== Weak Estimator fitted ===")

        # ---------------------------------------------------------------------
        # Step 4: Fit the Lu shrinkage estimator.
        #
        # This estimator is Bayesian and samples from the posterior over:
        #   - beta_p, beta_w, sigma (via r = log(sigma))
        #   - market shocks E_bar_t and market-product shocks n_{j,t}
        #   - sparsity indicators gamma_{j,t} and inclusion rates phi_t
        #
        # It uses the observed counts (qjt, q0t) and the same (pjt, wjt).
        # Proposal scales are tuned once using short pilot runs, then frozen.
        # ---------------------------------------------------------------------
        shrink = LuShrinkageEstimator(
            pjt=pjt,
            wjt=wjt,
            qjt=qjt,
            q0t=q0t,
            n_draws=n_draws,
            seed=seed,
        )
        print("=== Shrinkage Estimator built ===")
        shrink.fit(
            n_iter=shrink_n_iter,
            pilot_length=shrink_pilot_length,
            ridge=shrink_ridge,
            target_low=shrink_target_low,
            target_high=shrink_target_high,
            max_rounds=shrink_max_rounds,
            factor_rw=shrink_factor_rw,
            factor_tmh=shrink_factor_tmh,
        )
        res_shrink = shrink.get_results()
        print("=== Shrinkage Estimator fitted ===")

        # ---------------------------------------------------------------------
        # Step 5: Compare estimates to ground truth.
        #
        # print_assessment expects:
        #   - results: estimator outputs including recovered shocks E_hat
        #   - E_true : ground-truth demand shocks used in simulation (Ejt)
        #   - sigma_true: ground-truth sigma for price heterogeneity
        # ---------------------------------------------------------------------
        print("=== Strong BLP Estimator Results ===")
        print_assessment(results=res_strong, E_true=Ejt, sigma_true=sigma_true)

        print("=== Weak BLP Estimator Results ===")
        print_assessment(results=res_weak, E_true=Ejt, sigma_true=sigma_true)

        print("=== Shrinkage Estimator Results ===")
        print_assessment(results=res_shrink, E_true=Ejt, sigma_true=sigma_true)


if __name__ == "__main__":
    main()
