from __future__ import annotations

import numpy as np

from datasets.dgp import BasicLuChoiceModel, generate_market, generate_market_conditions
from market_shock_estimators.assess_estimator import (
    assess_estimator_results,
    print_assessment,
)
from market_shock_estimators.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from market_shock_estimators.lu_shrinkage import LuShrinkageEstimator


def main() -> None:
    # -----------------------------
    # Lu (Section 4) simulation spec
    # -----------------------------
    T, J, N = 25, 15, 1000
    beta_p_true, beta_w_true, sigma_true = -1.0, 0.5, 1.5
    seed = 123

    # Monte Carlo integration draws (estimators own RNG; orchestration passes only counts)
    n_draws = 20

    # BLP fit config
    blp_sigma_init = 1.0
    blp_sigma_min = 1e-3
    blp_sigma_max = 5.0
    blp_grid_step = (
        25  # We start the blp estimator optimisation with a coarse grid search
    )

    # Shrinkage run config
    mcmc_max_iter = 50
    mcmc_burn_in = 10
    mcmc_thin = 5

    for dgp_type in (1, 2, 3, 4):
        print(f"=== DGP {dgp_type} ===")

        # -----------------------------
        # DGP (Lu Section 4.1)
        # -----------------------------
        wjt, _, _, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=seed
        )

        # Pricing rule used in this codebase: p = alpha + 0.3 w + u
        pjt = alpha + 0.3 * wjt + ujt

        model = BasicLuChoiceModel(
            N=N,
            beta_p=beta_p_true,
            beta_w=beta_w_true,
            sigma=sigma_true,
            seed=seed,
        )
        print(f"=== Linear model built ===")

        uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

        sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=seed)
        print(f"=== Market generated ===")
        # -----------------------------
        # BLP (strong / weak IV)
        # Orchestration builds IV matrices; estimator owns RC integration draws internally.
        # Demand regressors (Lu-aligned): X = [p, w] (no constant).
        # -----------------------------

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
        print(f"=== Strong IVs and Estimator built ===")
        blp_strong.fit(
            sigma_init=blp_sigma_init,
            sigma_min=blp_sigma_min,
            sigma_max=blp_sigma_max,
            grid_step=blp_grid_step,
        )
        res_strong = blp_strong.get_results()
        print(f"=== Strong Estimator fitted ===")

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
        print(f"=== Weak IVs and Estimator built ===")
        blp_weak.fit(
            sigma_init=blp_sigma_init,
            sigma_min=blp_sigma_min,
            sigma_max=blp_sigma_max,
            grid_step=blp_grid_step,
        )
        res_weak = blp_weak.get_results()
        print(f"=== Weak Estimator fitted ===")

        # -----------------------------
        # Lu shrinkage (posterior sampling)
        # Assumes estimator constructs Z=p[...,None], initializes state, owns TF RNG, and runs MCMC internally.
        # -----------------------------
        shrink = LuShrinkageEstimator(
            pjt=pjt,
            wjt=wjt,
            qjt=qjt,
            q0t=q0t,
            n_draws=n_draws,
            seed=seed,
        )
        print(f"=== Shrinkage Estimator built ===")

        shrink.fit(
            n_iter=mcmc_max_iter,
            burn_in=mcmc_burn_in,
            thin=mcmc_thin,
        )
        res_shrink = shrink.get_results()
        print(f"=== Shrinkage Estimator fitted ===")

        # -----------------------------
        # Assessment
        # -----------------------------
        strong_BLPEst_assessment = assess_estimator_results(
            results=res_strong, E_true=Ejt, sigma_true=sigma_true
        )
        print_assessment(strong_BLPEst_assessment)
        weak_BLPEst_assessment = assess_estimator_results(
            results=res_weak, E_true=Ejt, sigma_true=sigma_true
        )
        print_assessment(weak_BLPEst_assessment)
        shrink_est_assessment = assess_estimator_results(
            results=res_shrink, E_true=Ejt, sigma_true=sigma_true
        )
        print_assessment(shrink_est_assessment)


if __name__ == "__main__":
    main()
