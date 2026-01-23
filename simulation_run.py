import numpy as np
from market_shock_estimators.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from market_shock_estimators.assess_estimator import (
    assess_estimator_results,
    print_assessment,
)

from market_shock_estimators.lu_shrinkage import LuShrinkageEstimator
from datasets.dgp import (
    generate_market_conditions,
    BasicLuChoiceModel,
    generate_market,
)


def main():
    T = 25
    J = 15
    N = 1000
    beta_p = -1.0
    beta_w = 0.5
    sigma = 1.5
    seed = 123

    rng = np.random.default_rng(seed)
    R = 200  # number of simulation draws for RC integration (minimal, can change later)
    draws = rng.standard_normal(R)

    DGP_SPECS = {
        1: {
            "name": "Lu DGP1",
            "sparsity": "sparse",
            "endogeneity": "off",
            "shock_dist": "discrete - sparse",
        },
        2: {
            "name": "Lu DGP2",
            "sparsity": "sparse",
            "endogeneity": "on",
            "shock_dist": "discrete - sparse",
        },
        3: {
            "name": "Lu DGP3",
            "sparsity": "dense",
            "endogeneity": "off",
            "shock_dist": "normal",
        },
        4: {
            "name": "Lu DGP4",
            "sparsity": "dense",
            "endogeneity": "on",
            "shock_dist": "normal",
        },
    }

    for dgp_type in (1, 2, 3, 4):

        spec = DGP_SPECS[dgp_type]
        print(f"=== Running DGP {dgp_type} ({spec['name']}) simulation ===")
        print(
            "[SIM]"
            f"T={T}, J={J}, N={N} | "
            f"sparsity={spec['sparsity']} | "
            f"endogeneity={spec['endogeneity']} | "
            f"shock_dist={spec['shock_dist']} | "
            f"RC_draws={R} | "
            f"seed={seed}"
        )

        wjt, E_bar_t, njt, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=seed
        )

        pjt = alpha + 0.3 * wjt + ujt

        model = BasicLuChoiceModel(
            N=N, beta_p=beta_p, beta_w=beta_w, sigma=sigma, seed=seed
        )

        uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

        sjt, s0t, qjt, q0t = generate_market(
            uijt,
            N=N,
            seed=seed,
        )

        # -----------------------------
        # Sanity checks (compact)
        # -----------------------------
        sum_inside = sjt.sum(axis=1)
        share_id_err = np.max(np.abs(sum_inside + s0t - 1.0))

        print(
            "[SIM] Data summary | "
            f"p:[{pjt.min():.3f},{pjt.max():.3f}] "
            f"w:[{wjt.min():.3f},{wjt.max():.3f}] | "
            f"s:[{sjt.min():.4f},{sjt.max():.4f}] "
            f"s0:[{s0t.min():.4f},{s0t.max():.4f}] | "
            f"sum_s:[{sum_inside.min():.4f},{sum_inside.max():.4f}] | "
            f"max|s+s0-1|={share_id_err:.2e}"
        )

        # ------------------------------------------------------------
        # BLP estimation: recover E_hat_jt
        # ------------------------------------------------------------

        print("=== Running BLP estimators ===")

        # Construct regressors X_jt
        # delta_jt = beta_p * p_jt + beta_w * w_jt + E_jt
        # Include constant if desired
        Xjt = np.stack(
            [
                np.ones_like(pjt),  # constant
                pjt,
                wjt,
            ],
            axis=2,  # shape (T, J, K)
        )

        # grid search parameters
        sigma_min = 1e-3
        sigma_max = 5.0
        grid_step = 25

        # ------------------------------------------------------------
        # Strong IV BLP (cost instruments)
        # ------------------------------------------------------------

        print("=== BLP with strong (cost-based) instruments ===")

        Zjt_strong = build_strong_IVs(
            wjt=wjt,
            ujt=ujt,
        )

        blp_cost = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            Xjt=Xjt,
            Zjt=Zjt_strong,
            v_draws=draws,
        )

        blp_cost.fit(
            sigma_init=1.0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            grid_step=grid_step,
        )
        res_cost = blp_cost.get_results()

        print("=== Strong-IV BLP completed ===")

        # ------------------------------------------------------------
        # Weak IV BLP (standard BLP instruments)
        # ------------------------------------------------------------

        print("=== BLP with weak (non-cost) instruments ===")

        Zjt_weak = build_weak_IVs(wjt=wjt)

        blp_weak = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            Xjt=Xjt,
            Zjt=Zjt_weak,
            v_draws=draws,
        )

        blp_weak.fit(
            sigma_init=1.0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            grid_step=grid_step,
        )
        res_weak = blp_weak.get_results()

        print("=== Weak-IV BLP completed ===")

        # ------------------------------------------------------------
        # Estimator assessment (works for BLP now, shrinkage later)
        # ------------------------------------------------------------
        ass_cost = assess_estimator_results(
            name="BLP-strong",
            results=res_cost,
            E_true=Ejt,
            sigma_true=sigma,
        )
        ass_weak = assess_estimator_results(
            name="BLP-weak",
            results=res_weak,
            E_true=Ejt,
            sigma_true=sigma,
        )

        # ------------------------------------------------------------
        # Lu shrinkage estimation (posterior sampling)
        # ------------------------------------------------------------
        print("=== Running Lu shrinkage estimator ===")

        shrink = LuShrinkageEstimator(
            x_jt=Xjt,
            q_jt=qjt,
            q0_t=q0t,
            draws=draws,
            price_index=1,  # Xjt = [const, pjt, wjt] so price is index 1
            # Hyperparameters: keep defaults for now; align later with Lu Section 4
            max_iter=1500,
            burn_in=500,
            thin=5,
            seed=seed,
        )

        shrink.fit()
        res_shrink = shrink.get_results()

        ass_shrink = assess_estimator_results(
            name="Lu-shrinkage",
            results=res_shrink,
            E_true=Ejt,
            sigma_true=sigma,
        )

        print("=== Lu shrinkage completed ===")

        print("=== Estimator assessment ===")
        print_assessment(ass_cost)
        print_assessment(ass_weak)
        print_assessment(ass_shrink)


if __name__ == "__main__":
    main()
