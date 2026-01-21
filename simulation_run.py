import numpy as np
from market_shock_estimators.blp import BLPEstimator, build_strong_IVs, build_weak_IVs

from datasets.dgp import (
    generate_market_conditions,
    BasicLuChoiceModel,
    generate_market_shares,
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

    for dgp_type in (1, 2, 3, 4):

        wjt, E_bar_t, njt, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=seed
        )

        pjt = alpha + 0.3 * wjt + ujt

        model = BasicLuChoiceModel(
            N=N, beta_p=beta_p, beta_w=beta_w, sigma=sigma, seed=seed
        )

        uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)
        sjt, s0t = generate_market_shares(uijt)
        print("market shares calculated")

        print("=== Sanity check: data ranges ===")
        print("pjt:   min", pjt.min(), "max", pjt.max())
        print("wjt:   min", wjt.min(), "max", wjt.max())
        print("sjt:   min", sjt.min(), "max", sjt.max())
        print("s0t:   min", s0t.min(), "max", s0t.max())
        print(
            "sum sjt (per market): min",
            sjt.sum(axis=1).min(),
            "max",
            sjt.sum(axis=1).max(),
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

        blp_cost.fit(sigma_init=1.0)
        E_hat_cost = blp_cost.get_E_hat()

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

        blp_weak.fit(sigma_init=1.0)
        E_hat_weak = blp_weak.get_E_hat()

        print("=== Weak-IV BLP completed ===")

        # ------------------------------------------------------------
        # Simple sanity checks (no evaluation logic yet)
        # ------------------------------------------------------------

        print("=== BLP output sanity check ===")
        print("E_hat_cost: min", E_hat_cost.min(), "max", E_hat_cost.max())
        print("E_hat_weak: min", E_hat_weak.min(), "max", E_hat_weak.max())


if __name__ == "__main__":
    main()
