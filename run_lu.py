"""Run the Lu simulation experiment and compare the fitted estimators."""

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
    """Convert tensor-valued result entries into NumPy-backed output."""

    # Normalize mixed TensorFlow and Python result dictionaries for downstream assessment.
    converted = {}
    for key, value in results.items():
        if isinstance(value, tf.Tensor):
            converted[key] = value.numpy()
        else:
            converted[key] = value
    return converted


def _normalize_results_for_assessment(results: dict) -> dict:
    """
    Standardize estimator outputs into the common assessment result format.

    Accepted estimator output formats:
      1. Full shock available directly:
         - E_full_hat
         - optionally E_bar_hat / njt_hat
      2. Decomposition available directly:
         - E_bar_hat
         - njt_hat
      3. Residual-style BLP output:
         - E_hat
         - int_hat

    Important:
      - For the BLP benchmark, E_hat is the post-intercept residual shock.
      - The Lu assessment compares against the full shock object.
      - Therefore, when only (int_hat, E_hat) are available, reconstruct:
            E_full_hat = int_hat + E_hat
        before deriving E_bar_hat and njt_hat.
    """

    # Start by converting any tensor outputs into NumPy form.
    out = _to_numpy_results(results)

    # Read whichever shock representation the estimator returned.
    E_bar_hat = out.get("E_bar_hat")
    njt_hat = out.get("njt_hat")
    E_full_hat = out.get("E_full_hat")
    E_hat = out.get("E_hat")
    int_hat = out.get("int_hat")

    # When a full shock matrix is available, reconstruct the missing decomposition if needed.
    if E_full_hat is not None:
        E_full_hat = np.asarray(E_full_hat, dtype=float)
        out["E_full_hat"] = E_full_hat

        if E_bar_hat is None:
            E_bar_hat = np.mean(E_full_hat, axis=1)
        else:
            E_bar_hat = np.asarray(E_bar_hat, dtype=float)

        if njt_hat is None:
            njt_hat = E_full_hat - E_bar_hat[:, None]
        else:
            njt_hat = np.asarray(njt_hat, dtype=float)

        out["E_bar_hat"] = E_bar_hat
        out["njt_hat"] = njt_hat
        return out

    # When the decomposed representation is already available, rebuild the full shock matrix.
    if E_bar_hat is not None and njt_hat is not None:
        E_bar_hat = np.asarray(E_bar_hat, dtype=float)
        njt_hat = np.asarray(njt_hat, dtype=float)
        out["E_bar_hat"] = E_bar_hat
        out["njt_hat"] = njt_hat
        out["E_full_hat"] = E_bar_hat[:, None] + njt_hat
        return out

    # When only a residual-style shock estimate is available, recover the full shock first.
    if E_hat is not None:
        E_hat = np.asarray(E_hat, dtype=float)

        if int_hat is None:
            raise ValueError(
                "Estimator results include E_hat but not int_hat. "
                "This is ambiguous for assessment because E_hat may be a "
                "post-intercept residual rather than the full shock."
            )

        E_full_hat = float(int_hat) + E_hat
        E_bar_hat = np.mean(E_full_hat, axis=1)
        njt_hat = E_full_hat - E_bar_hat[:, None]

        out["E_bar_hat"] = E_bar_hat
        out["njt_hat"] = njt_hat
        out["E_full_hat"] = E_full_hat
        return out

    return out


def main() -> None:
    """Run the Lu synthetic-market comparison across the four DGP settings."""

    # Set the shared simulation and estimation controls used across all DGP runs.
    T, J, N = 25, 15, 1000
    beta_p_true, beta_w_true, sigma_true = -1.0, 0.5, 1.5
    seed = 123
    chain_seed = tf.constant([seed, 0], dtype=tf.int32)
    n_draws = 500

    # Configure the BLP benchmark used for both the strong-IV and weak-IV comparisons.
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

    # Set the prior and numerical controls for the shrinkage posterior.
    posterior_config = LuPosteriorConfig(
        n_draws=n_draws,
        seed=seed,
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

    # Set the chain length, proposal scales, and tuning controls for the shrinkage sampler.
    shrinkage_config = LuShrinkageConfig(
        num_results=50000,
        num_burnin_steps=0,
        chunk_size=5000,
        k_beta=0.05,
        k_r=0.05,
        k_E_bar=0.05,
        k_njt=0.02,
        pilot_length=100,
        target_low=0.3,
        target_high=0.5,
        max_rounds=20,
        factor=1.5,
    )

    # Repeat the experiment across the four Lu DGP variants.
    for dgp_type in (1, 2, 3, 4):
        print(f"=== DGP {dgp_type} ===")

        # Generate the market conditions and implied prices for the chosen DGP.
        wjt, Ejt, ujt, alpha, E_bar_t, njt, support_true = generate_market_conditions(
            T=T,
            J=J,
            dgp_type=dgp_type,
            seed=seed,
        )
        pjt = alpha + 0.3 * wjt + ujt

        # Record the true shock decomposition for later assessment.
        E_bar_true = E_bar_t
        njt_true = njt
        E_full_true = Ejt

        # Instantiate the synthetic choice model at the true parameter values.
        model = BasicLuChoiceModel(
            N=N,
            beta_p=beta_p_true,
            beta_w=beta_w_true,
            sigma=sigma_true,
            seed=seed,
        )
        print("=== Linear model built ===")

        # Simulate utilities and observed market counts from the true DGP.
        uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)
        sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=seed)
        print("=== Market generated ===")

        # Fit the first benchmark using the strong instrument construction.
        Zjt_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
        blp_strong = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_strong,
            config=blp_config,
        )
        print("=== Strong IVs and estimator built ===")
        blp_strong.fit()
        res_strong = _normalize_results_for_assessment(blp_strong.get_results())
        print("=== Strong BLP estimator fitted ===")

        # Fit the second benchmark using the weak instrument construction.
        Zjt_weak = build_weak_IVs(wjt=wjt)
        blp_weak = BLPEstimator(
            sjt=sjt,
            s0t=s0t,
            pjt=pjt,
            wjt=wjt,
            Zjt=Zjt_weak,
            config=blp_config,
        )
        print("=== Weak IVs and estimator built ===")
        blp_weak.fit()
        res_weak = _normalize_results_for_assessment(blp_weak.get_results())
        print("=== Weak BLP estimator fitted ===")

        # Run the shrinkage chain on the same simulated market data.
        shrinkage_samples = run_chain(
            pjt=tf.convert_to_tensor(pjt, dtype=tf.float64),
            wjt=tf.convert_to_tensor(wjt, dtype=tf.float64),
            qjt=tf.convert_to_tensor(qjt, dtype=tf.float64),
            q0t=tf.convert_to_tensor(q0t, dtype=tf.float64),
            posterior_config=posterior_config,
            shrinkage_config=shrinkage_config,
            seed=chain_seed,
        )

        # Convert posterior sample summaries into the common assessment format.
        res_shrink = _normalize_results_for_assessment(
            summarize_samples(shrinkage_samples)
        )
        print("=== Shrinkage sampler run ===")

        # Evaluate all three estimators against the same true parameter and shock targets.
        print("=== Strong BLP Estimator Results ===")
        print_assessment(
            results=res_strong,
            beta_p_true=beta_p_true,
            beta_w_true=beta_w_true,
            sigma_true=sigma_true,
            E_bar_true=E_bar_true,
            njt_true=njt_true,
            E_full_true=E_full_true,
            support_true=support_true,
        )

        print("=== Weak BLP Estimator Results ===")
        print_assessment(
            results=res_weak,
            beta_p_true=beta_p_true,
            beta_w_true=beta_w_true,
            sigma_true=sigma_true,
            E_bar_true=E_bar_true,
            njt_true=njt_true,
            E_full_true=E_full_true,
            support_true=support_true,
        )

        print("=== Shrinkage Estimator Results ===")
        print_assessment(
            results=res_shrink,
            beta_p_true=beta_p_true,
            beta_w_true=beta_w_true,
            sigma_true=sigma_true,
            E_bar_true=E_bar_true,
            njt_true=njt_true,
            E_full_true=E_full_true,
            support_true=support_true,
        )


if __name__ == "__main__":
    main()
