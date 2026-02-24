import numpy as np


def print_assessment(
    results: dict,
    int_true: float,
    E_true: np.ndarray,
    sigma_true: float | None = None,
    support_true: np.ndarray | None = None,
) -> None:
    """
    Estimator-agnostic assessment for Lu Section 4 simulations (single-run diagnostics).

    Conventions (aligned to Lu reporting split):
      - "int_true" is the common/intercept component (Lu table "Int").
      - "E_true" is the deviation shock array (Lu table "xi"), shape (T, J).
      - results["int_hat"] is the estimated intercept (scalar).
      - results["E_hat"] is interpreted as E_hat (post-intercept deviations), shape (T, J).

    Expected results keys (from estimator.get_results()):
      - "success": bool (optional)
      - "int_hat": float or None (optional for some estimators)
      - "E_hat": np.ndarray or None, shape (T, J)
      - "sigma_hat": float or None

    Prints (debug-friendly):
      - Int: hat/true/error
      - E: rmse, mae, bias, corr + extras (std_ratio, null_rmse, norms, means)
      - sigma_hat (and abs/rel error if sigma_true is provided)
      - Prob (optional): single-run sparsity agreement if support_true and gamma_hat exist
    """
    success = results.get("success", True)

    int_hat = results.get("int_hat", None)
    E_hat = results.get("E_hat", None)
    sigma_hat = results.get("sigma_hat", None)

    # Intercept diagnostics
    int_hat_f = (
        float(int_hat) if (int_hat is not None and np.isfinite(int_hat)) else np.nan
    )
    int_true_f = float(int_true)
    int_err = float(int_hat_f - int_true_f) if np.isfinite(int_hat_f) else np.nan
    int_abs_err = float(abs(int_err)) if np.isfinite(int_err) else np.nan

    if E_hat is None:
        sigma_part = ""
        if (sigma_hat is not None) and np.isfinite(sigma_hat):
            sigma_part = f" | sigma_hat={float(sigma_hat):.6f}"
            if sigma_true is not None:
                sigma_abs_err = float(abs(float(sigma_hat) - float(sigma_true)))
                sigma_rel_err = (
                    float(sigma_abs_err / float(sigma_true))
                    if sigma_true != 0
                    else np.nan
                )
                sigma_part += (
                    f" (abs_err={sigma_abs_err:.3f}, rel_err={sigma_rel_err:.3f})"
                )

        print(
            f"success={bool(success)} | Int: hat={int_hat_f:.6f} true={int_true_f:.6f} "
            f"err={int_err:.6f} abs_err={int_abs_err:.6f} | E: E_hat=None"
            f"{sigma_part}"
        )
        return

    # E diagnostics
    E = np.asarray(E_true, dtype=float)
    E_hat = np.asarray(E_hat, dtype=float)

    if E.shape != E_hat.shape:
        raise ValueError(f"E_true vs E_hat: shape mismatch {E.shape} vs {E_hat.shape}")

    # Flatten for scalar metrics
    e = E.reshape(-1)
    eh = E_hat.reshape(-1)
    diff = eh - e

    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    e_std = float(np.std(e))
    eh_std = float(np.std(eh))
    std_ratio = float(eh_std / e_std) if e_std > 0 else np.nan

    if (e_std > 0) and (eh_std > 0):
        corr = float(np.corrcoef(eh, e)[0, 1])
    else:
        corr = np.nan

    rmse_null = float(np.sqrt(np.mean(e * e)))
    rmse_improvement = float(rmse_null - rmse)

    # Debug extras (cheap + informative)
    diff_std = float(np.std(diff))
    E_true_norm = float(np.linalg.norm(e))
    E_hat_norm = float(np.linalg.norm(eh))
    E_true_mean = float(np.mean(e))
    E_hat_mean = float(np.mean(eh))
    degenerate_E_hat = bool(eh_std < 1e-12)

    # Optional single-run sparsity agreement ("Prob.")
    prob_part = ""
    gamma_hat = results.get("gamma_hat", None)
    if (support_true is not None) and (gamma_hat is not None):
        support = np.asarray(support_true, dtype=bool)
        gamma = np.asarray(gamma_hat, dtype=float)
        if support.shape != gamma.shape:
            raise ValueError(
                f"support_true vs gamma_hat: shape mismatch {support.shape} vs {gamma.shape}"
            )
        support_hat = gamma >= 0.5
        prob = float(np.mean(support_hat == support))
        prob_part = f" | Prob={prob:.4f}"

    sigma_part = ""
    if (sigma_hat is not None) and np.isfinite(sigma_hat):
        sigma_part = f" | sigma_hat={float(sigma_hat):.6f}"
        if sigma_true is not None:
            sigma_abs_err = float(abs(float(sigma_hat) - float(sigma_true)))
            sigma_rel_err = (
                float(sigma_abs_err / float(sigma_true)) if sigma_true != 0 else np.nan
            )
            sigma_part += f" (abs_err={sigma_abs_err:.3f}, rel_err={sigma_rel_err:.3f})"

    print(
        f"success={bool(success)} | Int: hat={int_hat_f:.6f} true={int_true_f:.6f} "
        f"err={int_err:.6f} abs_err={int_abs_err:.6f} "
        f"| E: rmse={rmse:.4f} mae={mae:.4f} bias={bias:.4f} corr={corr:.4f} "
        f"| std_ratio={std_ratio:.4f} diff_sd={diff_std:.4f} "
        f"| null_rmse={rmse_null:.4f} improve={rmse_improvement:.4f} "
        f"| E_true_norm={E_true_norm:.4f} E_hat_norm={E_hat_norm:.4f} "
        f"| mean(E_true)={E_true_mean:.4f} mean(E_hat)={E_hat_mean:.4f}"
        f"{prob_part}"
        f"{sigma_part}" + (" | degenerate_E_hat" if degenerate_E_hat else "")
    )
