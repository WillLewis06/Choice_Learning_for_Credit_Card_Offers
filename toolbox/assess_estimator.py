import numpy as np


def print_assessment(
    results: dict,
    E_true: np.ndarray,
    sigma_true: float | None = None,
) -> None:
    """
    Estimator-agnostic assessment for Lu Section 4 simulations.

    Expected results keys (from estimator.get_results()):
      - "success": bool
      - "E_hat": np.ndarray or None, shape (T,J)
      - "sigma_hat": float or None

    Prints (Lu-comparable core):
      - rmse, mae, bias, corr
      - sigma_hat (and abs/rel error if sigma_true is provided)

    Prints (debug-friendly extras):
      - std_ratio (std(E_hat)/std(E_true))
      - null_rmse (E_hat = 0 baseline) and improve = null_rmse - rmse
      - E_true/E_hat norms and means
      - degeneracy / non-finite / shape mismatch flags
    """
    E_hat = results.get("E_hat", None)
    sigma_hat = results.get("sigma_hat", None)

    # Flatten for scalar metrics
    e = E_true.reshape(-1).astype(float)
    eh = E_hat.reshape(-1).astype(float)
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
        f"rmse={rmse:.4f} mae={mae:.4f} bias={bias:.4f} corr={corr:.4f} "
        f"| std_ratio={std_ratio:.4f} diff_sd={diff_std:.4f} "
        f"| null_rmse={rmse_null:.4f} improve={rmse_improvement:.4f} "
        f"| E_true_norm={E_true_norm:.4f} E_hat_norm={E_hat_norm:.4f} "
        f"| mean(E_true)={E_true_mean:.4f} mean(E_hat)={E_hat_mean:.4f}"
        f"{sigma_part}" + (" | degenerate_E_hat" if degenerate_E_hat else "")
    )
