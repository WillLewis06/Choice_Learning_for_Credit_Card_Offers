import numpy as np


def assess_estimator_results(
    *,
    name: str,
    results: dict,
    E_true: np.ndarray,
    sigma_true: float | None = None,
) -> dict:
    """
    Estimator-agnostic assessment for Lu Section 4 simulations.

    Expected results keys (from estimator.get_results()):
      - "success": bool
      - "E_hat": np.ndarray or None, shape (T,J)
      - "sigma_hat": float or None

    Computes:
      - Validity flags (success/finite/shape)
      - RMSE, MAE, bias, std_ratio, corr
      - Null baseline RMSE (E_hat = 0) and improvement
      - Optional sigma absolute/relative error when sigma_true is provided
    """
    out = {"name": name}

    success = bool(results.get("success", False))
    E_hat = results.get("E_hat", None)
    sigma_hat = results.get("sigma_hat", None)

    out["success"] = success
    out["sigma_hat"] = sigma_hat

    # Basic failure handling
    if (not success) or (E_hat is None):
        out.update(
            {
                "ok": False,
                "reason": "success=False or E_hat is None",
                "rmse": None,
                "mae": None,
                "bias": None,
                "corr": None,
                "std_ratio": None,
                "rmse_null": None,
                "rmse_improvement": None,
            }
        )
        return out

    if E_hat.shape != E_true.shape:
        out.update(
            {
                "ok": False,
                "reason": f"shape mismatch: E_hat{E_hat.shape} vs E_true{E_true.shape}",
                "rmse": None,
                "mae": None,
                "bias": None,
                "corr": None,
                "std_ratio": None,
                "rmse_null": None,
                "rmse_improvement": None,
            }
        )
        return out

    if not np.all(np.isfinite(E_hat)):
        nan_count = int(np.isnan(E_hat).sum())
        inf_count = int(np.isinf(E_hat).sum())
        out.update(
            {
                "ok": False,
                "reason": f"non-finite E_hat (nan={nan_count}, inf={inf_count})",
                "rmse": None,
                "mae": None,
                "bias": None,
                "corr": None,
                "std_ratio": None,
                "rmse_null": None,
                "rmse_improvement": None,
            }
        )
        return out

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

    # Pearson correlation (guard against zero-variance cases)
    if (e_std > 0) and (eh_std > 0):
        corr = float(np.corrcoef(eh, e)[0, 1])
    else:
        corr = np.nan

    # Null baseline: E_hat = 0
    rmse_null = float(np.sqrt(np.mean(e * e)))
    rmse_improvement = float(rmse_null - rmse)

    out.update(
        {
            "ok": True,
            "reason": None,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "corr": corr,
            "std_ratio": std_ratio,
            "rmse_null": rmse_null,
            "rmse_improvement": rmse_improvement,
        }
    )

    # Optional sigma accuracy
    if (sigma_true is not None) and (sigma_hat is not None) and np.isfinite(sigma_hat):
        sigma_abs_err = float(abs(float(sigma_hat) - float(sigma_true)))
        sigma_rel_err = (
            float(sigma_abs_err / float(sigma_true)) if sigma_true != 0 else np.nan
        )
    else:
        sigma_abs_err = None
        sigma_rel_err = None

    out["sigma_abs_err"] = sigma_abs_err
    out["sigma_rel_err"] = sigma_rel_err

    # Simple degeneracy flag: nearly-constant E_hat
    out["degenerate_E_hat"] = bool(eh_std < 1e-12)

    return out


def print_assessment(a: dict):
    if not a.get("ok", False):
        print(f"[SIM] {a['name']} | FAILED | {a.get('reason')}")
        return

    sigma_part = ""
    if a.get("sigma_hat", None) is not None and np.isfinite(a["sigma_hat"]):
        sigma_part = f" | sigma_hat={a['sigma_hat']:.6f}" + (
            f" (abs_err={a['sigma_abs_err']:.3f}, rel_err={a['sigma_rel_err']:.3f})"
            if a.get("sigma_abs_err", None) is not None
            else ""
        )

    print(
        f"[SIM] {a['name']} | "
        f"rmse={a['rmse']:.4f} mae={a['mae']:.4f} "
        f"bias={a['bias']:.4f} corr={a['corr']:.4f} "
        f"std_ratio={a['std_ratio']:.4f} "
        f"| null_rmse={a['rmse_null']:.4f} "
        f"improve={a['rmse_improvement']:.4f}"
        f"{sigma_part}"
        + (" | degenerate_E_hat" if a.get("degenerate_E_hat", False) else "")
    )
