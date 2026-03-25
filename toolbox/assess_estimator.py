import numpy as np


def _fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{float(x):.6f}"


def _scalar_metrics(true_value: float, hat_value: float) -> dict[str, float]:
    true_f = float(true_value)
    hat_f = float(hat_value)
    return {
        "hat": hat_f,
        "true": true_f,
        "abs_err": abs(hat_f - true_f),
    }


def _array_metrics(true_value: np.ndarray, hat_value: np.ndarray) -> dict[str, float]:
    true_arr = np.asarray(true_value, dtype=float)
    hat_arr = np.asarray(hat_value, dtype=float)

    if true_arr.shape != hat_arr.shape:
        raise ValueError(
            f"shape mismatch for array assessment: {true_arr.shape} vs {hat_arr.shape}"
        )

    true_flat = true_arr.reshape(-1)
    hat_flat = hat_arr.reshape(-1)
    diff = hat_flat - true_flat

    rmse = float(np.sqrt(np.mean(diff * diff)))
    bias = float(np.mean(diff))

    true_std = float(np.std(true_flat))
    hat_std = float(np.std(hat_flat))
    if true_std > 0.0 and hat_std > 0.0:
        corr = float(np.corrcoef(hat_flat, true_flat)[0, 1])
    else:
        corr = np.nan

    return {
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
    }


def _support_metrics(
    support_true: np.ndarray,
    gamma_hat: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    support = np.asarray(support_true, dtype=bool)
    gamma = np.asarray(gamma_hat, dtype=float)

    if support.shape != gamma.shape:
        raise ValueError(
            f"support_true vs gamma_hat: shape mismatch {support.shape} vs {gamma.shape}"
        )

    support_hat = gamma >= threshold

    tp = int(np.sum(support_hat & support))
    fp = int(np.sum(support_hat & (~support)))
    fn = int(np.sum((~support_hat) & support))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0.0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _print_scalar_block(name: str, true_value: float, hat_value: float) -> None:
    metrics = _scalar_metrics(true_value=true_value, hat_value=hat_value)
    print(
        f"  {name}: hat={_fmt(metrics['hat'])} "
        f"true={_fmt(metrics['true'])} "
        f"abs_err={_fmt(metrics['abs_err'])}"
    )


def _print_array_block(
    name: str, true_value: np.ndarray, hat_value: np.ndarray
) -> None:
    metrics = _array_metrics(true_value=true_value, hat_value=hat_value)
    print(
        f"  {name}: rmse={_fmt(metrics['rmse'])} "
        f"bias={_fmt(metrics['bias'])} "
        f"corr={_fmt(metrics['corr'])}"
    )


def _print_support_block(support_true: np.ndarray, gamma_hat: np.ndarray) -> None:
    metrics = _support_metrics(support_true=support_true, gamma_hat=gamma_hat)
    print(
        f"  support: precision={_fmt(metrics['precision'])} "
        f"recall={_fmt(metrics['recall'])} "
        f"f1={_fmt(metrics['f1'])}"
    )


def print_assessment(
    results: dict,
    beta_p_true: float,
    beta_w_true: float,
    sigma_true: float,
    E_bar_true: np.ndarray,
    njt_true: np.ndarray,
    E_full_true: np.ndarray,
    support_true: np.ndarray | None = None,
) -> None:
    """
    Estimator-agnostic assessment for Lu simulations.

    Required true inputs:
      - beta_p_true
      - beta_w_true
      - sigma_true
      - E_bar_true
      - njt_true
      - E_full_true

    Expected estimator outputs when available:
      - success
      - beta_p_hat
      - beta_w_hat
      - sigma_hat
      - E_bar_hat
      - njt_hat
      - E_full_hat
      - gamma_hat

    Printed metrics:
      - scalar parameters: estimate, true value, absolute error
      - shock targets: RMSE, bias, correlation
      - support: precision, recall, F1
    """
    success = bool(results.get("success", True))
    print(f"success={success}")

    beta_p_hat = results.get("beta_p_hat")
    beta_w_hat = results.get("beta_w_hat")
    sigma_hat = results.get("sigma_hat")
    E_bar_hat = results.get("E_bar_hat")
    njt_hat = results.get("njt_hat")
    E_full_hat = results.get("E_full_hat")
    gamma_hat = results.get("gamma_hat")

    if beta_p_hat is not None:
        _print_scalar_block("beta_p", true_value=beta_p_true, hat_value=beta_p_hat)
    if beta_w_hat is not None:
        _print_scalar_block("beta_w", true_value=beta_w_true, hat_value=beta_w_hat)
    if sigma_hat is not None:
        _print_scalar_block("sigma", true_value=sigma_true, hat_value=sigma_hat)

    if E_bar_hat is not None:
        _print_array_block("E_bar", true_value=E_bar_true, hat_value=E_bar_hat)
    if njt_hat is not None:
        _print_array_block("n", true_value=njt_true, hat_value=njt_hat)
    if E_full_hat is not None:
        _print_array_block("E_full", true_value=E_full_true, hat_value=E_full_hat)

    if support_true is not None and gamma_hat is not None:
        _print_support_block(support_true=support_true, gamma_hat=gamma_hat)
