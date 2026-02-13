"""
bonus2/bonus2_evaluate.py

Evaluation utilities for Bonus Q2 (habit + peer + DOW + seasonality) MNL choice model.

This mirrors the stockpiling_evaluate.py architecture:
  - pure NumPy
  - baseline vs fitted vs oracle comparisons
  - NLL-centric metrics with a few simple secondary metrics
  - optional parameter recovery metrics
  - optional MCMC acceptance reporting
  - a compact formatting helper

Conventions:
  - y_mit: integer choices, shape (M,N,T), values in {0..J}
      0 = outside, 1..J = inside products
  - p_choice_mntc: choice probabilities, shape (M,N,T,J+1), last axis c=0..J
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Validation
# =============================================================================


def _require_ndim(a: np.ndarray, ndim: int, name: str) -> None:
    if a.ndim != ndim:
        raise ValueError(
            f"{name}: expected ndim={ndim}, got ndim={a.ndim} shape={a.shape}"
        )


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: expected all finite, got non-finite entries")


def _validate_choice_panel_shapes(
    y_mit: Any,
    p_choice_mntc: Any,
    name_probs: str,
    tol_sum: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    y = np.asarray(y_mit)
    _require_ndim(y, 3, "y_mit")
    M, N, T = y.shape
    if M < 1 or N < 1 or T < 1:
        raise ValueError(f"y_mit: expected shape (M,N,T) with all >=1, got {y.shape}")

    try:
        y_int = y.astype(np.int64, copy=False)
    except Exception as e:
        raise ValueError(
            f"y_mit: expected integer-like, conversion failed ({e})"
        ) from e

    p = np.asarray(p_choice_mntc, dtype=np.float64)
    _require_ndim(p, 4, name_probs)
    _require_finite(p, name_probs)

    if p.shape[0] != M or p.shape[1] != N or p.shape[2] != T:
        raise ValueError(
            f"{name_probs}: expected leading shape (M,N,T)=({M},{N},{T}), got {p.shape[:3]}"
        )
    C = int(p.shape[3])
    if C < 2:
        raise ValueError(
            f"{name_probs}: expected last dim C>=2 (outside + >=1 product), got {C}"
        )
    J = C - 1

    if y_int.size:
        mn = int(y_int.min())
        mx = int(y_int.max())
        if mn < 0 or mx > J:
            raise ValueError(
                f"y_mit: expected values in [0,{J}] (0=outside, 1..J=inside), got min={mn}, max={mx}"
            )

    # Probabilities in [0,1] (allow small numerical slack)
    if (p < -1e-12).any() or (p > 1.0 + 1e-12).any():
        mn = float(p.min())
        mx = float(p.max())
        raise ValueError(
            f"{name_probs}: expected values in [0,1], got min={mn}, max={mx}"
        )

    # Probabilities sum to 1
    s = p.sum(axis=3)
    max_dev = float(np.max(np.abs(s - 1.0)))
    if max_dev > tol_sum:
        raise ValueError(
            f"{name_probs}: probs do not sum to 1; max|sum-1|={max_dev:.3e}"
        )

    return y_int, p, M, N, T, J


# =============================================================================
# Metrics
# =============================================================================


def _gather_p_true(y_int: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Gather p[m,i,t,y[m,i,t]].

    Args:
      y_int: (M,N,T) int64 in [0..J]
      p:     (M,N,T,J+1)
    Returns:
      p_true: (M,N,T) float64
    """
    # Use take_along_axis over last dimension
    idx = y_int[..., None]
    p_true = np.take_along_axis(p, idx, axis=3)[..., 0]
    return p_true


def predictive_metrics_from_choice_probs(
    y_mit: Any,
    p_choice_mntc: Any,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute predictive metrics for multinomial outcomes.

    Returns:
      {
        "nll_per_obs": float,
        "brier": float,
        "acc": float,
        "avg_p_true": float,
        "share_emp": (J+1,) array,
        "share_pred": (J+1,) array,
      }
    """
    y, p, M, N, T, J = _validate_choice_panel_shapes(
        y_mit, p_choice_mntc, "p_choice_mntc"
    )
    n_obs = M * N * T

    p_true = _gather_p_true(y, p)
    p_true_clip = np.clip(p_true, eps, 1.0)
    nll = float(-np.mean(np.log(p_true_clip)))

    # Multiclass Brier score: mean over obs of sum_c (p_c - 1{y=c})^2
    # Implement by: sum_c p_c^2 - 2*p_true + 1
    p2_sum = np.sum(p * p, axis=3)
    brier = float(np.mean(p2_sum - 2.0 * p_true + 1.0))

    # Top-1 accuracy
    y_hat = np.argmax(p, axis=3).astype(np.int64)
    acc = float(np.mean(y_hat == y))

    avg_p_true = float(np.mean(p_true))

    # Shares
    C = J + 1
    share_emp = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        share_emp[c] = np.mean(y == c)
    share_pred = np.mean(p, axis=(0, 1, 2))  # (C,)

    return {
        "nll_per_obs": nll,
        "brier": brier,
        "acc": acc,
        "avg_p_true": avg_p_true,
        "share_emp": share_emp,
        "share_pred": share_pred,
        "n_obs": int(n_obs),
        "J": int(J),
    }


def baseline_metrics_from_choices(
    y_mit: Any, J: int | None = None, eps: float = 1e-12
) -> dict[str, Any]:
    """
    Baseline model: constant probabilities equal to empirical shares.

    Args:
      y_mit: (M,N,T) int in [0..J]
      J: optional, if None inferred from max(y)
    """
    y = np.asarray(y_mit)
    _require_ndim(y, 3, "y_mit")
    try:
        y_int = y.astype(np.int64, copy=False)
    except Exception as e:
        raise ValueError(
            f"y_mit: expected integer-like, conversion failed ({e})"
        ) from e

    if J is None:
        J = int(y_int.max())
    if J < 1:
        raise ValueError(f"baseline: expected J>=1, got J={J}")

    M, N, T = y_int.shape
    C = J + 1

    share_emp = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        share_emp[c] = np.mean(y_int == c)

    # Constant probs for all observations: p = share_emp
    p = np.broadcast_to(share_emp, (M, N, T, C)).copy()

    mets = predictive_metrics_from_choice_probs(y_int, p, eps=eps)
    return {
        "nll_per_obs": mets["nll_per_obs"],
        "brier": mets["brier"],
        "acc": mets["acc"],
        "avg_p_true": mets["avg_p_true"],
        "share_emp": share_emp,
        "share_pred": share_emp.copy(),
        "baseline_shares": share_emp.copy(),
        "n_obs": mets["n_obs"],
        "J": J,
    }


# =============================================================================
# Parameter recovery
# =============================================================================


def parameter_metrics(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
    order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute basic parameter recovery metrics per key:
      rmse, mean_true, mean_hat, bias

    Args:
      theta_true, theta_hat: dict of arrays
      order: optional ordered list of keys to include.
    """
    if order is None:
        order = [
            "beta_habit_j",
            "beta_peer_j",
            "decay_rate_j",
            "beta_market_mj",
            "beta_dow_m",
            "beta_dow_j",
            "a_m",
            "b_m",
            "a_j",
            "b_j",
        ]

    out: dict[str, Any] = {}
    for k in order:
        if k not in theta_true:
            raise ValueError(f"theta_true missing key '{k}'")
        if k not in theta_hat:
            raise ValueError(f"theta_hat missing key '{k}'")

        a = np.asarray(theta_true[k], dtype=np.float64)
        b = np.asarray(theta_hat[k], dtype=np.float64)

        if a.shape != b.shape:
            raise ValueError(
                f"param '{k}': shape mismatch true={a.shape} vs hat={b.shape}"
            )
        if a.size == 0:
            rmse = 0.0
        else:
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        mean_true = float(np.mean(a)) if a.size else 0.0
        mean_hat = float(np.mean(b)) if b.size else 0.0
        bias = float(mean_hat - mean_true)

        out[k] = {
            "shape": tuple(a.shape),
            "rmse": rmse,
            "mean_true": mean_true,
            "mean_hat": mean_hat,
            "bias": bias,
        }

    return out


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_bonus2(
    y_mit: Any,
    p_choice_hat_mntc: Any,
    theta_hat: dict[str, Any] | None = None,
    theta_true: dict[str, Any] | None = None,
    p_choice_oracle_mntc: Any | None = None,
    mcmc: dict[str, Any] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Evaluate fitted probabilities against observed choices.

    Args:
      y_mit: (M,N,T)
      p_choice_hat_mntc: (M,N,T,J+1)
      theta_hat: optional dict of recovered parameters
      theta_true: optional dict of true parameters
      p_choice_oracle_mntc: optional oracle probabilities (truth model)
      mcmc: optional dict with acceptance rates or diagnostics
      eps: clip for log in NLL

    Returns:
      dict with keys:
        shape, models, deltas, param (optional), mcmc (optional)
    """
    y, p_hat, M, N, T, J = _validate_choice_panel_shapes(
        y_mit, p_choice_hat_mntc, "p_choice_hat_mntc"
    )

    baseline = baseline_metrics_from_choices(y, J=J, eps=eps)
    fitted = predictive_metrics_from_choice_probs(y, p_hat, eps=eps)

    models: dict[str, Any] = {"baseline": baseline, "fitted": fitted}

    if p_choice_oracle_mntc is not None:
        _, p_oracle, _, _, _, _ = _validate_choice_panel_shapes(
            y, p_choice_oracle_mntc, "p_choice_oracle_mntc"
        )
        oracle = predictive_metrics_from_choice_probs(y, p_oracle, eps=eps)
        models["oracle"] = oracle

    deltas: dict[str, Any] = {
        "nll_gain_fitted_vs_baseline": models["baseline"]["nll_per_obs"]
        - models["fitted"]["nll_per_obs"],
        "brier_gain_fitted_vs_baseline": models["baseline"]["brier"]
        - models["fitted"]["brier"],
        "acc_gain_fitted_vs_baseline": models["fitted"]["acc"]
        - models["baseline"]["acc"],
    }

    if "oracle" in models:
        deltas.update(
            {
                "nll_gap_fitted_minus_oracle": models["fitted"]["nll_per_obs"]
                - models["oracle"]["nll_per_obs"],
                "brier_gap_fitted_minus_oracle": models["fitted"]["brier"]
                - models["oracle"]["brier"],
                "acc_gap_fitted_minus_oracle": models["oracle"]["acc"]
                - models["fitted"]["acc"],
            }
        )

    out: dict[str, Any] = {
        "shape": {"M": M, "N": N, "T": T, "J": J, "n_obs": int(M * N * T)},
        "models": models,
        "deltas": deltas,
    }

    if theta_true is not None and theta_hat is not None:
        out["param"] = parameter_metrics(theta_true=theta_true, theta_hat=theta_hat)

    if mcmc is not None:
        out["mcmc"] = mcmc

    return out


# =============================================================================
# Formatting
# =============================================================================


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """
    Create a human-readable summary string.

    Includes:
      - baseline vs fitted vs oracle (if present) NLL/Brier/Acc
      - key deltas
      - outside share empirical vs predicted (fitted)
      - parameter recovery table (if present)
      - mcmc acceptance (if present)
    """
    shape = eval_out.get("shape", {})
    M = shape.get("M", "?")
    N = shape.get("N", "?")
    T = shape.get("T", "?")
    J = shape.get("J", "?")
    n_obs = shape.get("n_obs", "?")

    lines = []
    lines.append(f"=== Bonus2 evaluation ===")
    lines.append(f"panel: M={M}, N={N}, T={T}, J={J} | n_obs={n_obs}")
    lines.append("")

    models = eval_out["models"]
    order = ["baseline", "fitted"] + (["oracle"] if "oracle" in models else [])
    lines.append("models:")
    for name in order:
        m = models[name]
        lines.append(
            f"  {name:8s} | nll={_fmt(m['nll_per_obs'])} | brier={_fmt(m['brier'])} | acc={_fmt(m['acc'])} | avg_p_true={_fmt(m['avg_p_true'])}"
        )

    lines.append("")
    deltas = eval_out.get("deltas", {})
    for k, v in deltas.items():
        lines.append(f"{k}: {_fmt(float(v))}")

    # Shares: outside option (c=0)
    fitted = models["fitted"]
    share_emp = np.asarray(fitted["share_emp"])
    share_pred = np.asarray(fitted["share_pred"])
    if share_emp.size > 0:
        lines.append("")
        lines.append(
            f"outside_share: emp={_fmt(float(share_emp[0]))} | pred={_fmt(float(share_pred[0]))}"
        )

    # Parameter recovery
    if "param" in eval_out:
        lines.append("")
        lines.append("parameter recovery (rmse | mean_true -> mean_hat):")
        param = eval_out["param"]
        for k, stats in param.items():
            lines.append(
                f"  {k:14s} | rmse={_fmt(stats['rmse'])} | {_fmt(stats['mean_true'])} -> {_fmt(stats['mean_hat'])} | bias={_fmt(stats['bias'])}"
            )

    # MCMC acceptance
    if "mcmc" in eval_out:
        mcmc = eval_out["mcmc"]
        if (
            isinstance(mcmc, dict)
            and "accept" in mcmc
            and isinstance(mcmc["accept"], dict)
        ):
            lines.append("")
            lines.append("acceptance rates:")
            for k in sorted(mcmc["accept"].keys()):
                lines.append(f"  {k:12s}: {_fmt(float(mcmc['accept'][k]))}")

    return "\n".join(lines)
