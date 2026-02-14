"""
bonus2/bonus2_evaluate.py

Evaluation utilities for Bonus Q2 (habit + peer + DOW + seasonality) MNL choice model.

Design goals (mirrors stockpiling_evaluate.py):
- Pure NumPy (no TensorFlow).
- Clean, reviewer-friendly summary formatting.
- No trace averaging: theta_hat is treated as the final point estimate passed in.

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
    """Gather p[m,i,t,y[m,i,t]]."""
    idx = y_int[..., None]
    p_true = np.take_along_axis(p, idx, axis=3)[..., 0]
    return p_true


def predictive_metrics_from_choice_probs(
    y_mit: Any,
    p_choice_mntc: Any,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Predictive metrics for multinomial outcomes.

    Returns:
      {
        "nll_per_obs": float,
        "acc": float,
        "avg_p_true": float,
        "share_emp": (J+1,) array,
        "share_pred": (J+1,) array,
        "n_obs": int,
        "J": int,
      }
    """
    y, p, M, N, T, J = _validate_choice_panel_shapes(
        y_mit, p_choice_mntc, "p_choice_mntc"
    )
    n_obs = int(M * N * T)

    p_true = _gather_p_true(y, p)
    p_true_clip = np.clip(p_true, eps, 1.0)
    nll = float(-np.mean(np.log(p_true_clip)))

    y_hat = np.argmax(p, axis=3).astype(np.int64)
    acc = float(np.mean(y_hat == y))

    avg_p_true = float(np.mean(p_true))

    C = J + 1
    share_emp = np.zeros((C,), dtype=np.float64)
    for c in range(C):
        share_emp[c] = np.mean(y == c)
    share_pred = np.mean(p, axis=(0, 1, 2))

    return {
        "nll_per_obs": nll,
        "acc": acc,
        "avg_p_true": avg_p_true,
        "share_emp": share_emp,
        "share_pred": share_pred,
        "n_obs": n_obs,
        "J": int(J),
    }


def baseline_metrics_from_choices(
    y_mit: Any, J: int | None = None, eps: float = 1e-12
) -> dict[str, Any]:
    """
    Baseline model (fallback): constant probabilities equal to empirical shares.
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

    p = np.broadcast_to(share_emp, (M, N, T, C)).copy()
    mets = predictive_metrics_from_choice_probs(y_int, p, eps=eps)

    return {**mets, "baseline_shares": share_emp.copy()}


# =============================================================================
# Parameter recovery
# =============================================================================


_MEAN_RECOVERY_ORDER = [
    # model formulation order (mean-comparable blocks)
    "beta_market_mj",
    "beta_habit_j",
    "beta_peer_j",
    "decay_rate_j",
    "a_m",
    "b_m",
]

_STD_RECOVERY_ORDER = [
    # centered / mean-not-informative blocks, model formulation order
    "beta_dow_m",
    "beta_dow_j",
    "a_j",
    "b_j",
]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parameter_recovery_metrics(
    theta_true: dict[str, Any],
    theta_hat: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute parameter recovery in two sections:
      - mean recovery: rmse, mean_true, mean_hat, bias
      - std  recovery: rmse, std_true,  std_hat,  bias_std

    theta_hat is treated as the final point estimate passed in.
    """
    mean_block: dict[str, Any] = {}
    std_block: dict[str, Any] = {}

    # Mean-comparable
    for k in _MEAN_RECOVERY_ORDER:
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

        mt = float(np.mean(a)) if a.size else 0.0
        mh = float(np.mean(b)) if b.size else 0.0
        mean_block[k] = {
            "rmse": _rmse(a, b),
            "mean_true": mt,
            "mean_hat": mh,
            "bias": float(mh - mt),
        }

    # Std-comparable
    for k in _STD_RECOVERY_ORDER:
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

        st = float(np.std(a)) if a.size else 0.0
        sh = float(np.std(b)) if b.size else 0.0
        std_block[k] = {
            "rmse": _rmse(a, b),
            "std_true": st,
            "std_hat": sh,
            "bias_std": float(sh - st),
        }

    return {"mean": mean_block, "std": std_block}


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
    p_choice_baseline_mntc: Any | None = None,
) -> dict[str, Any]:
    """
    Evaluate fitted probabilities against observed choices.

    Output schema:
      {
        "shape": {...},
        "models": {"baseline":..., "fitted":..., "oracle":... (optional)},
        "deltas": {"fitted_vs_baseline":..., "fitted_vs_oracle":... (optional)},
        "param": {...} (optional),
        "mcmc": {"n_saved": int|None, "accept_rates": dict[str,float]} (optional)
      }
    """
    y, p_hat, M, N, T, J = _validate_choice_panel_shapes(
        y_mit, p_choice_hat_mntc, "p_choice_hat_mntc"
    )

    models: dict[str, Any] = {}

    # Baseline: caller-provided probabilities (preferred) or fallback constant-share baseline.
    if p_choice_baseline_mntc is not None:
        _, p_base, _, _, _, _ = _validate_choice_panel_shapes(
            y, p_choice_baseline_mntc, "p_choice_baseline_mntc"
        )
        models["baseline"] = predictive_metrics_from_choice_probs(y, p_base, eps=eps)
    else:
        models["baseline"] = baseline_metrics_from_choices(y, J=J, eps=eps)

    models["fitted"] = predictive_metrics_from_choice_probs(y, p_hat, eps=eps)

    if p_choice_oracle_mntc is not None:
        _, p_oracle, _, _, _, _ = _validate_choice_panel_shapes(
            y, p_choice_oracle_mntc, "p_choice_oracle_mntc"
        )
        models["oracle"] = predictive_metrics_from_choice_probs(y, p_oracle, eps=eps)

    # Deltas (schema-driven)
    base = models["baseline"]
    fit = models["fitted"]

    deltas: dict[str, dict[str, float]] = {}
    deltas["fitted_vs_baseline"] = {
        "delta_nll": float(base["nll_per_obs"] - fit["nll_per_obs"]),
        "delta_acc": float(fit["acc"] - base["acc"]),
    }

    if "oracle" in models:
        ora = models["oracle"]
        deltas["fitted_vs_oracle"] = {
            "delta_nll": float(fit["nll_per_obs"] - ora["nll_per_obs"]),
            "delta_acc": float(ora["acc"] - fit["acc"]),
        }

    out: dict[str, Any] = {
        "shape": {"M": M, "N": N, "T": T, "J": J, "n_obs": int(M * N * T)},
        "models": models,
        "deltas": deltas,
    }

    # Optional parameter recovery
    if theta_true is not None:
        if theta_hat is None:
            raise ValueError("theta_true provided but theta_hat is None")
        out["param"] = {"hat": parameter_recovery_metrics(theta_true, theta_hat)}

    # Optional MCMC acceptance (schema-driven)
    if mcmc is not None and isinstance(mcmc, dict):
        accept_rates = mcmc.get("accept_rates", mcmc.get("accept", {}))
        n_saved = mcmc.get("n_saved", None)
        out["mcmc"] = {"n_saved": n_saved, "accept_rates": accept_rates}

    return out


# =============================================================================
# Formatting helper
# =============================================================================


def format_evaluation_summary(eval_out: dict[str, Any]) -> str:
    """
    Format evaluation output into a reviewer-friendly report.
    """
    shp = eval_out.get("shape", {})
    models = eval_out.get("models", {})
    deltas = eval_out.get("deltas", {})
    params = eval_out.get("param", None)
    mcmc = eval_out.get("mcmc", None)

    M = shp.get("M")
    N = shp.get("N")
    T = shp.get("T")
    J = shp.get("J")
    n_obs = shp.get("n_obs")

    baseline = models.get("baseline", {})
    fitted = models.get("fitted", {})
    oracle = models.get("oracle", None)

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    if M is not None and N is not None and T is not None and J is not None:
        lines.append(f"data: M={M} N={N} T={T} J={J} | n_obs={n_obs}")
    lines.append("")

    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
        f"{'acc':>8} "
        f"{'p_true':>8} "
        f"{'out_emp':>8} "
        f"{'out_pred':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(tag: str, d: dict[str, Any]) -> str:
        out_emp = (
            float(np.asarray(d["share_emp"])[0]) if "share_emp" in d else float("nan")
        )
        out_pred = (
            float(np.asarray(d["share_pred"])[0]) if "share_pred" in d else float("nan")
        )
        return (
            f"{tag:<10}"
            f"{f6(float(d['nll_per_obs']))} "
            f"{f4(float(d['acc']))} "
            f"{f4(float(d['avg_p_true']))} "
            f"{f4(out_emp)} "
            f"{f4(out_pred)}"
        )

    if baseline:
        lines.append(row("baseline", baseline))
    if fitted:
        lines.append(row("fitted", fitted))
    if oracle is not None:
        lines.append(row("oracle", oracle))

    dvb = deltas.get("fitted_vs_baseline", None)
    if dvb is not None:
        lines.append("")
        lines.append(
            f"gain vs baseline: Δnll={dvb['delta_nll']:.6f} | Δacc={dvb['delta_acc']:.6f}"
        )

    dvo = deltas.get("fitted_vs_oracle", None)
    if dvo is not None:
        lines.append(
            f"fitted - oracle: Δnll={dvo['delta_nll']:.6f} | Δacc={dvo['delta_acc']:.6f}"
        )

    # Parameter recovery (hat): split into mean vs std sections, fixed order
    if params is not None and isinstance(params, dict) and "hat" in params:
        hat = params["hat"]
        if isinstance(hat, dict):
            mean_block = hat.get("mean", {})
            std_block = hat.get("std", {})

            if isinstance(mean_block, dict) and mean_block:
                lines.append("")
                lines.append("parameter recovery (hat, mean stats)")
                p_header = (
                    f"{'param':<14}"
                    f"{'rmse':>10} "
                    f"{'mean_true':>10} "
                    f"{'mean_hat':>10} "
                    f"{'bias':>10}"
                )
                lines.append(p_header)
                lines.append("-" * len(p_header))
                for k in _MEAN_RECOVERY_ORDER:
                    if k not in mean_block:
                        continue
                    pk = mean_block[k]
                    lines.append(
                        f"{k:<14}"
                        f"{f6(float(pk['rmse']))} "
                        f"{f6(float(pk['mean_true']))} "
                        f"{f6(float(pk['mean_hat']))} "
                        f"{f6(float(pk['bias']))}"
                    )

            if isinstance(std_block, dict) and std_block:
                lines.append("")
                lines.append("parameter recovery (hat, std stats)")
                p_header = (
                    f"{'param':<14}"
                    f"{'rmse':>10} "
                    f"{'std_true':>10} "
                    f"{'std_hat':>10} "
                    f"{'bias_std':>10}"
                )
                lines.append(p_header)
                lines.append("-" * len(p_header))
                for k in _STD_RECOVERY_ORDER:
                    if k not in std_block:
                        continue
                    pk = std_block[k]
                    lines.append(
                        f"{k:<14}"
                        f"{f6(float(pk['rmse']))} "
                        f"{f6(float(pk['std_true']))} "
                        f"{f6(float(pk['std_hat']))} "
                        f"{f6(float(pk['bias_std']))}"
                    )

    # MCMC acceptance (ordered by model formulation)
    if isinstance(mcmc, dict):
        rates = mcmc.get("accept_rates", {})
        n_saved = mcmc.get("n_saved", None)
        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("mcmc acceptance (elementwise rates)")
            if n_saved is not None:
                lines.append(f"n_saved: {n_saved}")

            a_header = f"{'block':<14}{'rate':>10}"
            lines.append(a_header)
            lines.append("-" * len(a_header))

            # Model-formulation order for blocks (then any extras)
            ordered = [
                "beta_market",
                "beta_habit",
                "beta_peer",
                "decay_rate",
                "beta_dow_m",
                "beta_dow_j",
                "a_m",
                "b_m",
                "a_j",
                "b_j",
            ]
            printed = []
            vals: list[float] = []

            for k in ordered:
                if k in rates:
                    r = float(rates[k])
                    printed.append(k)
                    vals.append(r)
                    lines.append(f"{k:<14}{r:>10.4f}")

            extras = sorted([k for k in rates.keys() if k not in set(printed)])
            for k in extras:
                r = float(rates[k])
                vals.append(r)
                lines.append(f"{k:<14}{r:>10.4f}")

            if vals:
                lines.append(f"mean rate: {float(np.mean(vals)):.4f}")

    return "\n".join(lines)
