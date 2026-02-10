# ching/stockpiling_evaluate.py
#
# Lightweight evaluation utilities for the stockpiling model.
#
# Design goals (simplified vs prior version):
# - No TensorFlow.
# - No posterior / DP / forward-filter mechanics inside evaluation.
# - Keep only:
#     (1) predictive fit metrics: NLL per obs + RMSE on buy probabilities
#     (2) parameter recovery: RMSE + means (no MAE)
#     (3) MCMC acceptance summaries (as returned by StockpilingEstimator.get_results)
#
# Predictive probabilities:
# - Preferred: pass p_buy_hat_imt (and optionally p_buy_oracle_imt) directly.
# - Backward-compatible fallback: if p_buy_hat_imt is not provided, we compute a cheap
#   myopic approximation p(buy) = sigmoid(u1 - u0), using:
#       u1 = u_scale[m]*u_m[m] - alpha[m,n]*price[s_t] - fc[m,n]
#       u0 = -v[m,n]  (if assume_stockout=True), else 0
#   This avoids DP/filtering but is not the full dynamic model.

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# =============================================================================
# Helpers
# =============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return None


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(np.asarray(x).item())
        except Exception:
            return None


# =============================================================================
# Parameter recovery
# =============================================================================


def parameter_metrics(
    theta_true: dict[str, np.ndarray],
    theta_hat: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Compare true vs fitted constrained parameters.

    For each key in theta_hat (beta, alpha, v, fc, lambda_c, u_scale):
      rmse, mean_true, mean_hat

    If theta_true lacks u_scale, it is treated as ones_like(theta_hat["u_scale"]).
    """
    out: dict[str, dict[str, float]] = {}

    def _rmse(t: np.ndarray, h: np.ndarray) -> float:
        d = (h - t).astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(d * d)))

    for k, hat in theta_hat.items():
        if k == "u_scale" and k not in theta_true:
            true = np.ones_like(np.asarray(hat, dtype=np.float64), dtype=np.float64)
        else:
            if k not in theta_true:
                continue
            true = np.asarray(theta_true[k], dtype=np.float64)

        hat_arr = np.asarray(hat, dtype=np.float64)
        out[k] = {
            "rmse": _rmse(true, hat_arr),
            "mean_true": float(np.mean(true)),
            "mean_hat": float(np.mean(hat_arr)),
        }

    return out


# =============================================================================
# Predictive fit (NLL + RMSE) from probabilities
# =============================================================================


def predictive_metrics_from_probs(
    a_imt: np.ndarray,
    p_buy_imt: np.ndarray,
    p_state_mt: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Compute predictive metrics given predicted buy probabilities.

    Inputs:
      a_imt: (M,N,T) 0/1
      p_buy_imt: (M,N,T) in [0,1]
      p_state_mt: optional (M,T) state indices for by-state summaries
    """
    a = np.asarray(a_imt, dtype=np.float64)
    p = np.asarray(p_buy_imt, dtype=np.float64)

    if a.shape != p.shape:
        raise ValueError(f"a_imt shape {a.shape} must match p_buy_imt shape {p.shape}")

    M, N, T = a.shape
    n_obs = int(M * N * T)

    p = np.clip(p, eps, 1.0 - eps)

    # NLL per obs
    nll = -np.mean(a * np.log(p) + (1.0 - a) * np.log(1.0 - p))

    # RMSE on probability predictions vs realized actions
    rmse_prob = float(np.sqrt(np.mean((p - a) ** 2)))

    buy_rate_emp = float(np.mean(a))
    buy_rate_pred = float(np.mean(p))

    # Baseline: constant p0 = empirical buy rate
    p0 = np.clip(buy_rate_emp, eps, 1.0 - eps)
    baseline_nll = -np.mean(a * np.log(p0) + (1.0 - a) * np.log(1.0 - p0))
    baseline_rmse = float(np.sqrt(np.mean((p0 - a) ** 2)))

    by_state_emp: dict[int, float] = {}
    by_state_pred: dict[int, float] = {}
    rmse_by_state = float("nan")

    if p_state_mt is not None:
        st = np.asarray(p_state_mt)
        if st.shape != (M, T):
            raise ValueError(
                f"p_state_mt must have shape (M,T) = {(M,T)}, got {st.shape}"
            )

        st_vals = np.unique(st.astype(np.int64, copy=False))
        diffs: list[float] = []

        for s in st_vals.tolist():
            mask_mt = st == s  # (M,T) bool
            count_mt = int(mask_mt.sum())
            if count_mt == 0:
                continue

            # Broadcast mask over N via multiplication (no boolean indexing).
            mask_mnt = mask_mt[:, None, :]  # (M,1,T) broadcasts in arithmetic
            den = float(count_mt * N)  # number of (m,n,t) entries in this state

            emp = float(np.sum(a * mask_mnt) / den)
            pred = float(np.sum(p * mask_mnt) / den)

            by_state_emp[int(s)] = emp
            by_state_pred[int(s)] = pred
            diffs.append((pred - emp) ** 2)

        if diffs:
            rmse_by_state = float(np.sqrt(np.mean(diffs)))

    return {
        "shape": {"M": int(M), "N": int(N), "T": int(T), "n_obs": n_obs},
        "nll_per_obs": float(nll),
        "rmse_prob": float(rmse_prob),
        "buy_rate_emp": float(buy_rate_emp),
        "buy_rate_pred": float(buy_rate_pred),
        "buy_rate_by_state_emp": by_state_emp,
        "buy_rate_by_state_pred": by_state_pred,
        "rmse_buy_rate_by_state": float(rmse_by_state),
        "baseline": {
            "p0": float(p0),
            "nll_per_obs": float(baseline_nll),
            "rmse_prob": float(baseline_rmse),
        },
    }


# =============================================================================
# Fallback: cheap myopic probabilities (no DP/filtering)
# =============================================================================


def myopic_buy_probabilities(
    a_imt: np.ndarray,
    p_state_mt: np.ndarray,
    u_m: np.ndarray,
    price_vals: np.ndarray,
    theta: dict[str, np.ndarray],
    assume_stockout: bool = True,
) -> np.ndarray:
    """
    Cheap fallback: p(buy) = sigmoid(u1 - u0) with inventory fixed/ignored.

    u1 = u_scale[m]*u_m[m] - alpha[m,n]*price[s_t] - fc[m,n]
    u0 = -v[m,n] if assume_stockout else 0
    """
    a = np.asarray(a_imt)
    M, N, T = a.shape

    st = np.asarray(p_state_mt, dtype=np.int64)
    if st.shape != (M, T):
        raise ValueError(f"p_state_mt must have shape (M,T) = {(M,T)}, got {st.shape}")

    u_m = np.asarray(u_m, dtype=np.float64).reshape(M)
    price_vals = np.asarray(price_vals, dtype=np.float64)
    price_mt = price_vals[st]  # (M,T)

    alpha = np.asarray(theta["alpha"], dtype=np.float64)  # (M,N)
    fc = np.asarray(theta["fc"], dtype=np.float64)  # (M,N)
    v = np.asarray(theta["v"], dtype=np.float64)  # (M,N)

    u_scale = theta.get("u_scale", None)
    if u_scale is None:
        u_scale = np.ones((M,), dtype=np.float64)
    u_scale = np.asarray(u_scale, dtype=np.float64).reshape(M)

    um_eff = (u_scale * u_m)[:, None, None]  # (M,1,1)
    price_mnt = price_mt[:, None, :]  # (M,1,T)

    u1 = um_eff - alpha[:, :, None] * price_mnt - fc[:, :, None]
    u0 = (-v[:, :, None]) if assume_stockout else 0.0

    return _sigmoid(u1 - u0)


# =============================================================================
# Top-level evaluation
# =============================================================================


def evaluate_stockpiling(
    a_imt: np.ndarray,
    p_state_mt: Optional[np.ndarray] = None,
    u_m: Optional[np.ndarray] = None,
    price_vals: Optional[np.ndarray] = None,
    theta_hat: Optional[dict[str, np.ndarray]] = None,
    theta_true: Optional[dict[str, np.ndarray]] = None,
    p_buy_hat_imt: Optional[np.ndarray] = None,
    p_buy_oracle_imt: Optional[np.ndarray] = None,
    mcmc: Optional[dict[str, Any]] = None,
    eps: float = 1e-12,
    assume_stockout: bool = True,
    **_unused: Any,  # accept legacy args (P_price, I_max, pi_I0, waste_cost, tol, max_iter, etc.)
) -> dict[str, Any]:
    """
    Evaluation wrapper.

    Returns:
      {
        "fit": {...},
        "oracle": {...},   # only if available
        "param": {...},    # only if theta_true provided
        "mcmc": {...},     # only if mcmc provided
      }

    Predictive fit source:
      - If p_buy_hat_imt is provided: use it directly.
      - Else: compute a myopic approximation from (theta_hat, u_m, price_vals, p_state_mt).
    """
    out: dict[str, Any] = {}

    # Fitted predictive metrics
    if p_buy_hat_imt is None:
        if theta_hat is None or u_m is None or price_vals is None or p_state_mt is None:
            raise ValueError(
                "Need either p_buy_hat_imt or (theta_hat, u_m, price_vals, p_state_mt) "
                "to compute fitted predictive metrics."
            )
        p_buy_hat_imt = myopic_buy_probabilities(
            a_imt=a_imt,
            p_state_mt=p_state_mt,
            u_m=u_m,
            price_vals=price_vals,
            theta=theta_hat,
            assume_stockout=assume_stockout,
        )

    out["fit"] = predictive_metrics_from_probs(
        a_imt=a_imt,
        p_buy_imt=p_buy_hat_imt,
        p_state_mt=p_state_mt,
        eps=eps,
    )

    # Parameter recovery
    if theta_true is not None and theta_hat is not None:
        out["param"] = parameter_metrics(theta_true, theta_hat)

    # Oracle predictive metrics (optional)
    oracle_available = False
    if p_buy_oracle_imt is not None:
        oracle_available = True
        out["oracle"] = predictive_metrics_from_probs(
            a_imt=a_imt,
            p_buy_imt=p_buy_oracle_imt,
            p_state_mt=p_state_mt,
            eps=eps,
        )
    elif (
        theta_true is not None
        and u_m is not None
        and price_vals is not None
        and p_state_mt is not None
    ):
        # If theta_true provided, allow oracle predictive metrics via the same myopic fallback.
        theta_oracle = dict(theta_true)
        if (
            theta_hat is not None
            and "u_scale" in theta_hat
            and "u_scale" not in theta_oracle
        ):
            theta_oracle["u_scale"] = np.ones_like(
                np.asarray(theta_hat["u_scale"]), dtype=np.float64
            )
        # Require the blocks used by the myopic predictor.
        needed = {"alpha", "fc", "v"}
        if needed.issubset(theta_oracle.keys()):
            oracle_available = True
            p_buy_oracle = myopic_buy_probabilities(
                a_imt=a_imt,
                p_state_mt=p_state_mt,
                u_m=u_m,
                price_vals=price_vals,
                theta=theta_oracle,
                assume_stockout=assume_stockout,
            )
            out["oracle"] = predictive_metrics_from_probs(
                a_imt=a_imt,
                p_buy_imt=p_buy_oracle,
                p_state_mt=p_state_mt,
                eps=eps,
            )

    if mcmc is not None:
        out["mcmc"] = mcmc

    return out


# =============================================================================
# Formatting helper (optional)
# =============================================================================


def format_evaluation_summary(
    eval_out: dict[str, Any],
    param_order: Optional[list[str]] = None,
) -> str:
    fit = eval_out["fit"]
    oracle = eval_out.get("oracle")
    params = eval_out.get("param")
    mcmc = eval_out.get("mcmc")

    shp = fit.get("shape", {})
    M = shp.get("M")
    N = shp.get("N")
    T = shp.get("T")
    n_obs = shp.get("n_obs")

    base = fit["baseline"]

    def f6(x: float) -> str:
        return f"{x:>10.6f}"

    def f4(x: float) -> str:
        return f"{x:>8.4f}"

    lines: list[str] = []

    # Header
    if M is not None and N is not None and T is not None:
        lines.append(f"data: M={M} N={N} T={T} | n_obs={n_obs}")
    lines.append("")

    # Main metrics table
    header = (
        f"{'model':<10}"
        f"{'nll':>10} "
        f"{'rmse':>10} "
        f"{'buy_emp':>8} "
        f"{'buy_pred':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(tag: str, d: dict[str, Any]) -> str:
        return (
            f"{tag:<10}"
            f"{f6(d['nll_per_obs'])} "
            f"{f6(d['rmse_prob'])} "
            f"{f4(d['buy_rate_emp'])} "
            f"{f4(d['buy_rate_pred'])}"
        )

    baseline_row = (
        f"{'baseline':<10}"
        f"{f6(base['nll_per_obs'])} "
        f"{f6(base['rmse_prob'])} "
        f"{f4(base['p0'])} "
        f"{'':>8}"
    )
    lines.append(baseline_row)
    lines.append(row("fitted", fit))
    if oracle is not None:
        lines.append(row("oracle", oracle))

    # Deltas
    lines.append("")
    nll_gain = base["nll_per_obs"] - fit["nll_per_obs"]
    rmse_gain = base["rmse_prob"] - fit["rmse_prob"]
    lines.append(f"gain vs baseline: Δnll={nll_gain:.6f} | Δrmse={rmse_gain:.6f}")
    if oracle is not None:
        lines.append(
            f"fitted - oracle: Δnll={(fit['nll_per_obs'] - oracle['nll_per_obs']):.6f} | "
            f"Δrmse={(fit['rmse_prob'] - oracle['rmse_prob']):.6f}"
        )

    # By-state table
    emp_s = fit.get("buy_rate_by_state_emp", {})
    pred_s = fit.get("buy_rate_by_state_pred", {})
    if emp_s:
        lines.append("")
        lines.append("buy rate by price state")
        st_header = f"{'state':<8}{'emp':>10} {'pred':>10} {'diff':>10}"
        lines.append(st_header)
        lines.append("-" * len(st_header))
        for s in sorted(emp_s.keys()):
            emp = float(emp_s[s])
            pred = float(pred_s.get(s, float("nan")))
            diff = pred - emp
            lines.append(f"{str(s):<8}{f6(emp)} {f6(pred)} {f6(diff)}")
        lines.append(f"rmse across states: {fit['rmse_buy_rate_by_state']:.6f}")

    # Parameter recovery table
    if params is not None and isinstance(params, dict) and params:
        if param_order is None:
            param_order = ["beta", "alpha", "v", "fc", "lambda_c", "u_scale"]

        present = [k for k in param_order if k in params]
        worst = sorted(present, key=lambda k: params[k]["rmse"], reverse=True)

        lines.append("")
        lines.append("parameter recovery (sorted by rmse)")
        p_header = (
            f"{'param':<10}"
            f"{'rmse':>10} "
            f"{'mean_true':>10} "
            f"{'mean_hat':>10} "
            f"{'bias':>10}"
        )
        lines.append(p_header)
        lines.append("-" * len(p_header))
        for k in worst:
            pk = params[k]
            bias = pk["mean_hat"] - pk["mean_true"]
            lines.append(
                f"{k:<10}"
                f"{f6(pk['rmse'])} "
                f"{f6(pk['mean_true'])} "
                f"{f6(pk['mean_hat'])} "
                f"{f6(bias)}"
            )

    # MCMC acceptance (elementwise)
    if isinstance(mcmc, dict):
        accept = mcmc.get("accept", {})
        rates = accept.get("rates", {})
        counts = accept.get("counts", {})
        n_saved = _as_int(mcmc.get("n_saved", None))

        if isinstance(rates, dict) and rates:
            lines.append("")
            lines.append("mcmc acceptance (elementwise)")
            if n_saved is not None:
                lines.append(f"n_saved: {n_saved}")

            a_header = f"{'block':<10}{'rate':>10}{'accepted':>12}{'proposed':>12}"
            lines.append(a_header)
            lines.append("-" * len(a_header))

            order = ["beta", "alpha", "v", "fc", "lambda_c", "u_scale"]

            total_acc = 0
            total_prop = 0

            for k in order:
                if k not in rates:
                    continue

                r = _as_float(rates.get(k))
                c = _as_int(counts.get(k))

                proposed = None
                if n_saved is not None and M is not None and N is not None:
                    if k == "u_scale":
                        proposed = n_saved * int(M)
                    else:
                        proposed = n_saved * int(M) * int(N)

                if c is not None:
                    total_acc += c
                if proposed is not None:
                    total_prop += proposed

                r_str = f"{r:>10.4f}" if r is not None else f"{'':>10}"
                c_str = f"{c:>12d}" if c is not None else f"{'':>12}"
                p_str = f"{proposed:>12d}" if proposed is not None else f"{'':>12}"
                lines.append(f"{k:<10}{r_str}{c_str}{p_str}")

            if total_prop > 0:
                overall = total_acc / max(1, total_prop)
                lines.append(f"overall: {overall:.4f} ({total_acc}/{total_prop})")

    return "\n".join(lines)
