"""
ching/stockpiling_input_validation.py

Single validation + normalization module for Phase-3 stockpiling (DGP + estimator).

Design goals:
  - All "expected vs got" input validation lives here.
  - Other modules call a single normalize_* function at their boundaries and then compute.
  - Dict-like inputs (sigmas, k, init_theta) are validated with strict schemas (missing + extra keys).

Public API:
  DGP:
    - validate_stockpiling_dgp_inputs(...)
    - normalize_stockpiling_dgp_inputs(...)

  Estimator init (__init__):
    - validate_stockpiling_estimator_init_inputs(...)
    - normalize_stockpiling_estimator_init_inputs(...)

  Estimator fit:
    - validate_stockpiling_estimator_fit_inputs(n_iter, k)
    - normalize_stockpiling_estimator_fit_inputs(n_iter, k)

  Fit-time initial state:
    - validate_stockpiling_estimator_init_theta_inputs(init_theta, M, J)
    - normalize_stockpiling_estimator_init_theta(init_theta, M, J)
"""

from __future__ import annotations

from typing import Any

import numpy as np


# =============================================================================
# Small helpers (private)
# =============================================================================


def _as_np(x: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: could not convert to numpy array ({e})") from e
    return a


def _require_ndim(a: np.ndarray, ndim: int, name: str) -> None:
    if a.ndim != ndim:
        raise ValueError(
            f"{name}: expected ndim={ndim}, got ndim={a.ndim} with shape={a.shape}"
        )


def _require_shape(a: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if a.shape != shape:
        raise ValueError(f"{name}: expected shape={shape}, got shape={a.shape}")


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: expected all finite, got non-finite entries")


def _require_int_scalar(x: Any, name: str, min_value: int | None = None) -> int:
    try:
        v = int(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: expected int scalar, got {x} ({e})") from e
    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _require_float_scalar(x: Any, name: str, min_value: float | None = None) -> float:
    try:
        v = float(x)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name}: expected float scalar, got {x} ({e})") from e
    if not np.isfinite(v):
        raise ValueError(f"{name}: expected finite float, got {v}")
    if min_value is not None and v < min_value:
        raise ValueError(f"{name}: expected >= {min_value}, got {v}")
    return v


def _require_dict_schema(
    d: Any, name: str, required_keys: set[str], allow_extra: bool
) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"{name}: expected dict, got type={type(d)}")

    missing = required_keys - set(d.keys())
    if missing:
        raise ValueError(f"{name}: missing keys {sorted(missing)}")

    extra = set(d.keys()) - required_keys
    if extra and not allow_extra:
        raise ValueError(f"{name}: unexpected keys {sorted(extra)}")

    return d


def _require_integer_valued_array(a: np.ndarray, name: str) -> None:
    if np.issubdtype(a.dtype, np.integer):
        return

    a64 = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a64).all():
        raise ValueError(f"{name}: expected all finite, got non-finite entries")

    if not np.equal(a64, np.floor(a64)).all():
        bad = a64[np.not_equal(a64, np.floor(a64))]
        ex = float(bad.flat[0]) if bad.size else float("nan")
        raise ValueError(f"{name}: expected integer-valued array, got example {ex}")


def _require_values_in_set(a: np.ndarray, name: str, allowed: set[int]) -> None:
    if a.size == 0:
        return
    vals = np.unique(a)
    bad = [int(v) for v in vals if int(v) not in allowed]
    if bad:
        raise ValueError(
            f"{name}: expected values in {sorted(allowed)}, got {bad[:10]}"
        )


def _require_in_int_range(a: np.ndarray, name: str, lo: int, hi_exclusive: int) -> None:
    if a.size == 0:
        return
    mn = int(a.min())
    mx = int(a.max())
    if mn < lo or mx >= hi_exclusive:
        raise ValueError(
            f"{name}: expected integer states in [{lo},{hi_exclusive-1}], got min={mn}, max={mx}"
        )


def _validate_probability_vector(pi: np.ndarray, name: str, atol: float) -> None:
    _require_finite(pi, name)
    if pi.size:
        if (pi < 0.0).any():
            mn = float(pi.min())
            raise ValueError(f"{name}: expected nonnegative entries, got min={mn}")
        s = float(pi.sum())
        if not np.isfinite(s) or abs(s - 1.0) > atol:
            raise ValueError(f"{name}: expected to sum to 1, got sum={s}")


def _validate_transition_matrix(P: np.ndarray, name: str, atol: float) -> None:
    _require_finite(P, name)
    if (P < 0.0).any():
        mn = float(P.min())
        raise ValueError(f"{name}: expected nonnegative entries, got min={mn}")
    row_sums = P.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=atol, rtol=0.0):
        mx_err = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"{name}: rows must sum to 1; max |sum-1|={mx_err}")


def _panel_dims_from_a(a_mnjt: Any) -> tuple[np.ndarray, int, int, int, int]:
    a = _as_np(a_mnjt, "a_mnjt")
    _require_ndim(a, 4, "a_mnjt")
    M, N, J, T = a.shape
    return a, int(M), int(N), int(J), int(T)


def _validate_price_inputs(
    P_price_mj: Any,
    price_vals_mj: Any,
    M: int,
    J: int,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    P = _as_np(P_price_mj, "P_price_mj", dtype=np.float64)
    _require_ndim(P, 4, "P_price_mj")
    if P.shape[0] != M or P.shape[1] != J:
        raise ValueError(
            f"P_price_mj: expected first dims (M,J)=({M},{J}), got {P.shape[:2]}"
        )

    S = int(P.shape[2])
    if S < 2:
        raise ValueError(f"P_price_mj: expected S>=2, got S={S}")
    if P.shape != (M, J, S, S):
        raise ValueError(
            f"P_price_mj: expected shape (M,J,S,S)=({M},{J},{S},{S}), got {P.shape}"
        )
    _validate_transition_matrix(P, "P_price_mj", atol=atol)

    pv = _as_np(price_vals_mj, "price_vals_mj", dtype=np.float64)
    _require_ndim(pv, 3, "price_vals_mj")
    if pv.shape != (M, J, S):
        raise ValueError(
            f"price_vals_mj: expected shape (M,J,S)=({M},{J},{S}), got {pv.shape}"
        )
    _require_finite(pv, "price_vals_mj")
    if pv.size and (pv <= 0.0).any():
        mn = float(pv.min())
        raise ValueError(
            f"price_vals_mj: expected strictly positive entries, got min={mn}"
        )

    return P, pv, S


def _coerce_scalar_or_1d(
    x: Any,
    name: str,
    length: int,
    dtype: Any,
    require_positive: bool,
) -> np.ndarray:
    length = _require_int_scalar(length, f"{name} length", min_value=1)

    a = _as_np(x, name, dtype=dtype)
    if a.ndim == 0:
        a = np.full((length,), float(a), dtype=dtype)
    else:
        _require_ndim(a, 1, name)

    _require_shape(a, (length,), name)
    _require_finite(a, name)

    if require_positive and a.size and (a <= 0.0).any():
        mn = float(a.min())
        raise ValueError(f"{name}: expected all > 0, got min={mn}")

    return a


def validate_stockpiling_phase3_config(cfg: Any, M: int, J: int) -> None:
    """
    Validate the Phase-3 configuration dict used by run_ching.py before constructing
    price processes / running the DGP / running MCMC.

    This is a *config preflight* check:
      - validates cfg schema (missing + extra keys)
      - validates scalar ranges for DGP/DP/price-process construction/MCMC
      - validates init_theta/k/sigmas schemas and ranges using existing validators

    Args:
      cfg: Phase-3 config dict (see CFG_PHASE3 in run_ching.py).
      M: Number of markets (derived from Phase-2 outputs).
      J: Number of products (derived from Phase-1 outputs).
    """
    M = _require_int_scalar(M, "M", min_value=1)
    J = _require_int_scalar(J, "J", min_value=1)

    required_keys = {
        # DGP / DP controls
        "N",
        "T",
        "I_max",
        "S",
        "waste_cost",
        "dp_tol",
        "dp_max_iter",
        # Price process construction
        "price_seed",
        "p_stay",
        "P_noise_sd",
        "P_min_prob",
        "price_base_low",
        "price_base_high",
        "discount_low",
        "discount_high",
        "price_noise_sd",
        # MCMC controls
        "mcmc_seed",
        "mcmc_n_iter",
        "init_theta",
        "sigmas",
        "k",
    }
    d = _require_dict_schema(cfg, "cfg_phase3", required_keys, allow_extra=False)

    # DGP / DP
    _require_int_scalar(d["N"], "cfg_phase3['N']", min_value=1)
    _require_int_scalar(d["T"], "cfg_phase3['T']", min_value=1)
    I_max = _require_int_scalar(d["I_max"], "cfg_phase3['I_max']", min_value=0)
    S = _require_int_scalar(d["S"], "cfg_phase3['S']", min_value=2)
    _ = S  # S used later for range checks, but keep validation explicit here

    _require_float_scalar(d["waste_cost"], "cfg_phase3['waste_cost']", min_value=0.0)
    _require_float_scalar(d["dp_tol"], "cfg_phase3['dp_tol']", min_value=0.0)
    _require_int_scalar(d["dp_max_iter"], "cfg_phase3['dp_max_iter']", min_value=1)

    # Price process construction (for build_price_processes)
    _require_int_scalar(d["price_seed"], "cfg_phase3['price_seed']", min_value=0)

    p_stay = _require_float_scalar(d["p_stay"], "cfg_phase3['p_stay']")
    if p_stay <= 0.0 or p_stay >= 1.0:
        raise ValueError(f"cfg_phase3['p_stay']: expected in (0,1), got {p_stay}")

    _require_float_scalar(d["P_noise_sd"], "cfg_phase3['P_noise_sd']", min_value=0.0)

    P_min_prob = _require_float_scalar(d["P_min_prob"], "cfg_phase3['P_min_prob']")
    if P_min_prob <= 0.0 or P_min_prob >= 1.0:
        raise ValueError(
            f"cfg_phase3['P_min_prob']: expected in (0,1), got {P_min_prob}"
        )

    base_low = _require_float_scalar(
        d["price_base_low"], "cfg_phase3['price_base_low']"
    )
    base_high = _require_float_scalar(
        d["price_base_high"], "cfg_phase3['price_base_high']"
    )
    if base_low <= 0.0:
        raise ValueError(f"cfg_phase3['price_base_low']: expected > 0, got {base_low}")
    if base_high <= base_low:
        raise ValueError(
            "cfg_phase3['price_base_high']: expected > cfg_phase3['price_base_low'], "
            f"got {base_high} <= {base_low}"
        )

    disc_low = _require_float_scalar(
        d["discount_low"], "cfg_phase3['discount_low']", min_value=0.0
    )
    disc_high = _require_float_scalar(d["discount_high"], "cfg_phase3['discount_high']")
    if disc_high <= disc_low:
        raise ValueError(
            "cfg_phase3['discount_high']: expected > cfg_phase3['discount_low'], "
            f"got {disc_high} <= {disc_low}"
        )
    if disc_high >= 1.0:
        raise ValueError(f"cfg_phase3['discount_high']: expected < 1, got {disc_high}")

    _require_float_scalar(
        d["price_noise_sd"], "cfg_phase3['price_noise_sd']", min_value=0.0
    )

    # MCMC controls
    _require_int_scalar(d["mcmc_seed"], "cfg_phase3['mcmc_seed']", min_value=0)
    n_iter = _require_int_scalar(
        d["mcmc_n_iter"], "cfg_phase3['mcmc_n_iter']", min_value=1
    )

    # init_theta schema/ranges (allows scalar or vector forms per existing validator)
    validate_stockpiling_estimator_init_theta_inputs(
        init_theta=d["init_theta"],
        M=M,
        J=J,
    )

    # k schema/ranges (reuses existing fit validator; also validates n_iter >= 1)
    validate_stockpiling_estimator_fit_inputs(
        n_iter=n_iter,
        k=d["k"],
    )

    # sigmas schema/ranges (same requirements as estimator __init__)
    required_sigma_keys = {"z_beta", "z_alpha", "z_v", "z_fc", "z_u_scale"}
    sig = _require_dict_schema(
        d["sigmas"], "cfg_phase3['sigmas']", required_sigma_keys, allow_extra=False
    )
    for key in sorted(required_sigma_keys):
        v = _require_float_scalar(
            sig[key], f"cfg_phase3['sigmas']['{key}']", min_value=0.0
        )
        if v <= 0.0:
            raise ValueError(f"cfg_phase3['sigmas']['{key}']: expected > 0, got {v}")

    # Optional: sanity check that I_max is consistent with init inventory distribution usage in run_ching.
    _ = I_max


# =============================================================================
# Public validators / normalizers: DGP
# =============================================================================


def validate_stockpiling_dgp_inputs(
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    N: int,
    T: int,
    I_max: int,
    waste_cost: float,
    seed: int,
    tol: float | None = None,
    max_iter: int | None = None,
) -> None:
    """
    Validate inputs to datasets/ching_dgp.generate_dgp(...) for Phase-3 stockpiling.

    Infers:
      - M from E_bar_true
      - J from delta_true
      - S from P_price_mj / price_vals_mj
    """
    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    _require_ndim(delta, 1, "delta_true")
    _require_finite(delta, "delta_true")
    J = int(delta.shape[0])
    if J < 1:
        raise ValueError(f"delta_true: expected length J>=1, got J={J}")

    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    _require_ndim(E_bar, 1, "E_bar_true")
    _require_finite(E_bar, "E_bar_true")
    M = int(E_bar.shape[0])
    if M < 1:
        raise ValueError(f"E_bar_true: expected length M>=1, got M={M}")

    njt = _as_np(njt_true, "njt_true", dtype=np.float64)
    _require_ndim(njt, 2, "njt_true")
    _require_shape(njt, (M, J), "njt_true")
    _require_finite(njt, "njt_true")

    _require_int_scalar(N, "N", min_value=1)
    _require_int_scalar(T, "T", min_value=1)
    _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_int_scalar(seed, "seed", min_value=0)

    _validate_price_inputs(P_price_mj, price_vals_mj, M=M, J=J)

    if tol is not None:
        _require_float_scalar(tol, "tol", min_value=0.0)
    if max_iter is not None:
        _require_int_scalar(max_iter, "max_iter", min_value=1)


def normalize_stockpiling_dgp_inputs(
    delta_true: Any,
    E_bar_true: Any,
    njt_true: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    N: int,
    T: int,
    I_max: int,
    waste_cost: float,
    seed: int,
    tol: float | None = None,
    max_iter: int | None = None,
) -> dict[str, Any]:
    """
    Validate and normalize DGP inputs into canonical numpy dtypes/shapes.

    Returns a dict with arrays converted to float64 and scalars to Python int/float,
    plus inferred dimensions M,J,S.
    """
    validate_stockpiling_dgp_inputs(
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        N=N,
        T=T,
        I_max=I_max,
        waste_cost=waste_cost,
        seed=seed,
        tol=tol,
        max_iter=max_iter,
    )

    delta = _as_np(delta_true, "delta_true", dtype=np.float64)
    E_bar = _as_np(E_bar_true, "E_bar_true", dtype=np.float64)
    njt = _as_np(njt_true, "njt_true", dtype=np.float64)

    M = int(E_bar.shape[0])
    J = int(delta.shape[0])
    P, pv, S = _validate_price_inputs(P_price_mj, price_vals_mj, M=M, J=J)

    out: dict[str, Any] = {
        "delta_true": delta,
        "E_bar_true": E_bar,
        "njt_true": njt,
        "P_price_mj": P,
        "price_vals_mj": pv,
        "N": int(N),
        "T": int(T),
        "I_max": int(I_max),
        "waste_cost": float(waste_cost),
        "seed": int(seed),
        "M": M,
        "J": J,
        "S": S,
    }
    if tol is not None:
        out["tol"] = float(tol)
    if max_iter is not None:
        out["max_iter"] = int(max_iter)
    return out


# =============================================================================
# Public validators / normalizers: Estimator init (__init__)
# =============================================================================


def validate_stockpiling_estimator_init_inputs(
    a_mnjt: Any,
    p_state_mjt: Any,
    u_mj: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    lambda_mn: Any,
    I_max: int,
    pi_I0: Any,
    waste_cost: float,
    tol: float,
    max_iter: int,
    sigmas: dict[str, Any],
    rng_seed: int,
) -> None:
    """
    Validate inputs to StockpilingEstimator.__init__.

    Validates internal consistency of shapes:
      a_mnjt        (M,N,J,T)    indicator in {0,1}
      p_state_mjt   (M,J,T)      integer in {0,...,S-1}
      u_mj          (M,J)        finite float
      lambda_mn     (M,N)        finite float in (0,1)
      price_vals_mj (M,J,S)      finite float > 0
      P_price_mj    (M,J,S,S)    finite row-stochastic
      pi_I0         (I_max+1,)   probability vector

    Also validates:
      tol >= 0, max_iter >= 1, rng_seed >= 0
      sigmas schema and strictly positive values
    """
    I_max = _require_int_scalar(I_max, "I_max", min_value=0)
    _require_float_scalar(waste_cost, "waste_cost", min_value=0.0)
    _require_float_scalar(tol, "tol", min_value=0.0)
    _require_int_scalar(max_iter, "max_iter", min_value=1)
    _require_int_scalar(rng_seed, "rng_seed", min_value=0)

    a, M, N, J, T = _panel_dims_from_a(a_mnjt)
    _require_integer_valued_array(a, "a_mnjt")
    _require_values_in_set(np.asarray(a, dtype=np.int64), "a_mnjt", {0, 1})

    p_state = _as_np(p_state_mjt, "p_state_mjt")
    _require_ndim(p_state, 3, "p_state_mjt")
    _require_shape(p_state, (M, J, T), "p_state_mjt")
    _require_integer_valued_array(p_state, "p_state_mjt")

    u = _as_np(u_mj, "u_mj", dtype=np.float64)
    _require_ndim(u, 2, "u_mj")
    _require_shape(u, (M, J), "u_mj")
    _require_finite(u, "u_mj")

    lam = _as_np(lambda_mn, "lambda_mn", dtype=np.float64)
    _require_ndim(lam, 2, "lambda_mn")
    _require_shape(lam, (M, N), "lambda_mn")
    _require_finite(lam, "lambda_mn")
    if lam.size and ((lam <= 0.0).any() or (lam >= 1.0).any()):
        mn = float(lam.min())
        mx = float(lam.max())
        raise ValueError(
            f"lambda_mn: expected entries in (0,1), got min={mn}, max={mx}"
        )

    P, pv, S = _validate_price_inputs(P_price_mj, price_vals_mj, M=M, J=J)
    _ = (P, pv)

    p_state_i64 = np.asarray(p_state, dtype=np.int64)
    _require_in_int_range(p_state_i64, "p_state_mjt", lo=0, hi_exclusive=S)

    pi = _as_np(pi_I0, "pi_I0", dtype=np.float64)
    _require_ndim(pi, 1, "pi_I0")
    _require_shape(pi, (I_max + 1,), "pi_I0")
    _validate_probability_vector(pi, "pi_I0", atol=1e-6)

    required_sigma_keys = {"z_beta", "z_alpha", "z_v", "z_fc", "z_u_scale"}
    sig = _require_dict_schema(sigmas, "sigmas", required_sigma_keys, allow_extra=False)
    for key in sorted(required_sigma_keys):
        v = _require_float_scalar(sig[key], f"sigmas['{key}']", min_value=0.0)
        if v <= 0.0:
            raise ValueError(f"sigmas['{key}']: expected > 0, got {v}")


def normalize_stockpiling_estimator_init_inputs(
    a_mnjt: Any,
    p_state_mjt: Any,
    u_mj: Any,
    price_vals_mj: Any,
    P_price_mj: Any,
    lambda_mn: Any,
    I_max: int,
    pi_I0: Any,
    waste_cost: float,
    tol: float,
    max_iter: int,
    sigmas: dict[str, Any],
    rng_seed: int,
) -> dict[str, Any]:
    """
    Validate and normalize inputs to StockpilingEstimator.__init__.

    Returns canonical numpy arrays and scalars plus inferred dimensions.
    """
    validate_stockpiling_estimator_init_inputs(
        a_mnjt=a_mnjt,
        p_state_mjt=p_state_mjt,
        u_mj=u_mj,
        price_vals_mj=price_vals_mj,
        P_price_mj=P_price_mj,
        lambda_mn=lambda_mn,
        I_max=I_max,
        pi_I0=pi_I0,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
        sigmas=sigmas,
        rng_seed=rng_seed,
    )

    a = np.asarray(_as_np(a_mnjt, "a_mnjt"), dtype=np.int32)
    p_state = np.asarray(_as_np(p_state_mjt, "p_state_mjt"), dtype=np.int32)
    u = _as_np(u_mj, "u_mj", dtype=np.float64)
    lam = _as_np(lambda_mn, "lambda_mn", dtype=np.float64)

    M, N, J, T = a.shape
    P, pv, S = _validate_price_inputs(P_price_mj, price_vals_mj, M=int(M), J=int(J))
    pi = _as_np(pi_I0, "pi_I0", dtype=np.float64)

    required_sigma_keys = {"z_beta", "z_alpha", "z_v", "z_fc", "z_u_scale"}
    sig = _require_dict_schema(sigmas, "sigmas", required_sigma_keys, allow_extra=False)
    sig_out = {key: float(sig[key]) for key in sorted(required_sigma_keys)}

    return {
        "a_mnjt": a,
        "p_state_mjt": p_state,
        "u_mj": u,
        "lambda_mn": lam,
        "P_price_mj": P,
        "price_vals_mj": pv,
        "pi_I0": pi,
        "I_max": int(I_max),
        "waste_cost": float(waste_cost),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "sigmas": sig_out,
        "rng_seed": int(rng_seed),
        "M": int(M),
        "N": int(N),
        "J": int(J),
        "T": int(T),
        "S": int(S),
    }


# =============================================================================
# Public validators / normalizers: Estimator fit
# =============================================================================


def validate_stockpiling_estimator_fit_inputs(
    n_iter: int,
    k: dict[str, Any],
) -> None:
    """
    Validate inputs to StockpilingEstimator.fit(n_iter, k).

    Args:
      n_iter: positive int
      k: dict with required keys {"beta","alpha","v","fc","u_scale"}.
         Step sizes must be finite floats.
         beta/alpha/v/fc must be > 0; u_scale may be 0 to freeze its update step.
    """
    _ = _require_int_scalar(n_iter, "n_iter", min_value=1)

    required = {"beta", "alpha", "v", "fc", "u_scale"}
    k_dict = _require_dict_schema(k, "k", required, allow_extra=False)

    for key in sorted(required):
        v = _require_float_scalar(k_dict[key], f"k['{key}']", min_value=0.0)
        if key == "u_scale":
            if v < 0.0:
                raise ValueError(f"k['{key}']: expected >= 0, got {v}")
        else:
            if v <= 0.0:
                raise ValueError(f"k['{key}']: expected > 0, got {v}")


def normalize_stockpiling_estimator_fit_inputs(
    n_iter: int,
    k: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize fit inputs into canonical Python types."""
    validate_stockpiling_estimator_fit_inputs(n_iter=n_iter, k=k)
    required = {"beta", "alpha", "v", "fc", "u_scale"}
    k_out = {key: float(k[key]) for key in sorted(required)}
    return {"n_iter": int(n_iter), "k": k_out}


# =============================================================================
# Public validators / normalizers: Fit-time initial state (init_theta)
# =============================================================================


def validate_stockpiling_estimator_init_theta_inputs(
    init_theta: Any,
    M: int,
    J: int,
) -> None:
    """
    Validate init_theta passed to StockpilingEstimator.fit.

    Required keys: {"beta","alpha","v","fc","u_scale"} and no extras.

    Shape conventions:
      - beta: scalar in (0,1)
      - alpha, v, fc: scalar or shape (J,)
      - u_scale: scalar or shape (M,)
    """
    M = _require_int_scalar(M, "M", min_value=1)
    J = _require_int_scalar(J, "J", min_value=1)

    required = {"beta", "alpha", "v", "fc", "u_scale"}
    d = _require_dict_schema(init_theta, "init_theta", required, allow_extra=False)

    beta = _require_float_scalar(d["beta"], "init_theta['beta']")
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"init_theta['beta']: expected in (0,1), got {beta}")

    _coerce_scalar_or_1d(
        d["alpha"],
        "init_theta['alpha']",
        length=J,
        dtype=np.float64,
        require_positive=True,
    )
    _coerce_scalar_or_1d(
        d["v"],
        "init_theta['v']",
        length=J,
        dtype=np.float64,
        require_positive=True,
    )
    _coerce_scalar_or_1d(
        d["fc"],
        "init_theta['fc']",
        length=J,
        dtype=np.float64,
        require_positive=True,
    )
    _coerce_scalar_or_1d(
        d["u_scale"],
        "init_theta['u_scale']",
        length=M,
        dtype=np.float64,
        require_positive=True,
    )


def normalize_stockpiling_estimator_init_theta(
    init_theta: Any,
    M: int,
    J: int,
) -> dict[str, Any]:
    """
    Validate and normalize init_theta into canonical shapes/dtypes.

    Returns:
      - beta: float
      - alpha, v, fc: np.ndarray float64 shape (J,)
      - u_scale: np.ndarray float64 shape (M,)
    """
    validate_stockpiling_estimator_init_theta_inputs(init_theta=init_theta, M=M, J=J)

    beta = float(init_theta["beta"])
    alpha = _coerce_scalar_or_1d(
        init_theta["alpha"],
        "init_theta['alpha']",
        length=int(J),
        dtype=np.float64,
        require_positive=True,
    )
    v = _coerce_scalar_or_1d(
        init_theta["v"],
        "init_theta['v']",
        length=int(J),
        dtype=np.float64,
        require_positive=True,
    )
    fc = _coerce_scalar_or_1d(
        init_theta["fc"],
        "init_theta['fc']",
        length=int(J),
        dtype=np.float64,
        require_positive=True,
    )
    u_scale = _coerce_scalar_or_1d(
        init_theta["u_scale"],
        "init_theta['u_scale']",
        length=int(M),
        dtype=np.float64,
        require_positive=True,
    )

    return {"beta": beta, "alpha": alpha, "v": v, "fc": fc, "u_scale": u_scale}
