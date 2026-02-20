"""
Input validation for the simulation-oriented BLP estimator.

Policy:
  - Validate only external configuration inputs.
  - No defaults and no fallbacks: missing fields are rejected.
  - No type coercion: fields must have the correct Python scalar types.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


REQUIRED_KEYS = (
    "n_draws",
    "seed",
    "damping",
    "tol",
    "share_tol",
    "max_iter",
    "fail_penalty",
    "sigma_lower",
    "sigma_upper",
    "sigma_grid_points",
    "nelder_mead_maxiter",
    "nelder_mead_xatol",
    "nelder_mead_fatol",
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _require_int(x: Any, name: str) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError(f"{name} must be an int; got {type(x)}.")
    return x


def _require_float(x: Any, name: str) -> float:
    if not isinstance(x, float) or isinstance(x, bool):
        raise TypeError(f"{name} must be a float; got {type(x)}.")
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite; got {x}.")
    return x


def validate_blp_config(config: Mapping[str, Any]) -> dict:
    """Validate an external config mapping for the BLP estimator and return a plain dict."""
    if not isinstance(config, Mapping):
        raise TypeError("BLP config must be a mapping (e.g., dict).")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing BLP config fields: {missing}")

    out = {
        "n_draws": _require_int(config["n_draws"], "n_draws"),
        "seed": _require_int(config["seed"], "seed"),
        "damping": _require_float(config["damping"], "damping"),
        "tol": _require_float(config["tol"], "tol"),
        "share_tol": _require_float(config["share_tol"], "share_tol"),
        "max_iter": _require_int(config["max_iter"], "max_iter"),
        "fail_penalty": _require_float(config["fail_penalty"], "fail_penalty"),
        "sigma_lower": _require_float(config["sigma_lower"], "sigma_lower"),
        "sigma_upper": _require_float(config["sigma_upper"], "sigma_upper"),
        "sigma_grid_points": _require_int(
            config["sigma_grid_points"], "sigma_grid_points"
        ),
        "nelder_mead_maxiter": _require_int(
            config["nelder_mead_maxiter"], "nelder_mead_maxiter"
        ),
        "nelder_mead_xatol": _require_float(
            config["nelder_mead_xatol"], "nelder_mead_xatol"
        ),
        "nelder_mead_fatol": _require_float(
            config["nelder_mead_fatol"], "nelder_mead_fatol"
        ),
    }

    _require(out["n_draws"] >= 1, "n_draws must be >= 1.")
    _require(out["seed"] >= 0, "seed must be >= 0.")
    _require(out["max_iter"] >= 1, "max_iter must be >= 1.")

    _require(0.0 < out["damping"] <= 1.0, "damping must satisfy 0 < damping <= 1.")
    _require(out["tol"] > 0.0, "tol must be > 0.")
    _require(out["share_tol"] > 0.0, "share_tol must be > 0.")
    _require(out["fail_penalty"] > 0.0, "fail_penalty must be > 0.")

    _require(out["sigma_lower"] > 0.0, "sigma_lower must be > 0.")
    _require(
        out["sigma_upper"] > out["sigma_lower"], "sigma_upper must be > sigma_lower."
    )
    _require(out["sigma_grid_points"] >= 2, "sigma_grid_points must be >= 2.")

    _require(out["nelder_mead_maxiter"] >= 1, "nelder_mead_maxiter must be >= 1.")
    _require(out["nelder_mead_xatol"] > 0.0, "nelder_mead_xatol must be > 0.")
    _require(out["nelder_mead_fatol"] > 0.0, "nelder_mead_fatol must be > 0.")

    return out
