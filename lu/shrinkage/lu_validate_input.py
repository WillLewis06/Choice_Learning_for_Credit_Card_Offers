"""
Input validation for the Lu shrinkage estimator.

Policy:
  - Validate only external inputs:
      * observed data tensors passed into the estimator constructor
      * configuration values passed from the orchestration layer
  - Do not validate internal sampler state, tf.Variables, or tuning internals.
  - No type coercion and no defaults: missing or wrongly typed config values
    must be rejected here.
  - All tensors are expected to be tf.float64.
"""

from __future__ import annotations

import tensorflow as tf


# -----------------------------------------------------------------------------
# Helpers (Python-side)
# -----------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _require_type(x, t, msg: str) -> None:
    if not isinstance(x, t):
        raise TypeError(msg)


def _require_int(x, name: str) -> None:
    """Require a Python int (reject bool)."""
    _require_type(x, int, f"{name} must be an int; got {type(x)}.")
    _require(not isinstance(x, bool), f"{name} must be an int (not bool).")


def _require_positive_int(x, name: str) -> None:
    _require_int(x, name)
    _require(x > 0, f"{name} must be > 0; got {x}.")


def _require_nonnegative_int(x, name: str) -> None:
    _require_int(x, name)
    _require(x >= 0, f"{name} must be >= 0; got {x}.")


def _require_float(x, name: str) -> None:
    """Require a Python float (reject int/bool)."""
    _require_type(x, float, f"{name} must be a float; got {type(x)}.")
    _require(not isinstance(x, bool), f"{name} must be a float (not bool).")


def _require_finite_float(x, name: str) -> None:
    _require_float(x, name)
    _require(
        x == x and x not in (float("inf"), float("-inf")),
        f"{name} must be finite; got {x}.",
    )


def _require_positive_float(x, name: str) -> None:
    _require_finite_float(x, name)
    _require(x > 0.0, f"{name} must be > 0; got {x}.")


def _require_tensor_rank(x: tf.Tensor, rank: int, name: str) -> None:
    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got rank {x.shape.rank}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype == tf.float64, f"{name} must be tf.float64; got {x.dtype}.")


def _require_all_finite(x: tf.Tensor, name: str) -> None:
    ok = bool(tf.reduce_all(tf.math.is_finite(x)).numpy())
    _require(ok, f"{name} must be finite (no NaN/inf).")


def _require_all_nonnegative(x: tf.Tensor, name: str) -> None:
    ok = bool(tf.reduce_all(x >= tf.constant(0.0, tf.float64)).numpy())
    _require(ok, f"{name} must be non-negative.")


def _require_prob_band(low: float, high: float, low_name: str, high_name: str) -> None:
    _require_finite_float(low, low_name)
    _require_finite_float(high, high_name)
    _require(0.0 <= low <= 1.0, f"{low_name} must be in [0,1]; got {low}.")
    _require(0.0 <= high <= 1.0, f"{high_name} must be in [0,1]; got {high}.")
    _require(low <= high, f"Must have {low_name} <= {high_name}; got {low} > {high}.")


# -----------------------------------------------------------------------------
# External inputs: observed data + configs
# -----------------------------------------------------------------------------


def init_validate_input(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    n_draws: int,
    seed: int,
) -> None:
    """Validate estimator constructor inputs (observed data + core MC settings)."""
    # Observed tensors: dtype + rank.
    _require_float64_tensor(pjt, "pjt")
    _require_float64_tensor(wjt, "wjt")
    _require_float64_tensor(qjt, "qjt")
    _require_float64_tensor(q0t, "q0t")

    _require_tensor_rank(pjt, 2, "pjt")
    _require_tensor_rank(wjt, 2, "wjt")
    _require_tensor_rank(qjt, 2, "qjt")
    _require_tensor_rank(q0t, 1, "q0t")

    # Static shapes.
    _require(
        pjt.shape[0] is not None and pjt.shape[1] is not None,
        f"pjt must have static shape (T,J); got {pjt.shape}.",
    )
    _require(
        q0t.shape[0] is not None, f"q0t must have static shape (T,); got {q0t.shape}."
    )

    T = pjt.shape[0]
    J = pjt.shape[1]
    _require(
        wjt.shape == pjt.shape,
        f"wjt must have shape (T,J)={pjt.shape}; got {wjt.shape}.",
    )
    _require(
        qjt.shape == pjt.shape,
        f"qjt must have shape (T,J)={pjt.shape}; got {qjt.shape}.",
    )
    _require(q0t.shape[0] == T, f"q0t must have length T={T}; got {q0t.shape[0]}.")

    # Basic value checks on external observed data.
    _require_all_finite(pjt, "pjt")
    _require_all_finite(wjt, "wjt")
    _require_all_finite(qjt, "qjt")
    _require_all_finite(q0t, "q0t")
    _require_all_nonnegative(qjt, "qjt")
    _require_all_nonnegative(q0t, "q0t")

    # MC settings (external config).
    _require_positive_int(n_draws, "n_draws")
    _require_nonnegative_int(seed, "seed")


def fit_validate_input(
    n_iter: int,
    pilot_length: int,
    ridge: float,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor_rw: float,
    factor_tmh: float,
) -> None:
    """Validate fit/tuning controls passed from orchestration."""
    _require_positive_int(n_iter, "n_iter")
    _require_positive_int(pilot_length, "pilot_length")
    _require_positive_int(max_rounds, "max_rounds")

    _require_finite_float(ridge, "ridge")
    _require(ridge >= 0.0, f"ridge must be >= 0; got {ridge}.")

    _require_prob_band(target_low, target_high, "target_low", "target_high")

    _require_finite_float(factor_rw, "factor_rw")
    _require_finite_float(factor_tmh, "factor_tmh")
    _require(factor_rw > 1.0, f"factor_rw must be > 1; got {factor_rw}.")
    _require(factor_tmh > 1.0, f"factor_tmh must be > 1; got {factor_tmh}.")


def posterior_validate_input(posterior_config) -> None:
    """Validate the posterior/prior config (required, no defaults)."""
    # Required attributes (fail early with a clear message).
    required = [
        "n_draws",
        "seed",
        "dtype",
        "eps",
        "beta_p_mean",
        "beta_p_var",
        "beta_w_mean",
        "beta_w_var",
        "r_mean",
        "r_var",
        "E_bar_mean",
        "E_bar_var",
        "T0_sq",
        "T1_sq",
        "a_phi",
        "b_phi",
    ]
    missing = [name for name in required if not hasattr(posterior_config, name)]
    _require(not missing, "posterior_config missing fields: " + ", ".join(missing))

    # dtype must be fixed to float64 (project invariant).
    _require(
        posterior_config.dtype == tf.float64,
        f"posterior_config.dtype must be tf.float64; got {posterior_config.dtype}.",
    )

    # Integers.
    _require_positive_int(posterior_config.n_draws, "posterior_config.n_draws")
    _require_nonnegative_int(posterior_config.seed, "posterior_config.seed")

    # Numerical stability guard.
    _require_finite_float(posterior_config.eps, "posterior_config.eps")
    _require(
        0.0 < posterior_config.eps < 0.5,
        f"posterior_config.eps must satisfy 0 < eps < 0.5; got {posterior_config.eps}.",
    )

    # Normal prior variances.
    _require_positive_float(posterior_config.beta_p_var, "posterior_config.beta_p_var")
    _require_positive_float(posterior_config.beta_w_var, "posterior_config.beta_w_var")
    _require_positive_float(posterior_config.r_var, "posterior_config.r_var")
    _require_positive_float(posterior_config.E_bar_var, "posterior_config.E_bar_var")

    # Spike/slab variances.
    _require_positive_float(posterior_config.T0_sq, "posterior_config.T0_sq")
    _require_positive_float(posterior_config.T1_sq, "posterior_config.T1_sq")
    _require(
        posterior_config.T1_sq > posterior_config.T0_sq,
        f"posterior_config.T1_sq must be > posterior_config.T0_sq; got {posterior_config.T1_sq} <= {posterior_config.T0_sq}.",
    )

    # Beta prior hyperparameters.
    _require_positive_float(posterior_config.a_phi, "posterior_config.a_phi")
    _require_positive_float(posterior_config.b_phi, "posterior_config.b_phi")

    # Means (finite floats).
    _require_finite_float(posterior_config.beta_p_mean, "posterior_config.beta_p_mean")
    _require_finite_float(posterior_config.beta_w_mean, "posterior_config.beta_w_mean")
    _require_finite_float(posterior_config.r_mean, "posterior_config.r_mean")
    _require_finite_float(posterior_config.E_bar_mean, "posterior_config.E_bar_mean")
