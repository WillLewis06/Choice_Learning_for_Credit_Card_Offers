from __future__ import annotations
import tensorflow as tf

# -----------------------------------------------------------------------------
# Helpers (Python-side)
# -----------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _require_type(x, types, msg: str) -> None:
    if not isinstance(x, types):
        raise TypeError(msg)


def _is_float_like(x) -> bool:
    # strict: require Python float/int (no numpy scalars unless you choose to add them)
    return isinstance(x, (float, int)) and not isinstance(x, bool)


def _require_tensor_rank(x: tf.Tensor, rank: int, name: str) -> None:
    _require(
        x.shape.rank == rank,
        f"{name} must have rank {rank}; got rank {x.shape.rank}.",
    )


def _require_float64_tensor(x: tf.Tensor, name: str) -> None:
    _require(tf.is_tensor(x), f"{name} must be a tf.Tensor.")
    _require(x.dtype == tf.float64, f"{name} must be tf.float64; got {x.dtype}.")


def _require_float64_variable(x, name: str) -> None:
    _require_type(x, tf.Variable, f"{name} must be a tf.Variable (no backward compat).")
    _require(x.dtype == tf.float64, f"{name} must be tf.float64; got {x.dtype}.")


def _require_positive_int(x: int, name: str) -> None:
    _require_type(x, int, f"{name} must be an int.")
    _require(x > 0, f"{name} must be > 0; got {x}.")


def _require_float(x: float, name: str) -> None:
    _require(_is_float_like(x), f"{name} must be a float or int; got {type(x)}.")


def _require_prob_band(low: float, high: float, low_name: str, high_name: str) -> None:
    _require_float(low, low_name)
    _require_float(high, high_name)
    _require(0.0 <= float(low) <= 1.0, f"{low_name} must be in [0,1]; got {low}.")
    _require(0.0 <= float(high) <= 1.0, f"{high_name} must be in [0,1]; got {high}.")
    _require(
        float(low) <= float(high),
        f"Must have {low_name} <= {high_name}; got {low} > {high}.",
    )


# -----------------------------------------------------------------------------
# Estimator entrypoints (Python-side)
# -----------------------------------------------------------------------------


def init_validate_input(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    n_draws: int,
    seed: int,
) -> None:
    """
    Validation contract for LuShrinkageEstimator.__init__.

    Strict (no coercion):
      - pjt, wjt, qjt: tf.float64 tensors, rank-2, identical shape (T,J) with T,J known.
      - q0t: tf.float64 tensor, rank-1 shape (T,) with same T as pjt.
      - n_draws: int > 0
      - seed: int (any value allowed)
    """
    _require_float64_tensor(pjt, "pjt")
    _require_float64_tensor(wjt, "wjt")
    _require_float64_tensor(qjt, "qjt")
    _require_float64_tensor(q0t, "q0t")

    _require_tensor_rank(pjt, 2, "pjt")
    _require_tensor_rank(wjt, 2, "wjt")
    _require_tensor_rank(qjt, 2, "qjt")
    _require_tensor_rank(q0t, 1, "q0t")

    _require(
        wjt.shape == pjt.shape,
        f"wjt must have same shape as pjt; got {wjt.shape} vs {pjt.shape}.",
    )
    _require(
        qjt.shape == pjt.shape,
        f"qjt must have same shape as pjt; got {qjt.shape} vs {pjt.shape}.",
    )

    # Require static T/J for simplicity (no backward compatibility).
    _require(
        pjt.shape[0] is not None and pjt.shape[1] is not None,
        f"pjt must have static shape (T,J); got {pjt.shape}.",
    )
    T = int(pjt.shape[0])

    _require(
        q0t.shape[0] is not None,
        f"q0t must have static shape (T,); got {q0t.shape}.",
    )
    _require(int(q0t.shape[0]) == T, f"q0t must be shape (T,) with T={T}.")

    _require_positive_int(n_draws, "n_draws")
    _require_type(seed, int, "seed must be an int.")


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
    """
    Validation contract for LuShrinkageEstimator.fit.

    Strict (no coercion):
      - n_iter, pilot_length, max_rounds: int > 0
      - ridge: float >= 0
      - target band: 0 <= target_low <= target_high <= 1
      - factor_rw, factor_tmh: float > 1
    """
    _require_positive_int(n_iter, "n_iter")
    _require_positive_int(pilot_length, "pilot_length")
    _require_positive_int(max_rounds, "max_rounds")

    _require_float(ridge, "ridge")
    _require(float(ridge) >= 0.0, f"ridge must be >= 0; got {ridge}.")

    _require_prob_band(target_low, target_high, "target_low", "target_high")

    _require_float(factor_rw, "factor_rw")
    _require_float(factor_tmh, "factor_tmh")
    _require(float(factor_rw) > 1.0, f"factor_rw must be > 1; got {factor_rw}.")
    _require(float(factor_tmh) > 1.0, f"factor_tmh must be > 1; got {factor_tmh}.")


def tune_k_validate_input(
    k0: tf.Tensor,
    pilot_length: int,
    target_low: float,
    target_high: float,
    max_rounds: int,
    factor: float,
    name: str,
) -> None:
    """
    Validation contract for tune_k.

    Strict (no coercion):
      - theta0: tf.float64 tensor (rank unrestricted; depends on step_fn)
      - k0: scalar tf.float64 > 0
      - pilot_length, max_rounds: int > 0
      - factor: float > 1
      - target band: 0 <= low <= high <= 1
      - name: str (for logging only)
    """
    _require_float64_tensor(k0, "k0")
    _require_tensor_rank(k0, 0, "k0")
    _require_positive_int(pilot_length, "pilot_length")
    _require_positive_int(max_rounds, "max_rounds")

    _require_prob_band(target_low, target_high, "target_low", "target_high")

    _require_float(factor, "factor")
    _require(float(factor) > 1.0, f"factor must be > 1; got {factor}.")

    _require_type(name, str, "name must be a str.")


def tune_shrinkage_validate_input(shrink) -> None:
    """
    Validation contract for tune_shrinkage(shrink).


    Strict:
      - shrink must expose the exact attributes required by tuning.
      - tuning control attributes must satisfy basic type/value constraints.
      - state fields used with .read_value() must be tf.Variable (no backward compat).

    Note: this is Python-side; do not call it from inside tf.function.
    """
    required = [
        "pilot_length",
        "ridge",
        "target_low",
        "target_high",
        "max_rounds",
        "factor_rw",
        "factor_tmh",
        "T",
        "qjt",
        "q0t",
        "pjt",
        "wjt",
        "beta_p",
        "beta_w",
        "r",
        "E_bar",
        "njt",
        "gamma",
        "phi",
        "posterior",
        "rng",
    ]
    missing = [name for name in required if not hasattr(shrink, name)]
    _require(not missing, "tune_shrinkage missing attributes: " + ", ".join(missing))

    # Validate the scalar tuning controls (strict).
    pilot_length = shrink.pilot_length
    max_rounds = shrink.max_rounds
    ridge = shrink.ridge
    target_low = shrink.target_low
    target_high = shrink.target_high
    factor_rw = shrink.factor_rw
    factor_tmh = shrink.factor_tmh

    _require_type(pilot_length, int, "pilot_length must be an int.")
    _require(pilot_length > 0, f"pilot_length must be > 0; got {pilot_length}.")

    _require_type(max_rounds, int, "max_rounds must be an int.")
    _require(max_rounds > 0, f"max_rounds must be > 0; got {max_rounds}.")

    _require_float(ridge, "ridge")
    _require(float(ridge) >= 0.0, f"ridge must be >= 0; got {ridge}.")

    _require_prob_band(target_low, target_high, "target_low", "target_high")

    _require_float(factor_rw, "factor_rw")
    _require_float(factor_tmh, "factor_tmh")
    _require(float(factor_rw) > 1.0, f"factor_rw must be > 1; got {factor_rw}.")
    _require(float(factor_tmh) > 1.0, f"factor_tmh must be > 1; got {factor_tmh}.")

    # Require the key data tensors exist and are tf.float64 with expected ranks.

    # Data tensors (must be tf.Tensor, float64).
    for name, rank in [("qjt", 2), ("pjt", 2), ("wjt", 2), ("q0t", 1)]:
        x = getattr(shrink, name)
        _require_float64_tensor(x, name)
        _require_tensor_rank(x, rank, name)

    # State must be tf.Variable (since tune_shrinkage uses .read_value()).
    for name, rank in [
        ("beta_p", 0),
        ("beta_w", 0),
        ("r", 0),
        ("E_bar", 1),
        ("njt", 2),
        ("gamma", 2),
        ("phi", 1),
    ]:
        x = getattr(shrink, name)
        _require_float64_variable(x, name)
        _require_tensor_rank(x.read_value(), rank, f"{name}.read_value()")


# -----------------------------------------------------------------------------
# TMH validation (TF-side) — callable inside @tf.function
# -----------------------------------------------------------------------------


@tf.function(reduce_retracing=True)
def tmh_step_validate_input_tf(
    theta0: tf.Tensor,
    k: tf.Tensor,
    ridge: tf.Tensor,
) -> None:
    """
    TF-graph-safe validation for tmh_step inputs.

    Intended usage: call inside the same @tf.function that calls tmh_step.

    Enforced:
      - theta0 is rank-1 tf.float64
      - k is scalar tf.float64 and k > 0
      - ridge is scalar tf.float64 and ridge >= 0
      - all are finite
    """
    tf.debugging.assert_type(theta0, tf.float64)
    tf.debugging.assert_rank(theta0, 1)
    tf.debugging.assert_all_finite(theta0, "theta0 contains non-finite values.")

    tf.debugging.assert_type(k, tf.float64)
    tf.debugging.assert_rank(k, 0)
    tf.debugging.assert_all_finite(k, "k is non-finite.")
    tf.debugging.assert_greater(k, tf.constant(0.0, tf.float64), "k must be > 0.")

    tf.debugging.assert_type(ridge, tf.float64)
    tf.debugging.assert_rank(ridge, 0)
    tf.debugging.assert_all_finite(ridge, "ridge is non-finite.")
    tf.debugging.assert_greater_equal(
        ridge, tf.constant(0.0, tf.float64), "ridge must be >= 0."
    )
