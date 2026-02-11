"""
Shared test helpers for the Lu test suite (no pytest fixtures).

This module is a normal Python helper module:
- No @pytest.fixture usage
- Tests should call the constructors directly (e.g., tiny_market_np()).

Notes
- Helpers here are eager-mode utilities intended for tests.
- Some helpers call `.numpy()` to produce clearer assertion failures.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorTF


# -----------------------------------------------------------------------------
# NumPy helpers
# -----------------------------------------------------------------------------
def assert_finite_np(x: np.ndarray, name: str = "array") -> None:
    x = np.asarray(x)
    ok = np.isfinite(x)
    if not np.all(ok):
        idx = np.argwhere(~ok)
        preview = idx[:5].tolist()
        raise AssertionError(f"{name} contains non-finite values at indices {preview}.")


def fixed_draws(n: int = 500, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_feasible_shares(J: int, tiny: bool, seed: int = 0) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    if tiny:
        s0 = 0.25
        tiny_share = 1e-8
        remaining_mass = 1.0 - s0 - tiny_share
        if remaining_mass <= 0.0:
            raise ValueError("Invalid construction: remaining mass <= 0.")

        w = rng.random(J - 1)
        w = w / w.sum()
        s_rest = remaining_mass * w
        s_obs = np.concatenate([np.array([tiny_share]), s_rest])
    else:
        s0 = 0.30
        w = rng.random(J)
        w = w / w.sum()
        s_obs = (1.0 - s0) * w

    if not np.all(s_obs > 0.0):
        raise AssertionError("Inside shares must be strictly positive.")
    if not (0.0 < s0 < 1.0):
        raise AssertionError("Outside share must be strictly between 0 and 1.")
    if not np.isclose(float(s_obs.sum()) + float(s0), 1.0, atol=1e-14):
        raise AssertionError("Shares must sum to 1.")

    return s_obs, float(s0)


# -----------------------------------------------------------------------------
# TensorFlow eager-mode assertion helpers
# -----------------------------------------------------------------------------
def assert_all_finite_tf(*tensors: Any) -> None:
    for i, x in enumerate(tensors):
        x_t = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x_t))
        if bool(ok.numpy()):
            continue

        if x_t.shape.rank == 0:
            raise AssertionError(
                f"Tensor {i} (scalar, dtype={x_t.dtype}) is non-finite: value={x_t.numpy()!r}."
            )

        bad_idx = tf.where(~tf.math.is_finite(x_t)).numpy()
        preview = bad_idx[:5].tolist()
        raise AssertionError(
            f"Tensor {i} (shape={tuple(x_t.shape)}, dtype={x_t.dtype}) "
            f"contains non-finite values at indices {preview}."
        )


def assert_binary_01_tf(x: Any) -> None:
    x_t = tf.convert_to_tensor(x)
    uniq = np.unique(x_t.numpy())
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise AssertionError(f"Expected exactly 0/1 values, got {uniq}.")


def assert_in_open_unit_interval_tf(x: Any) -> None:
    x_t = tf.convert_to_tensor(x)
    xv = x_t.numpy()
    if not np.all(xv > 0.0):
        raise AssertionError(f"Expected > 0, got min={xv.min()}.")
    if not np.all(xv < 1.0):
        raise AssertionError(f"Expected < 1, got max={xv.max()}.")


def assert_prob_simplex_tf(sjt_t: Any, s0t: Any, atol: float = 1e-12) -> None:
    sjt_t = tf.convert_to_tensor(sjt_t)
    s0t = tf.convert_to_tensor(s0t)

    if sjt_t.shape.rank != 1:
        raise AssertionError("sjt_t must be rank-1 (J,).")
    if s0t.shape.rank != 0:
        raise AssertionError("s0t must be a scalar.")

    assert_all_finite_tf(sjt_t, s0t)

    sjt = sjt_t.numpy()
    s0 = float(s0t.numpy())

    if np.min(sjt) < -atol or np.max(sjt) > 1.0 + atol:
        raise AssertionError("Inside shares violate bounds.")
    if s0 < -atol or s0 > 1.0 + atol:
        raise AssertionError("Outside share violates bounds.")

    err = abs(float(np.sum(sjt)) + s0 - 1.0)
    if err > atol:
        raise AssertionError(
            f"Simplex identity violated (err={err:.3e}, atol={atol:.3e})."
        )


# -----------------------------------------------------------------------------
# Small math / permutation utilities
# -----------------------------------------------------------------------------
def normal_logpdf_tf(x: Any, mean: Any, var: Any) -> tf.Tensor:
    x_t = tf.convert_to_tensor(x, dtype=tf.float64)
    mean_t = tf.convert_to_tensor(mean, dtype=tf.float64)
    var_t = tf.convert_to_tensor(var, dtype=tf.float64)

    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float64)
    return -0.5 * tf.math.log(two_pi * var_t) - 0.5 * tf.square(x_t - mean_t) / var_t


def permute_vec_tf(x: Any, perm: Any) -> tf.Tensor:
    x_t = tf.convert_to_tensor(x)
    perm_t = tf.convert_to_tensor(perm)
    return tf.gather(x_t, perm_t, axis=0)


def permute_TJ_tf(x: Any, perm: Any) -> tf.Tensor:
    x_t = tf.convert_to_tensor(x)
    perm_t = tf.convert_to_tensor(perm)
    return tf.gather(x_t, perm_t, axis=1)


# -----------------------------------------------------------------------------
# Canonical tiny market constructors (NO fixtures)
# -----------------------------------------------------------------------------
def tiny_market_np() -> Dict[str, Any]:
    """
    Canonical tiny market used across Lu tests.

    Shapes
    - pjt, wjt, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    """
    T, J = 2, 3
    pjt = np.array([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=np.float64)
    wjt = np.array([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=np.float64)
    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)

    # Critical: q0t must be a real array; do not reference any undefined name like `n`.
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return {"T": T, "J": J, "pjt": pjt, "wjt": wjt, "qjt": qjt, "q0t": q0t}


def tiny_market_data() -> Dict[str, Any]:
    """
    Backwards-compatible alias used by some tests.

    Returns a shallow copy with arrays copied to reduce mutation coupling.
    """
    base = tiny_market_np()
    out: Dict[str, Any] = {}
    for k, v in base.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v
    return out


def tiny_problem_tf(
    tiny_market: Dict[str, Any] | None = None,
    dtype: tf.dtypes.DType = tf.float64,
) -> Dict[str, Any]:
    if tiny_market is None:
        tiny_market = tiny_market_np()

    T, J = int(tiny_market["T"]), int(tiny_market["J"])

    pjt = tf.constant(tiny_market["pjt"], dtype=dtype)
    wjt = tf.constant(tiny_market["wjt"], dtype=dtype)
    qjt = tf.constant(tiny_market["qjt"], dtype=dtype)
    q0t = tf.constant(tiny_market["q0t"], dtype=dtype)

    beta_p = tf.constant(-1.0, dtype=dtype)
    beta_w = tf.constant(0.3, dtype=dtype)
    r = tf.constant(0.0, dtype=dtype)

    E_bar = tf.constant([0.1, -0.2], dtype=dtype)
    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=dtype)

    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype)
    phi = tf.constant([0.6, 0.4], dtype=dtype)

    return {
        "T": T,
        "J": J,
        "pjt": pjt,
        "wjt": wjt,
        "qjt": qjt,
        "q0t": q0t,
        "beta_p": beta_p,
        "beta_w": beta_w,
        "r": r,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
        "phi": phi,
    }


def tiny_inputs(dtype: tf.dtypes.DType = tf.float64) -> Dict[str, Any]:
    prob = tiny_problem_tf(dtype=dtype)
    keys = ["T", "J", "pjt", "wjt", "qjt", "q0t", "E_bar", "njt", "gamma", "phi"]
    return {k: prob[k] for k in keys}


def posterior_small(
    n_draws: int = 25,
    seed: int = 123,
    dtype: tf.dtypes.DType = tf.float64,
) -> LuPosteriorTF:
    return LuPosteriorTF(n_draws=n_draws, seed=seed, dtype=dtype)


def rng(seed: int = 123) -> tf.random.Generator:
    return tf.random.Generator.from_seed(seed)
