# tests/conftest.py
"""
Shared pytest fixtures and helper functions for the Lu (Section 4) test suite.

Design goals:
- Centralize repeated tiny (T=2, J=3) problem construction used across:
  - test_lu_shrinkage.py
  - test_lu_tuning.py
  - test_lu_posterior.py
  - test_lu_updates.py
- Centralize common test-only helpers (finite checks, simplex checks, bool-like checks).
- Centralize generic inversion-data generators used in test_inversion.py.

Non-goals:
- No pytest markers / skipping logic.
- No TF dtype fixtures (tests can write tf.float64 inline).
- No explicit seed-replicability tests; seeds here are only to keep stochastic tests stable.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow as tf

from market_shock_estimators.lu_posterior import LuPosteriorTF


# -----------------------------------------------------------------------------
# Generic helper functions (NumPy)
# -----------------------------------------------------------------------------
def assert_finite_np(x: np.ndarray, *, name: str = "array") -> None:
    x = np.asarray(x)
    ok = np.isfinite(x)
    if not np.all(ok):
        idx = np.argwhere(~ok)
        # show first few offending entries
        preview = idx[:5].tolist()
        raise AssertionError(f"{name} contains non-finite values at indices {preview}.")


def fixed_draws(n: int = 500, seed: int = 123) -> np.ndarray:
    """
    Deterministic standard-normal draws for inversion tests.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_feasible_shares(
    J: int, *, tiny: bool, seed: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Construct feasible (s_obs, s0) with strictly positive shares that sum to 1.

    If tiny=True, one product gets a very small share to stress numerical stability.
    """
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

    # Safety: strict positivity and exact normalization.
    assert np.all(s_obs > 0.0)
    assert 0.0 < s0 < 1.0
    assert np.isclose(float(s_obs.sum()) + float(s0), 1.0, atol=1e-14)

    return s_obs, float(s0)


# -----------------------------------------------------------------------------
# Generic helper functions (TensorFlow)
# -----------------------------------------------------------------------------
def assert_all_finite_tf(*tensors: Any) -> None:
    for x in tensors:
        x = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x))
        if not bool(ok.numpy()):
            raise AssertionError("Found non-finite values.")


def is_bool_like_tf(x: Any) -> bool:
    x = tf.convert_to_tensor(x)
    if x.dtype == tf.bool:
        return True
    if x.dtype.is_floating or x.dtype.is_integer:
        uniq = np.unique(x.numpy())
        return bool(np.all(np.isin(uniq, [0, 1])))
    return False


def assert_bool_like_tf(x: Any) -> None:
    x = tf.convert_to_tensor(x)
    if not is_bool_like_tf(x):
        raise AssertionError(f"Expected bool-like (bool or 0/1), got dtype={x.dtype}.")


def assert_binary_01_tf(x: Any) -> None:
    x = tf.convert_to_tensor(x)
    uniq = np.unique(x.numpy())
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise AssertionError(f"Expected exactly 0/1 values, got {uniq}.")


def assert_in_open_unit_interval_tf(x: Any) -> None:
    x = tf.convert_to_tensor(x)
    xv = x.numpy()
    if not np.all(xv > 0.0):
        raise AssertionError(f"Expected > 0, got min={xv.min()}")
    if not np.all(xv < 1.0):
        raise AssertionError(f"Expected < 1, got max={xv.max()}")


def assert_prob_simplex_tf(sjt_t: Any, s0t: Any, *, atol: float = 1e-12) -> None:
    """
    Checks:
    - sjt_t is rank-1 (J,)
    - s0t is scalar
    - all finite
    - bounds within [-atol, 1+atol]
    - simplex identity: sum(sjt_t) + s0t == 1 within atol
    """
    sjt_t = tf.convert_to_tensor(sjt_t)
    s0t = tf.convert_to_tensor(s0t)

    assert sjt_t.shape.rank == 1
    assert s0t.shape.rank == 0

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
# Canonical tiny Lu market (NumPy)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_market_np() -> Dict[str, Any]:
    """
    Canonical tiny market used across Lu tests.

    Shapes:
    - pjt, wjt, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    """
    T, J = 2, 3

    # Use the most common variant across posterior/updates/tuning.
    pjt = np.array([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=np.float64)
    wjt = np.array([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=np.float64)

    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return {"T": T, "J": J, "pjt": pjt, "wjt": wjt, "qjt": qjt, "q0t": q0t}


# Backwards-compatible alias for tests that currently use this name.
@pytest.fixture
def tiny_market_data(tiny_market_np: Dict[str, Any]) -> Dict[str, Any]:
    return dict(tiny_market_np)


# -----------------------------------------------------------------------------
# Canonical tiny Lu market + latent state bundle (TensorFlow)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_problem_tf(tiny_market_np: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical tiny tensors for posterior/update tests.

    Includes:
    - observed data tensors: pjt, wjt, qjt, q0t
    - latent state tensors: E_bar, njt, gamma, phi
    - parameter tensors: beta_p, beta_w, r
    """
    T, J = int(tiny_market_np["T"]), int(tiny_market_np["J"])

    pjt = tf.constant(tiny_market_np["pjt"], dtype=tf.float64)
    wjt = tf.constant(tiny_market_np["wjt"], dtype=tf.float64)
    qjt = tf.constant(tiny_market_np["qjt"], dtype=tf.float64)
    q0t = tf.constant(tiny_market_np["q0t"], dtype=tf.float64)

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

    E_bar = tf.constant([0.1, -0.2], dtype=tf.float64)
    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=tf.float64)

    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=tf.float64)
    phi = tf.constant([0.6, 0.4], dtype=tf.float64)

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


# Backwards-compatible aliases for current test names.
@pytest.fixture
def tiny_problem(tiny_problem_tf: Dict[str, Any]) -> Dict[str, Any]:
    return dict(tiny_problem_tf)


@pytest.fixture
def tiny_inputs(tiny_problem_tf: Dict[str, Any]) -> Dict[str, Any]:
    # posterior tests typically don't need beta_p/beta_w/r
    keys = ["T", "J", "pjt", "wjt", "qjt", "q0t", "E_bar", "njt", "gamma", "phi"]
    return {k: tiny_problem_tf[k] for k in keys}


# -----------------------------------------------------------------------------
# Shared posterior / RNG fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def posterior_small() -> LuPosteriorTF:
    # Small n_draws keeps tests fast; seed keeps stochastic tests stable.
    return LuPosteriorTF(n_draws=25, seed=123, dtype=tf.float64)


# Backwards-compatible alias used by existing tests.
@pytest.fixture
def posterior(posterior_small: LuPosteriorTF) -> LuPosteriorTF:
    return posterior_small


@pytest.fixture
def rng() -> tf.random.Generator:
    return tf.random.Generator.from_seed(123)
