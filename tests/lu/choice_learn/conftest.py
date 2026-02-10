"""
Shared pytest fixtures and helper functions for the choice-learn + Lu sparse-shock
test suite.

Design goals:
- Centralize repeated tiny (T=2, J=3) problem construction used across:
  - cl_shrinkage / cl_tuning / cl_posterior / cl_updates tests
- Centralize common test-only helpers (finite checks, simplex checks, bool-like checks).
- Centralize generic inversion-data generators used in any inversion-style tests.

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

from lu.choice_learn.cl_posterior import LuPosteriorTF


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
# Canonical tiny choice-learn + Lu market (NumPy)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_market_np() -> Dict[str, Any]:
    """
    Canonical tiny market used across choice-learn + Lu tests.

    Shapes:
    - delta_cl, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    """
    T, J = 2, 3

    # Fixed baseline logits (e.g., from the trained choice-learn model).
    delta_cl = np.array(
        [[0.20, -0.10, 0.00], [0.05, 0.15, -0.20]],
        dtype=np.float64,
    )

    # Observed counts (inside + outside).
    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return {"T": T, "J": J, "delta_cl": delta_cl, "qjt": qjt, "q0t": q0t}


# Backwards-compatible alias for tests that currently use this name.
@pytest.fixture
def tiny_market_data(tiny_market_np: Dict[str, Any]) -> Dict[str, Any]:
    return dict(tiny_market_np)


# -----------------------------------------------------------------------------
# Canonical tiny market + latent state bundle (TensorFlow)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_problem_tf(tiny_market_np: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical tiny tensors for posterior/update tests.

    Includes:
    - observed data tensors: delta_cl, qjt, q0t
    - latent state tensors: alpha, E_bar, njt, gamma, phi
    """
    T, J = int(tiny_market_np["T"]), int(tiny_market_np["J"])

    delta_cl = tf.constant(tiny_market_np["delta_cl"], dtype=tf.float64)
    qjt = tf.constant(tiny_market_np["qjt"], dtype=tf.float64)
    q0t = tf.constant(tiny_market_np["q0t"], dtype=tf.float64)

    alpha = tf.constant(1.0, dtype=tf.float64)

    E_bar = tf.constant([0.1, -0.2], dtype=tf.float64)
    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=tf.float64)

    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=tf.float64)
    phi = tf.constant([0.6, 0.4], dtype=tf.float64)

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
        "alpha": alpha,
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
    # Posterior tests often want only observed data + latent state.
    keys = ["T", "J", "delta_cl", "qjt", "q0t", "alpha", "E_bar", "njt", "gamma", "phi"]
    return {k: tiny_problem_tf[k] for k in keys}


# -----------------------------------------------------------------------------
# Shared posterior / RNG fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def posterior_small() -> LuPosteriorTF:
    return LuPosteriorTF(dtype=tf.float64)


# Backwards-compatible alias used by existing tests.
@pytest.fixture
def posterior(posterior_small: LuPosteriorTF) -> LuPosteriorTF:
    return posterior_small


@pytest.fixture
def rng() -> tf.random.Generator:
    return tf.random.Generator.from_seed(123)
