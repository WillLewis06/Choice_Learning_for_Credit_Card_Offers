"""
Shared pytest fixtures and helper functions for the Lu (Section 4) test suite.

Design goals
- Centralize repeated tiny (T=2, J=3) problem construction used across Lu tests.
- Centralize common test-only helpers (finite checks, simplex checks, bool-like checks).
- Centralize deterministic generators used by BLP inversion tests (fixed draws, feasible shares).

Notes
- TensorFlow helpers in this file are eager-mode test utilities. Several helpers call
  `.numpy()` to provide clear assertion messages and are not intended to be used in
  tf.function-compiled code.
- Fixtures in this file return mutable NumPy arrays. Tests should treat fixture outputs
  as read-only unless a test is explicitly about mutation behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorTF


# -----------------------------------------------------------------------------
# Generic helper functions (NumPy)
# -----------------------------------------------------------------------------
def assert_finite_np(x: np.ndarray, name: str = "array") -> None:
    """
    Assert that all entries of `x` are finite (not NaN/inf).

    Raises AssertionError with a short index preview if non-finite values are found.
    """
    x = np.asarray(x)
    ok = np.isfinite(x)
    if not np.all(ok):
        idx = np.argwhere(~ok)
        preview = idx[:5].tolist()
        raise AssertionError(f"{name} contains non-finite values at indices {preview}.")


def fixed_draws(n: int = 500, seed: int = 123) -> np.ndarray:
    """
    Deterministic standard-normal draws for inversion tests.

    The intent is test stability: using the same draws makes simulated inversion
    tests deterministic.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_feasible_shares(J: int, tiny: bool, seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    Construct feasible (s_obs, s0) with strictly positive shares that sum to 1.

    Inputs
    - J: number of inside goods
    - tiny: if True, product 0 gets a very small share to stress numerical stability
    - seed: RNG seed used only to stabilize tests

    Returns
    - s_obs: (J,) inside shares, strictly positive
    - s0: scalar outside share in (0, 1)
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
        # Convention: place the tiny share in position 0 for deterministic stress testing.
        s_obs = np.concatenate([np.array([tiny_share]), s_rest])
    else:
        s0 = 0.30
        w = rng.random(J)
        w = w / w.sum()
        s_obs = (1.0 - s0) * w

    # Safety: strict positivity and normalization.
    assert np.all(s_obs > 0.0)
    assert 0.0 < s0 < 1.0
    assert np.isclose(float(s_obs.sum()) + float(s0), 1.0, atol=1e-14)

    return s_obs, float(s0)


# -----------------------------------------------------------------------------
# Generic helper functions (TensorFlow; eager-mode test utilities)
# -----------------------------------------------------------------------------
def assert_all_finite_tf(*tensors: Any) -> None:
    """
    Assert all provided tensors contain only finite values.

    Eager-mode helper: converts inputs to tf.Tensor and uses `.numpy()` to produce
    informative assertion failures (not intended for tf.function code paths).
    """
    for i, x in enumerate(tensors):
        x_t = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x_t))

        if bool(ok.numpy()):
            continue

        rank = x_t.shape.rank
        if rank == 0:
            raise AssertionError(
                f"Tensor {i} (scalar, dtype={x_t.dtype}) is non-finite: value={x_t.numpy()!r}."
            )

        bad_idx = tf.where(~tf.math.is_finite(x_t)).numpy()
        preview = bad_idx[:5].tolist()
        raise AssertionError(
            f"Tensor {i} (shape={tuple(x_t.shape)}, dtype={x_t.dtype}) "
            f"contains non-finite values at indices {preview}."
        )


def is_bool_like_tf(x: Any) -> bool:
    """
    Return True if `x` is bool-like:
    - dtype is tf.bool, or
    - numeric dtype and all realized values are in {0, 1}.

    Eager-mode helper: inspects realized values via `.numpy()`.
    """
    x_t = tf.convert_to_tensor(x)
    if x_t.dtype == tf.bool:
        return True
    if x_t.dtype.is_floating or x_t.dtype.is_integer:
        uniq = np.unique(x_t.numpy())
        return bool(np.all(np.isin(uniq, [0, 1])))
    return False


def assert_bool_like_tf(x: Any) -> None:
    """
    Assert `x` is bool-like (tf.bool or numeric with values in {0, 1}).
    """
    x_t = tf.convert_to_tensor(x)
    if not is_bool_like_tf(x_t):
        raise AssertionError(
            f"Expected bool-like (bool or 0/1), got dtype={x_t.dtype}."
        )


def assert_binary_01_tf(x: Any) -> None:
    """
    Assert `x` contains only {0, 1} values (numeric), using realized values in eager mode.
    """
    x_t = tf.convert_to_tensor(x)
    uniq = np.unique(x_t.numpy())
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise AssertionError(f"Expected exactly 0/1 values, got {uniq}.")


def assert_in_open_unit_interval_tf(x: Any) -> None:
    """
    Assert all realized values of `x` lie strictly in (0, 1), in eager mode.
    """
    x_t = tf.convert_to_tensor(x)
    xv = x_t.numpy()
    if not np.all(xv > 0.0):
        raise AssertionError(f"Expected > 0, got min={xv.min()}")
    if not np.all(xv < 1.0):
        raise AssertionError(f"Expected < 1, got max={xv.max()}")


def assert_prob_simplex_tf(sjt_t: Any, s0t: Any, atol: float = 1e-12) -> None:
    """
    Assert a single-market outside-option simplex constraint in eager mode.

    Checks
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
# Generic helper functions (TensorFlow; small math / permutation utilities)
# -----------------------------------------------------------------------------
def normal_logpdf_tf(x: Any, mean: Any, var: Any) -> tf.Tensor:
    """
    Elementwise log-density of a Normal(mean, var) distribution.

    Parameters
    - x: value(s) at which to evaluate the density
    - mean: normal mean (broadcastable to x)
    - var: normal variance (broadcastable to x)

    Returns
    - tf.Tensor with the broadcasted shape of x/mean/var (dtype float64)

    Notes
    - This is a test helper used in closed-form sanity checks. It is written for
      clarity (not speed) and is intended for eager-mode tests.
    """
    x_t = tf.convert_to_tensor(x, dtype=tf.float64)
    mean_t = tf.convert_to_tensor(mean, dtype=tf.float64)
    var_t = tf.convert_to_tensor(var, dtype=tf.float64)

    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float64)
    return -0.5 * tf.math.log(two_pi * var_t) - 0.5 * tf.square(x_t - mean_t) / var_t


def permute_vec_tf(x: Any, perm: Any) -> tf.Tensor:
    """
    Permute a 1-D tensor using an index permutation.

    Intended usage: permutation-invariance tests where we reorder products.

    Parameters
    - x: rank-1 tensor-like (J,)
    - perm: rank-1 integer tensor-like (J,), a permutation of [0, ..., J-1]

    Returns
    - tf.Tensor with the same shape as x and permuted entries.
    """
    x_t = tf.convert_to_tensor(x)
    perm_t = tf.convert_to_tensor(perm)
    return tf.gather(x_t, perm_t, axis=0)


def permute_TJ_tf(x: Any, perm: Any) -> tf.Tensor:
    """
    Permute the product axis of a (T, J) tensor using an index permutation.

    Parameters
    - x: tensor-like with shape (T, J)
    - perm: rank-1 integer tensor-like (J,), a permutation of [0, ..., J-1]

    Returns
    - tf.Tensor with the same shape as x and columns permuted by `perm`.
    """
    x_t = tf.convert_to_tensor(x)
    perm_t = tf.convert_to_tensor(perm)
    return tf.gather(x_t, perm_t, axis=1)


def assert_scalar_positive_tf(x: Any, name: str = "x") -> None:
    """
    Assert `x` is a finite, strictly positive scalar in eager mode.

    This is used by tuning tests that expect positive step sizes, scales, or
    other scalar hyperparameters.

    Parameters
    - x: tensor-like scalar
    - name: label used in error messages
    """
    x_t = tf.convert_to_tensor(x, dtype=tf.float64)
    if x_t.shape.rank != 0:
        raise AssertionError(f"{name} must be a scalar; got shape={tuple(x_t.shape)}.")
    assert_all_finite_tf(x_t)
    if not (float(x_t.numpy()) > 0.0):
        raise AssertionError(f"{name} must be > 0; got value={float(x_t.numpy())}.")


# -----------------------------------------------------------------------------
# Canonical tiny Lu market (NumPy)
# -----------------------------------------------------------------------------
@pytest.fixture
def tiny_market_np() -> Dict[str, Any]:
    """
    Canonical tiny market used across Lu tests.

    Shapes
    - pjt, wjt, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    """
    T, J = 2, 3

    # Common tiny variant used across posterior/updates/tuning.
    pjt = np.array([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=np.float64)
    wjt = np.array([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=np.float64)

    qjt = np.array([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=np.float64)
    q0t = np.array([20.0, 15.0], dtype=np.float64)

    return {"T": T, "J": J, "pjt": pjt, "wjt": wjt, "qjt": qjt, "q0t": q0t}


@pytest.fixture
def tiny_market_data(tiny_market_np: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backwards-compatible alias for tests that use `tiny_market_data`.

    Returns a shallow dict copy with array values copied to reduce accidental mutation
    coupling between aliases within a single test.
    """
    out: Dict[str, Any] = {}
    for k, v in tiny_market_np.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v
    return out


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


@pytest.fixture
def tiny_problem(tiny_problem_tf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backwards-compatible alias for tests that use `tiny_problem`.
    """
    return dict(tiny_problem_tf)


@pytest.fixture
def tiny_inputs(tiny_problem_tf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backwards-compatible alias for posterior tests that expect only observed data
    plus latent state (no beta_p/beta_w/r).
    """
    keys = ["T", "J", "pjt", "wjt", "qjt", "q0t", "E_bar", "njt", "gamma", "phi"]
    return {k: tiny_problem_tf[k] for k in keys}


# -----------------------------------------------------------------------------
# Shared posterior / RNG fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def posterior_small() -> LuPosteriorTF:
    """
    Small posterior instance used in unit tests.

    Small `n_draws` keeps tests fast; the seed stabilizes stochastic tests.
    """
    return LuPosteriorTF(n_draws=25, seed=123, dtype=tf.float64)


@pytest.fixture
def posterior(posterior_small: LuPosteriorTF) -> LuPosteriorTF:
    """
    Backwards-compatible alias for tests that use `posterior`.
    """
    return posterior_small


@pytest.fixture
def rng() -> tf.random.Generator:
    """
    Deterministic TF RNG used by kernel / sampler tests.
    """
    return tf.random.Generator.from_seed(123)
