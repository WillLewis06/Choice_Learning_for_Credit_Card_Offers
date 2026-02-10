# tests/ching/conftest.py
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict

import numpy as np
import pytest

# Must be set before importing TensorFlow to suppress most C++ logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from ching.stockpiling_estimator import StockpilingEstimator  # noqa: E402
from ching.stockpiling_posterior import build_inventory_maps  # noqa: E402


def _stable_int_hash(text: str, mod: int = 1_000_000) -> int:
    """Deterministic hash for per-test seeding (independent of PYTHONHASHSEED)."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def pytest_configure(config: Any) -> None:
    """Global pytest configuration for TF-heavy tests."""
    # Make TF python logging quieter; TF_CPP_MIN_LOG_LEVEL handles most C++ logs.
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass


# =============================================================================
# Core small test configuration
# =============================================================================


@pytest.fixture(scope="session")
def tiny_dims() -> Dict[str, int]:
    """
    Small dimensions used across Phase-3 tests.

    Conventions:
      a_imt:      (M, N, T)
      p_state_mt: (M, T)
      ccp_buy:    (M, N, S, I) where I = I_max + 1
    """
    return {"M": 2, "N": 3, "T": 5, "S": 2, "I_max": 2}


@pytest.fixture(scope="session")
def tiny_dp_config() -> Dict[str, Any]:
    """DP / filtering configuration used in posterior and estimator tests."""
    return {
        "waste_cost": 0.1,
        "eps": 1.0e-12,
        "tol": 1.0e-8,
        "max_iter": 200,
    }


@pytest.fixture(scope="session")
def price_process(tiny_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Prices by state and Markov transition matrix for price states."""
    S = tiny_dims["S"]
    if S != 2:
        # Keep defaults simple; if you later parametrize S>2, update this fixture.
        raise ValueError("price_process fixture currently assumes S=2.")

    price_vals = np.asarray([1.0, 0.8], dtype=np.float64)
    P_price = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    return {"price_vals": price_vals, "P_price": P_price}


@pytest.fixture(scope="session")
def pi_I0_uniform(tiny_dims: Dict[str, int]) -> np.ndarray:
    """Uniform initial inventory belief pi_I0 over {0,...,I_max}."""
    I = tiny_dims["I_max"] + 1
    return (np.ones(I, dtype=np.float64) / float(I)).astype(np.float64)


# =============================================================================
# Seeding / RNG fixtures
# =============================================================================


@pytest.fixture(scope="session")
def seed_base() -> int:
    """Base seed for the Phase-3 test suite."""
    return 12345


@pytest.fixture(scope="function")
def test_seed(request: pytest.FixtureRequest, seed_base: int) -> int:
    """Per-test deterministic seed derived from the test node id."""
    return int(seed_base + _stable_int_hash(request.node.nodeid))


@pytest.fixture(scope="function")
def np_rng(test_seed: int) -> np.random.Generator:
    """Per-test NumPy RNG."""
    return np.random.default_rng(int(test_seed))


@pytest.fixture(scope="function")
def tf_seed(test_seed: int) -> None:
    """Set TensorFlow global seed for a test."""
    tf.random.set_seed(int(test_seed))


# =============================================================================
# Shared synthetic data fixtures (NumPy-first)
# =============================================================================


@pytest.fixture(scope="function")
def panel_np(tiny_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Deterministic, non-degenerate panel:
      - p_state_mt cycles through states (ensures both states appear).
      - a_imt uses a deterministic pattern with both 0 and 1.
    """
    M, N, T, S = tiny_dims["M"], tiny_dims["N"], tiny_dims["T"], tiny_dims["S"]

    # Price states: cycle through {0,...,S-1}, shift by market index.
    p_state_mt = np.empty((M, T), dtype=np.int32)
    for m in range(M):
        p_state_mt[m, :] = (np.arange(T, dtype=np.int32) + m) % S

    # Actions: deterministic pattern across (m,n,t), contains both 0 and 1.
    a_imt = np.empty((M, N, T), dtype=np.int32)
    for m in range(M):
        for n in range(N):
            a_imt[m, n, :] = (
                (m * 31 + n * 17 + np.arange(T, dtype=np.int32)) % 2
            ).astype(np.int32)

    return {"a_imt": a_imt, "p_state_mt": p_state_mt}


@pytest.fixture(scope="session")
def u_m_np(tiny_dims: Dict[str, int]) -> np.ndarray:
    """Market utilities u_m (M,) used as fixed inputs in Phase-3."""
    M = tiny_dims["M"]
    return np.linspace(0.5, 1.5, M, dtype=np.float64)


# =============================================================================
# Priors and parameter blocks
# =============================================================================


@pytest.fixture(scope="session")
def sigmas() -> Dict[str, float]:
    """Prior scales for z-blocks (must include all keys used by posterior/estimator)."""
    return {
        "z_beta": 1.0,
        "z_alpha": 1.0,
        "z_v": 1.0,
        "z_fc": 1.0,
        "z_lambda": 1.0,
        "z_u_scale": 1.0,
    }


@pytest.fixture(scope="function")
def theta_constrained_np(tiny_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Reasonable constrained theta values (primarily for evaluation/predictive tests).
    Shapes:
      beta, alpha, v, fc, lambda_c: (M,N)
      u_scale: (M,)
    """
    M, N = tiny_dims["M"], tiny_dims["N"]

    beta = np.full((M, N), 0.6, dtype=np.float64)
    alpha = np.full((M, N), 1.0, dtype=np.float64)
    v = np.full((M, N), 2.0, dtype=np.float64)
    fc = np.full((M, N), 0.2, dtype=np.float64)
    lambda_c = np.full((M, N), 0.3, dtype=np.float64)
    u_scale = np.full((M,), 1.0, dtype=np.float64)

    return {
        "beta": beta,
        "alpha": alpha,
        "v": v,
        "fc": fc,
        "lambda_c": lambda_c,
        "u_scale": u_scale,
    }


@pytest.fixture(scope="function")
def z_blocks_np(tiny_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Prior-mode unconstrained blocks (all zeros).
    Shapes:
      z_beta, z_alpha, z_v, z_fc, z_lambda: (M,N)
      z_u_scale: (M,)
    """
    M, N = tiny_dims["M"], tiny_dims["N"]
    z_mn = np.zeros((M, N), dtype=np.float64)
    z_m = np.zeros((M,), dtype=np.float64)

    return {
        "z_beta": z_mn.copy(),
        "z_alpha": z_mn.copy(),
        "z_v": z_mn.copy(),
        "z_fc": z_mn.copy(),
        "z_lambda": z_mn.copy(),
        "z_u_scale": z_m.copy(),
    }


# =============================================================================
# TF-converted fixtures for posterior-level unit tests
# =============================================================================


@pytest.fixture(scope="session")
def inventory_maps_tf(tiny_dims: Dict[str, int]) -> Any:
    """Precomputed inventory maps for a given I_max."""
    I_max_tf = tf.convert_to_tensor(int(tiny_dims["I_max"]), dtype=tf.int32)
    return build_inventory_maps(I_max_tf)


@pytest.fixture(scope="function")
def z_blocks_tf(z_blocks_np: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    """TF float64 tensors for z-blocks (matches posterior/estimator expectations)."""
    return {
        k: tf.convert_to_tensor(v, dtype=tf.float64) for k, v in z_blocks_np.items()
    }


@pytest.fixture(scope="function")
def tf_inputs(
    tf_seed: None,
    panel_np: Dict[str, np.ndarray],
    u_m_np: np.ndarray,
    price_process: Dict[str, np.ndarray],
    pi_I0_uniform: np.ndarray,
    tiny_dp_config: Dict[str, Any],
) -> Dict[str, tf.Tensor]:
    """
    Canonical TF inputs for posterior calls:
      a_imt, p_state_mt: int32
      u_m, price_vals, P_price, pi_I0: float64
      waste_cost, eps, tol: float64
      max_iter: int32
    """
    return {
        "a_imt": tf.convert_to_tensor(panel_np["a_imt"], dtype=tf.int32),
        "p_state_mt": tf.convert_to_tensor(panel_np["p_state_mt"], dtype=tf.int32),
        "u_m": tf.convert_to_tensor(u_m_np, dtype=tf.float64),
        "price_vals": tf.convert_to_tensor(
            price_process["price_vals"], dtype=tf.float64
        ),
        "P_price": tf.convert_to_tensor(price_process["P_price"], dtype=tf.float64),
        "pi_I0": tf.convert_to_tensor(pi_I0_uniform, dtype=tf.float64),
        "waste_cost": tf.convert_to_tensor(
            float(tiny_dp_config["waste_cost"]), dtype=tf.float64
        ),
        "eps": tf.convert_to_tensor(float(tiny_dp_config["eps"]), dtype=tf.float64),
        "tol": tf.convert_to_tensor(float(tiny_dp_config["tol"]), dtype=tf.float64),
        "max_iter": tf.convert_to_tensor(
            int(tiny_dp_config["max_iter"]), dtype=tf.int32
        ),
    }


# =============================================================================
# Optional: silence iteration printing during fit()
# =============================================================================


@pytest.fixture(scope="function")
def silence_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Silence tf.print progress lines by patching the reporting function
    before any tracing occurs in a test.
    """
    import ching.stockpiling_diagnostics as diag_mod

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(diag_mod, "report_iteration_progress", _noop, raising=True)


# =============================================================================
# Estimator fixtures
# =============================================================================


@pytest.fixture(scope="session")
def proposal_scales_tiny() -> Dict[str, float]:
    """Small positive proposal scales for quick fit() smoke tests."""
    return {
        "k_beta": 0.05,
        "k_alpha": 0.05,
        "k_v": 0.05,
        "k_fc": 0.05,
        "k_lambda": 0.05,
        "k_u_scale": 0.05,
    }


@pytest.fixture(scope="function")
def estimator_tiny(
    tf_seed: None,
    silence_progress: None,
    panel_np: Dict[str, np.ndarray],
    u_m_np: np.ndarray,
    price_process: Dict[str, np.ndarray],
    pi_I0_uniform: np.ndarray,
    tiny_dims: Dict[str, int],
    tiny_dp_config: Dict[str, Any],
    sigmas: Dict[str, float],
    test_seed: int,
) -> StockpilingEstimator:
    """
    Ready-to-run StockpilingEstimator instance for small tests.
    """
    return StockpilingEstimator(
        a_imt=panel_np["a_imt"],
        p_state_mt=panel_np["p_state_mt"],
        u_m=u_m_np,
        price_vals=price_process["price_vals"],
        P_price=price_process["P_price"],
        I_max=int(tiny_dims["I_max"]),
        pi_I0=pi_I0_uniform,
        waste_cost=float(tiny_dp_config["waste_cost"]),
        eps=float(tiny_dp_config["eps"]),
        tol=float(tiny_dp_config["tol"]),
        max_iter=int(tiny_dp_config["max_iter"]),
        sigmas=sigmas,
        seed=int(test_seed),
    )
