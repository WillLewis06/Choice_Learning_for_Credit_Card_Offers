# ching_conftest.py
"""
Deterministic test builders for the refactored Ching-style stockpiling code.

This module is imported directly by unit tests rather than used as pytest
fixtures, so it provides plain Python helper functions that construct small,
valid inputs aligned with the current public API:

- observed data uses:
    a_mnjt, s_mjt, u_mj, P_price_mj, price_vals_mj, lambda_mn, waste_cost,
    pi_I0, inventory_maps
- posterior config uses:
    tol, max_iter, eps, sigma_z_beta, sigma_z_alpha, sigma_z_v,
    sigma_z_fc, sigma_z_u_scale
- sampler config uses:
    num_results, chunk_size, k_beta, k_alpha, k_v, k_fc, k_u_scale
- initial state uses explicit unconstrained z-blocks packed into
    StockpilingState
- end-to-end sampling uses:
    run_chain(..., pi_I0, inventory_maps, posterior_config,
              stockpiling_config, initial_state, seed)

The helpers below intentionally keep dimensions tiny and deterministic so tests
are stable and cheap to run.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Reduce TF C++ logging before TensorFlow is imported anywhere downstream.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__all__ = [
    "tiny_dims",
    "tiny_dp_config",
    "price_process",
    "simulate_s_mjt_from_P",
    "panel_np",
    "u_mj_np",
    "lambda_mn_np",
    "pi_I0_uniform",
    "z_blocks_np",
    "posterior_config",
    "sampler_config_tf",
    "initial_state_tf",
    "seed_tf",
    "inventory_maps_tf",
    "core_inputs_np",
    "observed_inputs_tf",
    "posterior_bundle_tf",
    "run_chain_inputs_tf",
]


# =============================================================================
# Core tiny dimensions / numerical controls
# =============================================================================


def tiny_dims() -> dict[str, int]:
    """Return a small deterministic dimension dict shared across tests."""
    return {
        "M": 2,
        "N": 3,
        "J": 2,
        "T": 6,
        "S": 2,
        "I_max": 2,
    }


def tiny_dp_config() -> dict[str, Any]:
    """
    Return small numerical controls used by posterior/model tests.

    Notes:
      - eps is now part of StockpilingPosteriorConfig and is required.
      - tol is a Python float and max_iter is a Python int, matching validation.
    """
    return {
        "waste_cost": 0.25,
        "tol": 1e-10,
        "max_iter": 200,
        "eps": 1e-12,
    }


# =============================================================================
# Price-process builders
# =============================================================================


def price_process(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Build a valid row-stochastic price-state Markov chain and positive prices.

    Returns:
      P_price_mj:    (M, J, S, S) float64
      price_vals_mj: (M, J, S)    float64
    """
    M = int(dims["M"])
    J = int(dims["J"])
    S = int(dims["S"])

    p_stay = 0.85
    P = np.zeros((M, J, S, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            for s in range(S):
                P[m, j, s, :] = (1.0 - p_stay) / float(S - 1)
                P[m, j, s, s] = p_stay

    price_vals = np.zeros((M, J, S), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            base = 2.0 + 0.15 * float(m) + 0.25 * float(j)
            for s in range(S):
                # Monotone state variation while staying strictly positive.
                price_vals[m, j, s] = base * (0.85 ** float(s))

    return {"P_price_mj": P, "price_vals_mj": price_vals}


def simulate_s_mjt_from_P(
    P_price_mj: np.ndarray,
    T: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate an observed price-state path from P_price_mj.

    Returns:
      s_mjt: (M, J, T) int32 with values in {0, ..., S - 1}
    """
    P = np.asarray(P_price_mj, dtype=np.float64)
    M, J, S, _ = P.shape
    T = int(T)

    rng = np.random.default_rng(int(seed))
    s_mjt = np.zeros((M, J, T), dtype=np.int32)

    # Deterministic initial state.
    s_mjt[:, :, 0] = 0
    for t in range(1, T):
        for m in range(M):
            for j in range(J):
                s_prev = int(s_mjt[m, j, t - 1])
                s_next = rng.choice(S, p=P[m, j, s_prev, :])
                s_mjt[m, j, t] = int(s_next)

    return s_mjt


# =============================================================================
# Observed panel and fixed inputs
# =============================================================================


def panel_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Build deterministic observed panel inputs.

    Returns:
      a_mnjt: (M, N, J, T) int32 in {0, 1}
      s_mjt:  (M, J, T)    int32 in {0, ..., S - 1}
    """
    M = int(dims["M"])
    N = int(dims["N"])
    J = int(dims["J"])
    T = int(dims["T"])
    S = int(dims["S"])

    rng = np.random.default_rng(123)
    a_mnjt = (rng.random((M, N, J, T)) < 0.25).astype(np.int32)

    s_mjt = np.zeros((M, J, T), dtype=np.int32)
    for m in range(M):
        for j in range(J):
            for t in range(T):
                s_mjt[m, j, t] = int((m + j + t) % S)

    return {"a_mnjt": a_mnjt, "s_mjt": s_mjt}


def u_mj_np(dims: dict[str, int]) -> np.ndarray:
    """Build a small deterministic Phase-1/2 intercept array. Shape (M, J)."""
    M = int(dims["M"])
    J = int(dims["J"])

    u = np.zeros((M, J), dtype=np.float64)
    for m in range(M):
        for j in range(J):
            u[m, j] = 0.4 + 0.1 * float(m) + 0.05 * float(j)
    return u


def lambda_mn_np(dims: dict[str, int]) -> np.ndarray:
    """Build consumption probabilities in the open unit interval. Shape (M, N)."""
    M = int(dims["M"])
    N = int(dims["N"])

    lam = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        for n in range(N):
            lam[m, n] = 0.2 + 0.1 * float(m) + 0.05 * float(n)

    return np.clip(lam, 1e-3, 1.0 - 1e-3)


def pi_I0_uniform(dims: dict[str, int]) -> np.ndarray:
    """
    Build a uniform initial inventory distribution.

    This helper is used directly by the refactored posterior and sampler path,
    both of which now take pi_I0 explicitly.
    """
    I = int(dims["I_max"]) + 1
    return np.ones((I,), dtype=np.float64) / float(I)


# =============================================================================
# Unconstrained state builders
# =============================================================================


def z_blocks_np(dims: dict[str, int]) -> dict[str, np.ndarray]:
    """
    Build deterministic unconstrained z-blocks with the current sampler shapes.

    Shapes:
      z_beta:    scalar ()
      z_alpha:   (J,)
      z_v:       (J,)
      z_fc:      (J,)
      z_u_scale: (M,)
    """
    M = int(dims["M"])
    J = int(dims["J"])

    return {
        "z_beta": np.asarray(0.0, dtype=np.float64),
        "z_alpha": np.zeros((J,), dtype=np.float64),
        "z_v": np.zeros((J,), dtype=np.float64),
        "z_fc": np.zeros((J,), dtype=np.float64),
        "z_u_scale": np.zeros((M,), dtype=np.float64),
    }


# =============================================================================
# Refactored config/state builders
# =============================================================================


def posterior_config():
    """
    Build a valid StockpilingPosteriorConfig for posterior and sampler tests.
    """
    from ching.stockpiling_posterior import StockpilingPosteriorConfig

    ctrl = tiny_dp_config()
    return StockpilingPosteriorConfig(
        tol=float(ctrl["tol"]),
        max_iter=int(ctrl["max_iter"]),
        eps=float(ctrl["eps"]),
        sigma_z_beta=0.5,
        sigma_z_alpha=0.5,
        sigma_z_v=0.5,
        sigma_z_fc=0.5,
        sigma_z_u_scale=0.5,
    )


def sampler_config_tf(
    dims: dict[str, int],
    num_results: int = 8,
    chunk_size: int = 4,
):
    """
    Build a valid StockpilingConfig with float64 tensor proposal scales.
    """
    import tensorflow as tf
    from ching.stockpiling_estimator import StockpilingConfig

    M = int(dims["M"])
    J = int(dims["J"])

    return StockpilingConfig(
        num_results=int(num_results),
        chunk_size=int(chunk_size),
        k_beta=tf.constant(0.05, dtype=tf.float64),
        k_alpha=tf.fill((J,), tf.constant(0.08, dtype=tf.float64)),
        k_v=tf.fill((J,), tf.constant(0.08, dtype=tf.float64)),
        k_fc=tf.fill((J,), tf.constant(0.08, dtype=tf.float64)),
        k_u_scale=tf.fill((M,), tf.constant(0.05, dtype=tf.float64)),
    )


def initial_state_tf(
    dims: dict[str, int],
    z_blocks: dict[str, np.ndarray] | None = None,
):
    """
    Build a valid StockpilingState from explicit unconstrained z-blocks.

    If z_blocks is omitted, deterministic default z-blocks are constructed from
    dims and passed into the core build_initial_state helper.
    """
    import tensorflow as tf
    from ching.stockpiling_estimator import build_initial_state

    if z_blocks is None:
        z_blocks = z_blocks_np(dims)

    return build_initial_state(
        z_beta=tf.convert_to_tensor(z_blocks["z_beta"], dtype=tf.float64),
        z_alpha=tf.convert_to_tensor(z_blocks["z_alpha"], dtype=tf.float64),
        z_v=tf.convert_to_tensor(z_blocks["z_v"], dtype=tf.float64),
        z_fc=tf.convert_to_tensor(z_blocks["z_fc"], dtype=tf.float64),
        z_u_scale=tf.convert_to_tensor(z_blocks["z_u_scale"], dtype=tf.float64),
    )


def seed_tf(seed: int = 123):
    """
    Build a stateless RNG seed tensor of shape (2,).

    TensorFlow stateless RNG accepts int32 or int64 seeds. Tests use int32 for
    consistency and compactness.
    """
    import tensorflow as tf

    seed = int(seed)
    return tf.constant([seed, seed + 1], dtype=tf.int32)


def inventory_maps_tf(I_max: int):
    """
    Build inventory maps using the core implementation.
    """
    from ching.stockpiling_model import build_inventory_maps

    return build_inventory_maps(int(I_max))


# =============================================================================
# Higher-level bundles used by refactored tests
# =============================================================================


def core_inputs_np() -> dict[str, Any]:
    """
    Build deterministic raw NumPy/scalar inputs shared across tests.

    This helper stops short of TensorFlow conversion so validation tests can
    explicitly control dtypes and conversion paths when needed.
    """
    dims = tiny_dims()
    ctrl = tiny_dp_config()
    panel = panel_np(dims)
    price = price_process(dims)

    return {
        "dims": dims,
        "a_mnjt": panel["a_mnjt"],
        "s_mjt": panel["s_mjt"],
        "u_mj": u_mj_np(dims),
        "P_price_mj": price["P_price_mj"],
        "price_vals_mj": price["price_vals_mj"],
        "lambda_mn": lambda_mn_np(dims),
        "waste_cost": float(ctrl["waste_cost"]),
        "pi_I0": pi_I0_uniform(dims),
    }


def observed_inputs_tf() -> dict[str, Any]:
    """
    Convert deterministic observed inputs into TensorFlow objects.

    Returns the exact observed-data pieces expected by the refactored posterior
    and sampler path, plus dims for convenience.
    """
    import tensorflow as tf

    raw = core_inputs_np()
    dims = raw["dims"]

    return {
        "dims": dims,
        "a_mnjt": tf.convert_to_tensor(raw["a_mnjt"], dtype=tf.int32),
        "s_mjt": tf.convert_to_tensor(raw["s_mjt"], dtype=tf.int32),
        "u_mj": tf.convert_to_tensor(raw["u_mj"], dtype=tf.float64),
        "P_price_mj": tf.convert_to_tensor(raw["P_price_mj"], dtype=tf.float64),
        "price_vals_mj": tf.convert_to_tensor(raw["price_vals_mj"], dtype=tf.float64),
        "lambda_mn": tf.convert_to_tensor(raw["lambda_mn"], dtype=tf.float64),
        "waste_cost": tf.convert_to_tensor(raw["waste_cost"], dtype=tf.float64),
        "pi_I0": tf.convert_to_tensor(raw["pi_I0"], dtype=tf.float64),
        "inventory_maps": inventory_maps_tf(int(dims["I_max"])),
    }


def posterior_bundle_tf() -> dict[str, Any]:
    """
    Build a complete posterior bundle used by posterior/update/model tests.

    Returns:
      - observed TensorFlow inputs
      - posterior_config
      - instantiated StockpilingPosteriorTF
    """
    from ching.stockpiling_posterior import StockpilingPosteriorTF

    observed = observed_inputs_tf()
    config = posterior_config()

    posterior = StockpilingPosteriorTF(
        config=config,
        a_mnjt=observed["a_mnjt"],
        s_mjt=observed["s_mjt"],
        u_mj=observed["u_mj"],
        P_price_mj=observed["P_price_mj"],
        price_vals_mj=observed["price_vals_mj"],
        lambda_mn=observed["lambda_mn"],
        waste_cost=observed["waste_cost"],
        pi_I0=observed["pi_I0"],
        inventory_maps=observed["inventory_maps"],
    )

    return {
        **observed,
        "posterior_config": config,
        "posterior": posterior,
    }


def run_chain_inputs_tf(
    seed: int = 123,
    num_results: int = 8,
    chunk_size: int = 4,
) -> dict[str, Any]:
    """
    Build a validation-ready input bundle matching the refactored run_chain API.
    """
    observed = observed_inputs_tf()
    dims = observed["dims"]

    return {
        "dims": dims,
        "a_mnjt": observed["a_mnjt"],
        "s_mjt": observed["s_mjt"],
        "u_mj": observed["u_mj"],
        "P_price_mj": observed["P_price_mj"],
        "price_vals_mj": observed["price_vals_mj"],
        "lambda_mn": observed["lambda_mn"],
        "waste_cost": observed["waste_cost"],
        "pi_I0": observed["pi_I0"],
        "inventory_maps": observed["inventory_maps"],
        "posterior_config": posterior_config(),
        "stockpiling_config": sampler_config_tf(
            dims=dims,
            num_results=num_results,
            chunk_size=chunk_size,
        ),
        "initial_state": initial_state_tf(dims=dims),
        "seed": seed_tf(seed),
    }
