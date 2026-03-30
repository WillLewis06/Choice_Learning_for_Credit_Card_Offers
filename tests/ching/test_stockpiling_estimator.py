# tests/ching/test_stockpiling_estimator.py
"""Unit tests for `ching.stockpiling_estimator`.

This file targets the refactored Ching sampler API:
- StockpilingConfig
- StockpilingState
- StockpilingKernelResults
- StockpilingRunResult
- StockpilingHybridKernel
- build_initial_state(...) from explicit z-block tensors
- _num_chunks(...)
- _last_state(...)
- _concat_sample_chunks(...)
- run_chain(...) with explicit pi_I0
- summarize_samples(...)

The old estimator-style API (StockpilingEstimator, fit(), predict_probabilities())
is no longer part of this module and is not tested here.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

# Reduce TensorFlow C++ logging before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import ching_conftest as cc  # noqa: E402
import tensorflow as tf  # noqa: E402

import ching.stockpiling_estimator as est_mod  # noqa: E402
from ching.stockpiling_estimator import (  # noqa: E402
    StockpilingHybridKernel,
    StockpilingKernelResults,
    StockpilingRunResult,
    StockpilingState,
    _concat_sample_chunks,
    _last_state,
    _num_chunks,
    build_initial_state,
    run_chain,
    summarize_samples,
)

DTYPE = tf.float64
SEED_DTYPE = tf.int32
ATOL = 1e-12
RTOL = 0.0


def _tf(x) -> tf.Tensor:
    """Create a tf.float64 constant."""
    return tf.constant(x, dtype=DTYPE)


def _seed(a: int, b: int) -> tf.Tensor:
    """Create a stateless seed tensor."""
    return tf.constant([a, b], dtype=SEED_DTYPE)


def _make_fake_trace(N: int = 4, M: int = 2, J: int = 2) -> StockpilingState:
    """Build a fake retained trace with leading draw dimension N."""
    z_beta = _tf(np.linspace(-1.0, 1.0, N))
    z_alpha = _tf(np.arange(N * J, dtype=np.float64).reshape(N, J) / 10.0)
    z_v = _tf(np.arange(N * J, dtype=np.float64).reshape(N, J) / 20.0 - 0.3)
    z_fc = _tf(np.arange(N * J, dtype=np.float64).reshape(N, J) / 30.0 - 0.2)
    z_u_scale = _tf(np.arange(N * M, dtype=np.float64).reshape(N, M) / 40.0 - 0.1)

    return StockpilingState(
        z_beta=z_beta,
        z_alpha=z_alpha,
        z_v=z_v,
        z_fc=z_fc,
        z_u_scale=z_u_scale,
    )


def _assert_all_finite_tf(*xs: tf.Tensor) -> None:
    """Assert that all supplied tensors contain only finite values."""
    for x in xs:
        x_np = tf.convert_to_tensor(x).numpy()
        if not np.all(np.isfinite(x_np)):
            raise AssertionError("Tensor contains non-finite values.")


def _assert_state_shapes(state: StockpilingState, M: int, J: int) -> None:
    """Basic shape contract for a single chain state."""
    assert state.z_beta.shape == ()
    assert tuple(state.z_alpha.shape) == (J,)
    assert tuple(state.z_v.shape) == (J,)
    assert tuple(state.z_fc.shape) == (J,)
    assert tuple(state.z_u_scale.shape) == (M,)


def _assert_trace_shapes(samples: StockpilingState, N: int, M: int, J: int) -> None:
    """Basic shape contract for a retained state trace."""
    assert tuple(samples.z_beta.shape) == (N,)
    assert tuple(samples.z_alpha.shape) == (N, J)
    assert tuple(samples.z_v.shape) == (N, J)
    assert tuple(samples.z_fc.shape) == (N, J)
    assert tuple(samples.z_u_scale.shape) == (N, M)


def _assert_accept_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance diagnostic in [0.0, 1.0]."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(f"Expected scalar acceptance diagnostic, got {x_t.shape}.")
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")
    x_val = float(x_t.numpy())
    if not np.isfinite(x_val):
        raise AssertionError("Acceptance diagnostic is not finite.")
    if not (0.0 <= x_val <= 1.0):
        raise AssertionError(f"Acceptance diagnostic out of bounds: {x_val}.")


# -----------------------------------------------------------------------------
# Small helper tests
# -----------------------------------------------------------------------------
def test_num_chunks_handles_zero_exact_multiple_and_round_up() -> None:
    assert _num_chunks(0, 5) == 0
    assert _num_chunks(-3, 5) == 0
    assert _num_chunks(6, 3) == 2
    assert _num_chunks(7, 3) == 3
    assert _num_chunks(1, 8) == 1


def test_last_state_extracts_terminal_draw() -> None:
    samples = _make_fake_trace(N=4, M=2, J=2)
    state = _last_state(samples)

    assert isinstance(state, StockpilingState)
    _assert_state_shapes(state, M=2, J=2)

    tf.debugging.assert_equal(state.z_beta, samples.z_beta[-1])
    tf.debugging.assert_equal(state.z_alpha, samples.z_alpha[-1])
    tf.debugging.assert_equal(state.z_v, samples.z_v[-1])
    tf.debugging.assert_equal(state.z_fc, samples.z_fc[-1])
    tf.debugging.assert_equal(state.z_u_scale, samples.z_u_scale[-1])


def test_concat_sample_chunks_concatenates_draw_dimension() -> None:
    chunk_1 = _make_fake_trace(N=2, M=2, J=2)
    chunk_2 = _make_fake_trace(N=3, M=2, J=2)

    out = _concat_sample_chunks([chunk_1, chunk_2])

    assert isinstance(out, StockpilingState)
    _assert_trace_shapes(out, N=5, M=2, J=2)

    tf.debugging.assert_equal(out.z_beta[:2], chunk_1.z_beta)
    tf.debugging.assert_equal(out.z_beta[2:], chunk_2.z_beta)
    tf.debugging.assert_equal(out.z_alpha[:2], chunk_1.z_alpha)
    tf.debugging.assert_equal(out.z_alpha[2:], chunk_2.z_alpha)
    tf.debugging.assert_equal(out.z_v[:2], chunk_1.z_v)
    tf.debugging.assert_equal(out.z_v[2:], chunk_2.z_v)
    tf.debugging.assert_equal(out.z_fc[:2], chunk_1.z_fc)
    tf.debugging.assert_equal(out.z_fc[2:], chunk_2.z_fc)
    tf.debugging.assert_equal(out.z_u_scale[:2], chunk_1.z_u_scale)
    tf.debugging.assert_equal(out.z_u_scale[2:], chunk_2.z_u_scale)


def test_concat_sample_chunks_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="No retained sample chunks"):
        _concat_sample_chunks([])


# -----------------------------------------------------------------------------
# Initial-state construction
# -----------------------------------------------------------------------------
def test_build_initial_state_from_explicit_z_blocks() -> None:
    dims = cc.tiny_dims()
    z_blocks = cc.z_blocks_np(dims)

    state = build_initial_state(
        z_beta=tf.convert_to_tensor(z_blocks["z_beta"], dtype=DTYPE),
        z_alpha=tf.convert_to_tensor(z_blocks["z_alpha"], dtype=DTYPE),
        z_v=tf.convert_to_tensor(z_blocks["z_v"], dtype=DTYPE),
        z_fc=tf.convert_to_tensor(z_blocks["z_fc"], dtype=DTYPE),
        z_u_scale=tf.convert_to_tensor(z_blocks["z_u_scale"], dtype=DTYPE),
    )

    assert isinstance(state, StockpilingState)
    _assert_state_shapes(state, M=int(dims["M"]), J=int(dims["J"]))

    tf.debugging.assert_equal(
        state.z_beta,
        tf.convert_to_tensor(z_blocks["z_beta"], dtype=DTYPE),
    )
    tf.debugging.assert_equal(
        state.z_alpha,
        tf.convert_to_tensor(z_blocks["z_alpha"], dtype=DTYPE),
    )
    tf.debugging.assert_equal(
        state.z_v,
        tf.convert_to_tensor(z_blocks["z_v"], dtype=DTYPE),
    )
    tf.debugging.assert_equal(
        state.z_fc,
        tf.convert_to_tensor(z_blocks["z_fc"], dtype=DTYPE),
    )
    tf.debugging.assert_equal(
        state.z_u_scale,
        tf.convert_to_tensor(z_blocks["z_u_scale"], dtype=DTYPE),
    )


# -----------------------------------------------------------------------------
# Hybrid kernel
# -----------------------------------------------------------------------------
def test_hybrid_kernel_bootstrap_results_are_zero() -> None:
    bundle = cc.posterior_bundle_tf()
    posterior = bundle["posterior"]
    dims = bundle["dims"]
    config = cc.sampler_config_tf(dims=dims, num_results=4, chunk_size=2)
    initial_state = cc.initial_state_tf(dims)

    kernel = StockpilingHybridKernel(
        posterior=posterior,
        config=config,
    )

    results = kernel.bootstrap_results(initial_state)

    assert isinstance(results, StockpilingKernelResults)

    for x in [
        results.beta_accept,
        results.alpha_accept,
        results.v_accept,
        results.fc_accept,
        results.u_scale_accept,
    ]:
        assert x.shape == ()
        assert x.dtype == DTYPE
        tf.debugging.assert_equal(x, _tf(0.0))


def test_hybrid_kernel_one_step_returns_valid_state_and_results() -> None:
    bundle = cc.posterior_bundle_tf()
    posterior = bundle["posterior"]
    dims = bundle["dims"]
    current_state = cc.initial_state_tf(dims)
    config = cc.sampler_config_tf(dims=dims, num_results=4, chunk_size=2)

    kernel = StockpilingHybridKernel(
        posterior=posterior,
        config=config,
    )
    previous_results = kernel.bootstrap_results(current_state)

    def fake_beta_one_step(
        posterior,
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_u_scale,
        k_beta,
        seed,
    ):
        del posterior, z_alpha, z_v, z_fc, z_u_scale, k_beta, seed
        return z_beta + tf.constant(0.1, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE)

    def fake_alpha_one_step(
        posterior,
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_u_scale,
        k_alpha,
        seed,
    ):
        del posterior, z_beta, z_v, z_fc, z_u_scale, k_alpha, seed
        return z_alpha + tf.constant(0.2, dtype=DTYPE), tf.constant(0.8, dtype=DTYPE)

    def fake_v_one_step(
        posterior,
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_u_scale,
        k_v,
        seed,
    ):
        del posterior, z_beta, z_alpha, z_fc, z_u_scale, k_v, seed
        return z_v + tf.constant(0.3, dtype=DTYPE), tf.constant(0.6, dtype=DTYPE)

    def fake_fc_one_step(
        posterior,
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_u_scale,
        k_fc,
        seed,
    ):
        del posterior, z_beta, z_alpha, z_v, z_u_scale, k_fc, seed
        return z_fc + tf.constant(0.4, dtype=DTYPE), tf.constant(0.4, dtype=DTYPE)

    def fake_u_scale_one_step(
        posterior,
        z_beta,
        z_alpha,
        z_v,
        z_fc,
        z_u_scale,
        k_u_scale,
        seed,
    ):
        del posterior, z_beta, z_alpha, z_v, z_fc, k_u_scale, seed
        return z_u_scale + tf.constant(0.5, dtype=DTYPE), tf.constant(0.2, dtype=DTYPE)

    with (
        patch.object(est_mod, "beta_one_step", side_effect=fake_beta_one_step),
        patch.object(est_mod, "alpha_one_step", side_effect=fake_alpha_one_step),
        patch.object(est_mod, "v_one_step", side_effect=fake_v_one_step),
        patch.object(est_mod, "fc_one_step", side_effect=fake_fc_one_step),
        patch.object(est_mod, "u_scale_one_step", side_effect=fake_u_scale_one_step),
    ):
        new_state, new_results = kernel.one_step(
            current_state=current_state,
            previous_kernel_results=previous_results,
            seed=_seed(1, 2),
        )

    assert isinstance(new_state, StockpilingState)
    assert isinstance(new_results, StockpilingKernelResults)

    _assert_state_shapes(new_state, M=int(dims["M"]), J=int(dims["J"]))
    _assert_all_finite_tf(
        new_state.z_beta,
        new_state.z_alpha,
        new_state.z_v,
        new_state.z_fc,
        new_state.z_u_scale,
    )

    tf.debugging.assert_near(new_state.z_beta, _tf(0.1), atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(
        new_state.z_alpha,
        tf.fill((dims["J"],), tf.constant(0.2, dtype=DTYPE)),
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        new_state.z_v,
        tf.fill((dims["J"],), tf.constant(0.3, dtype=DTYPE)),
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        new_state.z_fc,
        tf.fill((dims["J"],), tf.constant(0.4, dtype=DTYPE)),
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        new_state.z_u_scale,
        tf.fill((dims["M"],), tf.constant(0.5, dtype=DTYPE)),
        atol=ATOL,
        rtol=RTOL,
    )

    _assert_accept_scalar(new_results.beta_accept)
    _assert_accept_scalar(new_results.alpha_accept)
    _assert_accept_scalar(new_results.v_accept)
    _assert_accept_scalar(new_results.fc_accept)
    _assert_accept_scalar(new_results.u_scale_accept)

    tf.debugging.assert_equal(new_results.beta_accept, _tf(1.0))
    tf.debugging.assert_equal(new_results.alpha_accept, _tf(0.8))
    tf.debugging.assert_equal(new_results.v_accept, _tf(0.6))
    tf.debugging.assert_equal(new_results.fc_accept, _tf(0.4))
    tf.debugging.assert_equal(new_results.u_scale_accept, _tf(0.2))


# -----------------------------------------------------------------------------
# Public chain entrypoint
# -----------------------------------------------------------------------------
def test_run_chain_raises_on_nonpositive_num_results() -> None:
    inputs = cc.run_chain_inputs_tf(seed=123, num_results=2, chunk_size=1)
    dims = inputs["dims"]

    bad_config = cc.sampler_config_tf(dims=dims, num_results=0, chunk_size=1)

    with pytest.raises(ValueError, match="num_results"):
        run_chain(
            a_mnjt=inputs["a_mnjt"],
            s_mjt=inputs["s_mjt"],
            u_mj=inputs["u_mj"],
            P_price_mj=inputs["P_price_mj"],
            price_vals_mj=inputs["price_vals_mj"],
            lambda_mn=inputs["lambda_mn"],
            waste_cost=inputs["waste_cost"],
            pi_I0=inputs["pi_I0"],
            inventory_maps=inputs["inventory_maps"],
            posterior_config=inputs["posterior_config"],
            stockpiling_config=bad_config,
            initial_state=inputs["initial_state"],
            seed=inputs["seed"],
        )


def test_run_chain_raises_on_bad_seed_shape() -> None:
    inputs = cc.run_chain_inputs_tf(seed=123, num_results=2, chunk_size=1)
    bad_seed = tf.constant([1, 2, 3], dtype=SEED_DTYPE)

    with pytest.raises(ValueError, match="seed"):
        run_chain(
            a_mnjt=inputs["a_mnjt"],
            s_mjt=inputs["s_mjt"],
            u_mj=inputs["u_mj"],
            P_price_mj=inputs["P_price_mj"],
            price_vals_mj=inputs["price_vals_mj"],
            lambda_mn=inputs["lambda_mn"],
            waste_cost=inputs["waste_cost"],
            pi_I0=inputs["pi_I0"],
            inventory_maps=inputs["inventory_maps"],
            posterior_config=inputs["posterior_config"],
            stockpiling_config=inputs["stockpiling_config"],
            initial_state=inputs["initial_state"],
            seed=bad_seed,
        )


def test_run_chain_returns_retained_trace_with_num_results_draws() -> None:
    inputs = cc.run_chain_inputs_tf(seed=123, num_results=2, chunk_size=1)
    dims = inputs["dims"]

    out = run_chain(
        a_mnjt=inputs["a_mnjt"],
        s_mjt=inputs["s_mjt"],
        u_mj=inputs["u_mj"],
        P_price_mj=inputs["P_price_mj"],
        price_vals_mj=inputs["price_vals_mj"],
        lambda_mn=inputs["lambda_mn"],
        waste_cost=inputs["waste_cost"],
        pi_I0=inputs["pi_I0"],
        inventory_maps=inputs["inventory_maps"],
        posterior_config=inputs["posterior_config"],
        stockpiling_config=inputs["stockpiling_config"],
        initial_state=inputs["initial_state"],
        seed=inputs["seed"],
    )

    assert isinstance(out, StockpilingRunResult)
    _assert_trace_shapes(
        out.samples,
        N=2,
        M=int(dims["M"]),
        J=int(dims["J"]),
    )

    _assert_all_finite_tf(
        out.samples.z_beta,
        out.samples.z_alpha,
        out.samples.z_v,
        out.samples.z_fc,
        out.samples.z_u_scale,
    )

    assert len(out.chunk_summaries) == _num_chunks(2, 1)
    assert out.mcmc_summary["n_saved"] == 2
    assert out.mcmc_summary["num_chunks"] == _num_chunks(2, 1)

    expected_summary_keys = {
        "n_saved",
        "beta_accept",
        "alpha_accept",
        "v_accept",
        "fc_accept",
        "u_scale_accept",
        "num_chunks",
        "joint_logpost_last",
    }
    assert set(out.mcmc_summary.keys()) == expected_summary_keys


# -----------------------------------------------------------------------------
# Posterior-summary output
# -----------------------------------------------------------------------------
def test_summarize_samples_schema_shapes_and_constrained_means() -> None:
    samples = _make_fake_trace(N=4, M=2, J=2)

    res = summarize_samples(samples)

    expected_keys = {"beta", "alpha", "v", "fc", "u_scale"}
    assert set(res.keys()) == expected_keys

    assert res["beta"].shape == ()
    assert tuple(res["alpha"].shape) == (2,)
    assert tuple(res["v"].shape) == (2,)
    assert tuple(res["fc"].shape) == (2,)
    assert tuple(res["u_scale"].shape) == (2,)

    _assert_all_finite_tf(
        res["beta"],
        res["alpha"],
        res["v"],
        res["fc"],
        res["u_scale"],
    )

    expected_beta = tf.reduce_mean(tf.sigmoid(samples.z_beta), axis=0)
    expected_alpha = tf.reduce_mean(tf.exp(samples.z_alpha), axis=0)
    expected_v = tf.reduce_mean(tf.exp(samples.z_v), axis=0)
    expected_fc = tf.reduce_mean(tf.exp(samples.z_fc), axis=0)
    expected_u_scale = tf.reduce_mean(tf.exp(samples.z_u_scale), axis=0)

    tf.debugging.assert_near(res["beta"], expected_beta, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["alpha"], expected_alpha, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["v"], expected_v, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["fc"], expected_fc, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(
        res["u_scale"],
        expected_u_scale,
        atol=ATOL,
        rtol=RTOL,
    )
