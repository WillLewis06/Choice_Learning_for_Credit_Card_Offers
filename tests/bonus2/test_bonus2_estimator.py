# tests/bonus2/test_bonus2_estimator.py
"""
Unit tests for `bonus2.bonus2_estimator`.

This file targets the refactored estimator API:
- Bonus2SamplerConfig
- Bonus2State
- Bonus2HybridKernel
- Bonus2HybridKernelResults
- build_initial_state(...)
- _num_chunks(...)
- _last_state(...)
- _concat_sample_chunks(...)
- run_chain(...)
- summarize_samples(...)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import tensorflow as tf

import bonus2_conftest as bc
from bonus2 import bonus2_diagnostics as diagnostics
from bonus2 import bonus2_posterior as posterior_lib
from bonus2.bonus2_estimator import (
    Bonus2HybridKernel,
    Bonus2HybridKernelResults,
    Bonus2SamplerConfig,
    Bonus2State,
    _concat_sample_chunks,
    _last_state,
    _num_chunks,
    build_initial_state,
    run_chain,
    summarize_samples,
)

DTYPE = tf.float64
ATOL = 1e-12
RTOL = 0.0


def _make_panel(
    dims: dict[str, int] | None = None,
    hyper: dict[str, float | int] | None = None,
) -> dict[str, object]:
    """Build a small valid panel for estimator tests."""
    d = bc.tiny_dims() if dims is None else dict(dims)
    h = bc.tiny_hyperparams() if hyper is None else dict(hyper)

    return bc.panel_np(
        dims=d,
        hyper=h,
        y_pattern="alternating",
        neighbor_pattern="ring",
        weekend_pattern="0101",
    )


def _make_posterior(
    panel: dict[str, object],
    config_overrides: dict[str, float] | None = None,
) -> posterior_lib.Bonus2PosteriorTF:
    """Construct the posterior object used in estimator tests."""
    config = bc.posterior_config(overrides=config_overrides)
    inputs = bc.posterior_inputs_tf(panel)
    return posterior_lib.Bonus2PosteriorTF(config=config, inputs=inputs)


def _make_sampler_config(
    overrides: dict[str, float | int] | None = None,
) -> Bonus2SamplerConfig:
    """Construct a small sampler config."""
    return bc.sampler_config(overrides=overrides)


def _make_fake_trace(
    N: int = 3,
    dims: dict[str, int] | None = None,
) -> Bonus2State:
    """Build a fake retained trace with leading draw dimension N."""
    d = bc.tiny_dims() if dims is None else dict(dims)
    M, J, K = int(d["M"]), int(d["J"]), int(d["K"])

    return Bonus2State(
        z_beta_intercept_j=tf.constant(
            np.arange(N * J, dtype=np.float64).reshape(N, J) / 10.0,
            dtype=DTYPE,
        ),
        z_beta_habit_j=tf.constant(
            np.arange(N * J, dtype=np.float64).reshape(N, J) / 20.0,
            dtype=DTYPE,
        ),
        z_beta_peer_j=tf.constant(
            np.arange(N * J, dtype=np.float64).reshape(N, J) / 30.0,
            dtype=DTYPE,
        ),
        z_beta_weekend_jw=tf.constant(
            np.arange(N * J * 2, dtype=np.float64).reshape(N, J, 2) / 40.0,
            dtype=DTYPE,
        ),
        z_a_m=tf.constant(
            np.arange(N * M * K, dtype=np.float64).reshape(N, M, K) / 50.0,
            dtype=DTYPE,
        ),
        z_b_m=tf.constant(
            np.arange(N * M * K, dtype=np.float64).reshape(N, M, K) / 60.0,
            dtype=DTYPE,
        ),
    )


def _assert_all_finite_tf(*xs: tf.Tensor) -> None:
    """Assert that all supplied tensors contain only finite values."""
    for x in xs:
        x_np = tf.convert_to_tensor(x).numpy()
        if not np.all(np.isfinite(x_np)):
            raise AssertionError("Tensor contains non-finite values.")


def _assert_binary_accept_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance indicator in {0.0, 1.0}."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(
            f"Expected scalar acceptance indicator, got shape {x_t.shape}.",
        )
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")
    x_val = float(x_t.numpy())
    if x_val not in (0.0, 1.0):
        raise AssertionError(
            f"Expected acceptance indicator in {{0.0, 1.0}}, got {x_val}.",
        )


def _assert_state_shapes(state: Bonus2State, M: int, J: int, K: int) -> None:
    """Basic shape contract for a single chain state."""
    assert tuple(state.z_beta_intercept_j.shape) == (J,)
    assert tuple(state.z_beta_habit_j.shape) == (J,)
    assert tuple(state.z_beta_peer_j.shape) == (J,)
    assert tuple(state.z_beta_weekend_jw.shape) == (J, 2)
    assert tuple(state.z_a_m.shape) == (M, K)
    assert tuple(state.z_b_m.shape) == (M, K)


def _assert_trace_shapes(
    samples: Bonus2State,
    N: int,
    M: int,
    J: int,
    K: int,
) -> None:
    """Basic shape contract for a retained state trace."""
    assert tuple(samples.z_beta_intercept_j.shape) == (N, J)
    assert tuple(samples.z_beta_habit_j.shape) == (N, J)
    assert tuple(samples.z_beta_peer_j.shape) == (N, J)
    assert tuple(samples.z_beta_weekend_jw.shape) == (N, J, 2)
    assert tuple(samples.z_a_m.shape) == (N, M, K)
    assert tuple(samples.z_b_m.shape) == (N, M, K)


def _quiet_report_chunk_progress(
    trace: diagnostics.Bonus2ChunkTrace,
    chunk_idx: int,
    total_chunks: int | None = None,
) -> diagnostics.Bonus2ChunkSummary:
    """Return the chunk summary without printing."""
    return diagnostics.summarize_chunk_trace(
        trace=trace,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
    )


def _quiet_report_run_summary(
    summaries,
) -> None:
    """Suppress final run-summary printing in tests."""
    del summaries
    return None


def test_num_chunks_handles_zero_exact_multiple_and_round_up() -> None:
    assert _num_chunks(0, 5) == 0
    assert _num_chunks(6, 3) == 2
    assert _num_chunks(7, 3) == 3
    assert _num_chunks(1, 8) == 1


def test_last_state_extracts_terminal_draw() -> None:
    dims = bc.tiny_dims()
    samples = _make_fake_trace(N=4, dims=dims)

    state = _last_state(samples)

    assert isinstance(state, Bonus2State)
    _assert_state_shapes(
        state=state,
        M=int(dims["M"]),
        J=int(dims["J"]),
        K=int(dims["K"]),
    )

    tf.debugging.assert_equal(state.z_beta_intercept_j, samples.z_beta_intercept_j[-1])
    tf.debugging.assert_equal(state.z_beta_habit_j, samples.z_beta_habit_j[-1])
    tf.debugging.assert_equal(state.z_beta_peer_j, samples.z_beta_peer_j[-1])
    tf.debugging.assert_equal(state.z_beta_weekend_jw, samples.z_beta_weekend_jw[-1])
    tf.debugging.assert_equal(state.z_a_m, samples.z_a_m[-1])
    tf.debugging.assert_equal(state.z_b_m, samples.z_b_m[-1])


def test_concat_sample_chunks_concatenates_draw_dimension() -> None:
    dims = bc.tiny_dims()
    M, J, K = int(dims["M"]), int(dims["J"]), int(dims["K"])

    chunk_1 = _make_fake_trace(N=2, dims=dims)
    chunk_2 = _make_fake_trace(N=3, dims=dims)

    out = _concat_sample_chunks([chunk_1, chunk_2])

    assert isinstance(out, Bonus2State)
    _assert_trace_shapes(out, N=5, M=M, J=J, K=K)

    tf.debugging.assert_equal(out.z_beta_intercept_j[:2], chunk_1.z_beta_intercept_j)
    tf.debugging.assert_equal(out.z_beta_intercept_j[2:], chunk_2.z_beta_intercept_j)

    tf.debugging.assert_equal(out.z_beta_habit_j[:2], chunk_1.z_beta_habit_j)
    tf.debugging.assert_equal(out.z_beta_habit_j[2:], chunk_2.z_beta_habit_j)

    tf.debugging.assert_equal(out.z_beta_peer_j[:2], chunk_1.z_beta_peer_j)
    tf.debugging.assert_equal(out.z_beta_peer_j[2:], chunk_2.z_beta_peer_j)

    tf.debugging.assert_equal(out.z_beta_weekend_jw[:2], chunk_1.z_beta_weekend_jw)
    tf.debugging.assert_equal(out.z_beta_weekend_jw[2:], chunk_2.z_beta_weekend_jw)

    tf.debugging.assert_equal(out.z_a_m[:2], chunk_1.z_a_m)
    tf.debugging.assert_equal(out.z_a_m[2:], chunk_2.z_a_m)

    tf.debugging.assert_equal(out.z_b_m[:2], chunk_1.z_b_m)
    tf.debugging.assert_equal(out.z_b_m[2:], chunk_2.z_b_m)


def test_build_initial_state_defaults() -> None:
    dims = bc.tiny_dims()
    M, J, K = int(dims["M"]), int(dims["J"]), int(dims["K"])

    state = build_initial_state(
        num_markets=M,
        num_products=J,
        num_harmonics=K,
    )

    assert isinstance(state, Bonus2State)
    _assert_state_shapes(state=state, M=M, J=J, K=K)

    tf.debugging.assert_equal(state.z_beta_intercept_j, tf.zeros((J,), dtype=DTYPE))
    tf.debugging.assert_equal(state.z_beta_habit_j, tf.zeros((J,), dtype=DTYPE))
    tf.debugging.assert_equal(state.z_beta_peer_j, tf.zeros((J,), dtype=DTYPE))
    tf.debugging.assert_equal(state.z_beta_weekend_jw, tf.zeros((J, 2), dtype=DTYPE))
    tf.debugging.assert_equal(state.z_a_m, tf.zeros((M, K), dtype=DTYPE))
    tf.debugging.assert_equal(state.z_b_m, tf.zeros((M, K), dtype=DTYPE))


def test_hybrid_kernel_bootstrap_results_are_zero() -> None:
    dims = bc.tiny_dims()
    panel = _make_panel(dims=dims)
    posterior = _make_posterior(panel)
    config = _make_sampler_config()

    kernel = Bonus2HybridKernel(
        posterior=posterior,
        config=config,
    )
    initial_state = build_initial_state(
        num_markets=int(dims["M"]),
        num_products=int(dims["J"]),
        num_harmonics=int(dims["K"]),
    )

    results = kernel.bootstrap_results(initial_state)

    assert isinstance(results, Bonus2HybridKernelResults)

    for x in [
        results.beta_intercept_accept,
        results.beta_habit_accept,
        results.beta_peer_accept,
        results.beta_weekend_accept,
        results.a_accept,
        results.b_accept,
    ]:
        assert x.shape == ()
        assert x.dtype == DTYPE
        tf.debugging.assert_equal(x, tf.constant(0.0, dtype=DTYPE))


def test_hybrid_kernel_one_step_returns_valid_state_and_results() -> None:
    dims = bc.tiny_dims()
    panel = _make_panel(dims=dims)
    posterior = _make_posterior(panel)
    config = _make_sampler_config()

    kernel = Bonus2HybridKernel(
        posterior=posterior,
        config=config,
    )
    current_state = build_initial_state(
        num_markets=int(dims["M"]),
        num_products=int(dims["J"]),
        num_harmonics=int(dims["K"]),
    )
    previous_results = kernel.bootstrap_results(current_state)

    new_state, new_results = kernel.one_step(
        current_state=current_state,
        previous_kernel_results=previous_results,
        seed=tf.constant([1, 2], dtype=tf.int32),
    )

    assert isinstance(new_state, Bonus2State)
    assert isinstance(new_results, Bonus2HybridKernelResults)

    _assert_state_shapes(
        state=new_state,
        M=int(dims["M"]),
        J=int(dims["J"]),
        K=int(dims["K"]),
    )
    _assert_all_finite_tf(
        new_state.z_beta_intercept_j,
        new_state.z_beta_habit_j,
        new_state.z_beta_peer_j,
        new_state.z_beta_weekend_jw,
        new_state.z_a_m,
        new_state.z_b_m,
    )

    _assert_binary_accept_scalar(new_results.beta_intercept_accept)
    _assert_binary_accept_scalar(new_results.beta_habit_accept)
    _assert_binary_accept_scalar(new_results.beta_peer_accept)
    _assert_binary_accept_scalar(new_results.beta_weekend_accept)
    _assert_binary_accept_scalar(new_results.a_accept)
    _assert_binary_accept_scalar(new_results.b_accept)


def test_run_chain_returns_retained_trace_and_chunk_summaries() -> None:
    dims = bc.tiny_dims()
    panel = _make_panel(dims=dims)
    posterior_config = bc.posterior_config()
    sampler_config = _make_sampler_config(
        overrides={
            "num_results": 3,
            "num_burnin_steps": 2,
            "chunk_size": 2,
        },
    )

    with (
        patch(
            "bonus2.bonus2_estimator.diagnostics.report_chunk_progress",
            new=_quiet_report_chunk_progress,
        ),
        patch(
            "bonus2.bonus2_estimator.diagnostics.report_run_summary",
            new=_quiet_report_run_summary,
        ),
    ):
        samples, summaries = run_chain(
            y_mit=tf.convert_to_tensor(panel["y_mit"], dtype=tf.int32),
            delta_mj=tf.convert_to_tensor(panel["delta_mj"], dtype=tf.float64),
            is_weekend_t=tf.convert_to_tensor(panel["is_weekend_t"], dtype=tf.int32),
            season_sin_kt=tf.convert_to_tensor(
                panel["season_sin_kt"],
                dtype=tf.float64,
            ),
            season_cos_kt=tf.convert_to_tensor(
                panel["season_cos_kt"],
                dtype=tf.float64,
            ),
            neighbors_m=panel["neighbors_m"],
            lookback=int(panel["lookback"]),
            decay=float(panel["decay"]),
            posterior_config=posterior_config,
            sampler_config=sampler_config,
        )

    assert isinstance(samples, Bonus2State)
    _assert_trace_shapes(
        samples=samples,
        N=int(sampler_config.num_results),
        M=int(dims["M"]),
        J=int(dims["J"]),
        K=int(dims["K"]),
    )
    _assert_all_finite_tf(
        samples.z_beta_intercept_j,
        samples.z_beta_habit_j,
        samples.z_beta_peer_j,
        samples.z_beta_weekend_jw,
        samples.z_a_m,
        samples.z_b_m,
    )

    assert isinstance(summaries, list)
    assert len(summaries) == _num_chunks(
        total_steps=int(sampler_config.num_results),
        chunk_size=int(sampler_config.chunk_size),
    )
    assert all(isinstance(s, diagnostics.Bonus2ChunkSummary) for s in summaries)


def test_summarize_samples_schema_shapes_and_identities() -> None:
    dims = bc.tiny_dims()
    M, J, K = int(dims["M"]), int(dims["J"]), int(dims["K"])
    samples = _make_fake_trace(N=4, dims=dims)

    res = summarize_samples(samples)

    expected_keys = {
        "beta_intercept_j",
        "beta_habit_j",
        "beta_peer_j",
        "beta_weekend_jw",
        "a_m",
        "b_m",
    }
    assert set(res.keys()) == expected_keys

    assert tuple(res["beta_intercept_j"].shape) == (J,)
    assert tuple(res["beta_habit_j"].shape) == (J,)
    assert tuple(res["beta_peer_j"].shape) == (J,)
    assert tuple(res["beta_weekend_jw"].shape) == (J, 2)
    assert tuple(res["a_m"].shape) == (M, K)
    assert tuple(res["b_m"].shape) == (M, K)

    _assert_all_finite_tf(
        res["beta_intercept_j"],
        res["beta_habit_j"],
        res["beta_peer_j"],
        res["beta_weekend_jw"],
        res["a_m"],
        res["b_m"],
    )

    expected_beta_intercept = tf.reduce_mean(samples.z_beta_intercept_j, axis=0)
    expected_beta_habit = tf.reduce_mean(samples.z_beta_habit_j, axis=0)
    expected_beta_peer = tf.reduce_mean(samples.z_beta_peer_j, axis=0)
    expected_beta_weekend = tf.reduce_mean(samples.z_beta_weekend_jw, axis=0)
    expected_a = tf.reduce_mean(samples.z_a_m, axis=0)
    expected_b = tf.reduce_mean(samples.z_b_m, axis=0)

    tf.debugging.assert_near(
        res["beta_intercept_j"],
        expected_beta_intercept,
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        res["beta_habit_j"],
        expected_beta_habit,
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        res["beta_peer_j"],
        expected_beta_peer,
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        res["beta_weekend_jw"],
        expected_beta_weekend,
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        res["a_m"],
        expected_a,
        atol=ATOL,
        rtol=RTOL,
    )
    tf.debugging.assert_near(
        res["b_m"],
        expected_b,
        atol=ATOL,
        rtol=RTOL,
    )
