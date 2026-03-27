"""
Unit tests for lu.shrinkage.lu_shrinkage.

This file targets the refactored shrinkage API:
- LuShrinkageConfig
- LuShrinkageState
- LuHybridKernel
- LuHybridKernelResults
- build_initial_state(...)
- _num_chunks(...)
- _last_state(...)
- _concat_sample_chunks(...)
- run_chain(...)
- summarize_samples(...)

The old estimator-style API (LuShrinkageEstimator, fit(), get_results(), phi)
is no longer part of this module and is not tested here.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

import lu.shrinkage.lu_shrinkage as lu_shrinkage_mod
from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF
from lu.shrinkage.lu_shrinkage import (
    LuHybridKernel,
    LuHybridKernelResults,
    LuShrinkageConfig,
    LuShrinkageState,
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


def _make_posterior_config(n_draws: int, seed: int) -> LuPosteriorConfig:
    """Build a small posterior config for shrinkage tests."""
    return LuPosteriorConfig(
        n_draws=int(n_draws),
        seed=int(seed),
        eps=1e-12,
        beta_p_mean=-1.0,
        beta_p_var=1.0,
        beta_w_mean=0.3,
        beta_w_var=1.0,
        r_mean=0.0,
        r_var=1.0,
        E_bar_mean=0.0,
        E_bar_var=1.0,
        T0_sq=1e-2,
        T1_sq=1.0,
        a_phi=1.0,
        b_phi=1.0,
    )


def _make_shrinkage_config(
    num_results: int = 4,
    num_burnin_steps: int = 2,
    chunk_size: int = 2,
) -> LuShrinkageConfig:
    """Build a small sampler config for shrinkage tests."""
    return LuShrinkageConfig(
        num_results=int(num_results),
        num_burnin_steps=int(num_burnin_steps),
        chunk_size=int(chunk_size),
        k_beta=0.10,
        k_r=0.10,
        k_E_bar=0.10,
        k_njt=0.10,
        pilot_length=2,
        target_low=0.30,
        target_high=0.50,
        max_rounds=2,
        factor=1.25,
    )


def _make_problem() -> dict:
    """Canonical tiny panel used across shrinkage tests."""
    T, J = 2, 3

    pjt = _tf([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]])
    wjt = _tf([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]])
    qjt = _tf([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]])
    q0t = _tf([20.0, 15.0])

    return {
        "T": T,
        "J": J,
        "pjt": pjt,
        "wjt": wjt,
        "qjt": qjt,
        "q0t": q0t,
    }


def _make_posterior() -> LuPosteriorTF:
    """Construct the posterior object used in shrinkage tests."""
    return LuPosteriorTF(_make_posterior_config(n_draws=25, seed=123))


def _make_fake_trace(N: int = 3, T: int = 2, J: int = 3) -> LuShrinkageState:
    """Build a fake retained trace with leading draw dimension N."""
    return LuShrinkageState(
        beta_p=_tf(np.arange(N, dtype=np.float64)),
        beta_w=_tf(np.arange(N, dtype=np.float64) + 10.0),
        r=_tf(np.linspace(-0.2, 0.2, N)),
        E_bar=_tf(np.arange(N * T, dtype=np.float64).reshape(N, T) / 10.0),
        njt=_tf(np.arange(N * T * J, dtype=np.float64).reshape(N, T, J) / 100.0),
        gamma=_tf((np.arange(N * T * J).reshape(N, T, J) % 2).astype(np.float64)),
    )


def _assert_all_finite_tf(*xs: tf.Tensor) -> None:
    """Assert that all supplied tensors contain only finite values."""
    for x in xs:
        x_np = tf.convert_to_tensor(x).numpy()
        if not np.all(np.isfinite(x_np)):
            raise AssertionError("Tensor contains non-finite values.")


def _assert_binary_01_tf(x: tf.Tensor) -> None:
    """Assert that a tensor is finite and takes values only in {0, 1}."""
    x_np = tf.convert_to_tensor(x).numpy()
    if not np.all(np.isfinite(x_np)):
        raise AssertionError("Tensor contains non-finite values.")
    if not np.all((x_np == 0.0) | (x_np == 1.0)):
        raise AssertionError("Tensor contains values outside {0, 1}.")


def _assert_binary_accept_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance indicator in {0.0, 1.0}."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(
            f"Expected scalar acceptance indicator, got shape {x_t.shape}."
        )
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")
    x_val = float(x_t.numpy())
    if x_val not in (0.0, 1.0):
        raise AssertionError(
            f"Expected acceptance indicator in {{0.0, 1.0}}, got {x_val}."
        )


def _assert_accept_rate_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance rate in [0.0, 1.0]."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(f"Expected scalar acceptance rate, got shape {x_t.shape}.")
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")
    x_val = float(x_t.numpy())
    if not np.isfinite(x_val):
        raise AssertionError("Acceptance rate is not finite.")
    if not (0.0 <= x_val <= 1.0):
        raise AssertionError(f"Acceptance rate out of bounds: {x_val} not in [0, 1].")


def _assert_state_shapes(state: LuShrinkageState, T: int, J: int) -> None:
    """Basic shape contract for a single chain state."""
    assert state.beta_p.shape == ()
    assert state.beta_w.shape == ()
    assert state.r.shape == ()
    assert tuple(state.E_bar.shape) == (T,)
    assert tuple(state.njt.shape) == (T, J)
    assert tuple(state.gamma.shape) == (T, J)


def _assert_trace_shapes(samples: LuShrinkageState, N: int, T: int, J: int) -> None:
    """Basic shape contract for a retained state trace."""
    assert tuple(samples.beta_p.shape) == (N,)
    assert tuple(samples.beta_w.shape) == (N,)
    assert tuple(samples.r.shape) == (N,)
    assert tuple(samples.E_bar.shape) == (N, T)
    assert tuple(samples.njt.shape) == (N, T, J)
    assert tuple(samples.gamma.shape) == (N, T, J)


@contextmanager
def _patched_tuning_identity():
    """Patch tune_shrinkage to return the input config unchanged."""

    def _stub_tune_shrinkage(
        posterior,
        qjt,
        q0t,
        pjt,
        wjt,
        initial_state,
        shrinkage_config,
        pilot_length,
        target_low,
        target_high,
        max_rounds,
        factor,
        seed,
    ):
        del posterior, qjt, q0t, pjt, wjt, initial_state
        del pilot_length, target_low, target_high, max_rounds, factor, seed
        return shrinkage_config

    with patch.object(
        lu_shrinkage_mod, "tune_shrinkage", side_effect=_stub_tune_shrinkage
    ):
        yield


# -----------------------------------------------------------------------------
# Small helper tests
# -----------------------------------------------------------------------------
def test_num_chunks_handles_zero_exact_multiple_and_round_up():
    assert _num_chunks(0, 5) == 0
    assert _num_chunks(-3, 5) == 0
    assert _num_chunks(6, 3) == 2
    assert _num_chunks(7, 3) == 3
    assert _num_chunks(1, 8) == 1


def test_last_state_extracts_terminal_draw():
    samples = _make_fake_trace(N=4, T=2, J=3)
    state = _last_state(samples)

    assert isinstance(state, LuShrinkageState)
    _assert_state_shapes(state, T=2, J=3)

    tf.debugging.assert_equal(state.beta_p, samples.beta_p[-1])
    tf.debugging.assert_equal(state.beta_w, samples.beta_w[-1])
    tf.debugging.assert_equal(state.r, samples.r[-1])
    tf.debugging.assert_equal(state.E_bar, samples.E_bar[-1])
    tf.debugging.assert_equal(state.njt, samples.njt[-1])
    tf.debugging.assert_equal(state.gamma, samples.gamma[-1])


def test_concat_sample_chunks_concatenates_draw_dimension():
    chunk_1 = _make_fake_trace(N=2, T=2, J=3)
    chunk_2 = _make_fake_trace(N=3, T=2, J=3)

    out = _concat_sample_chunks([chunk_1, chunk_2])

    assert isinstance(out, LuShrinkageState)
    _assert_trace_shapes(out, N=5, T=2, J=3)

    tf.debugging.assert_equal(out.beta_p[:2], chunk_1.beta_p)
    tf.debugging.assert_equal(out.beta_p[2:], chunk_2.beta_p)
    tf.debugging.assert_equal(out.E_bar[:2], chunk_1.E_bar)
    tf.debugging.assert_equal(out.E_bar[2:], chunk_2.E_bar)
    tf.debugging.assert_equal(out.njt[:2], chunk_1.njt)
    tf.debugging.assert_equal(out.njt[2:], chunk_2.njt)
    tf.debugging.assert_equal(out.gamma[:2], chunk_1.gamma)
    tf.debugging.assert_equal(out.gamma[2:], chunk_2.gamma)


def test_concat_sample_chunks_raises_on_empty_input():
    with pytest.raises(ValueError, match="No retained sample chunks"):
        _concat_sample_chunks([])


# -----------------------------------------------------------------------------
# Initial-state construction
# -----------------------------------------------------------------------------
def test_build_initial_state_defaults():
    problem = _make_problem()
    posterior = _make_posterior()

    state = build_initial_state(pjt=problem["pjt"], posterior=posterior)

    assert isinstance(state, LuShrinkageState)
    _assert_state_shapes(state, T=problem["T"], J=problem["J"])

    tf.debugging.assert_equal(state.beta_p, _tf(0.0))
    tf.debugging.assert_equal(state.beta_w, _tf(0.0))
    tf.debugging.assert_equal(state.r, _tf(0.0))
    tf.debugging.assert_equal(state.E_bar, tf.zeros((problem["T"],), dtype=DTYPE))
    tf.debugging.assert_equal(
        state.njt, tf.zeros((problem["T"], problem["J"]), dtype=DTYPE)
    )
    tf.debugging.assert_equal(
        state.gamma, tf.zeros((problem["T"], problem["J"]), dtype=DTYPE)
    )


# -----------------------------------------------------------------------------
# Hybrid kernel
# -----------------------------------------------------------------------------
def test_hybrid_kernel_bootstrap_results_are_zero():
    problem = _make_problem()
    posterior = _make_posterior()
    config = _make_shrinkage_config()
    initial_state = build_initial_state(problem["pjt"], posterior)

    kernel = LuHybridKernel(
        posterior=posterior,
        qjt=problem["qjt"],
        q0t=problem["q0t"],
        pjt=problem["pjt"],
        wjt=problem["wjt"],
        config=config,
    )

    results = kernel.bootstrap_results(initial_state)

    assert isinstance(results, LuHybridKernelResults)

    for x in [
        results.beta_accept,
        results.r_accept,
        results.E_bar_accept,
        results.njt_accept,
    ]:
        assert x.shape == ()
        assert x.dtype == DTYPE
        tf.debugging.assert_equal(x, _tf(0.0))


def test_hybrid_kernel_one_step_returns_valid_state_and_results():
    problem = _make_problem()
    posterior = _make_posterior()
    config = _make_shrinkage_config()
    current_state = build_initial_state(problem["pjt"], posterior)

    kernel = LuHybridKernel(
        posterior=posterior,
        qjt=problem["qjt"],
        q0t=problem["q0t"],
        pjt=problem["pjt"],
        wjt=problem["wjt"],
        config=config,
    )

    previous_results = kernel.bootstrap_results(current_state)

    new_state, new_results = kernel.one_step(
        current_state=current_state,
        previous_kernel_results=previous_results,
        seed=_seed(1, 2),
    )

    assert isinstance(new_state, LuShrinkageState)
    assert isinstance(new_results, LuHybridKernelResults)

    _assert_state_shapes(new_state, T=problem["T"], J=problem["J"])
    _assert_all_finite_tf(
        new_state.beta_p,
        new_state.beta_w,
        new_state.r,
        new_state.E_bar,
        new_state.njt,
        new_state.gamma,
    )
    _assert_binary_01_tf(new_state.gamma)

    _assert_binary_accept_scalar(new_results.beta_accept)
    _assert_binary_accept_scalar(new_results.r_accept)
    _assert_accept_rate_scalar(new_results.E_bar_accept)
    _assert_accept_rate_scalar(new_results.njt_accept)


def test_hybrid_kernel_one_step_is_deterministic_for_fixed_seed():
    problem = _make_problem()
    posterior = _make_posterior()
    config = _make_shrinkage_config()
    current_state = build_initial_state(problem["pjt"], posterior)

    kernel = LuHybridKernel(
        posterior=posterior,
        qjt=problem["qjt"],
        q0t=problem["q0t"],
        pjt=problem["pjt"],
        wjt=problem["wjt"],
        config=config,
    )

    previous_results = kernel.bootstrap_results(current_state)
    seed = _seed(11, 12)

    out_1 = kernel.one_step(
        current_state=current_state,
        previous_kernel_results=previous_results,
        seed=seed,
    )
    out_2 = kernel.one_step(
        current_state=current_state,
        previous_kernel_results=previous_results,
        seed=seed,
    )

    state_1, results_1 = out_1
    state_2, results_2 = out_2

    tf.debugging.assert_equal(state_1.beta_p, state_2.beta_p)
    tf.debugging.assert_equal(state_1.beta_w, state_2.beta_w)
    tf.debugging.assert_equal(state_1.r, state_2.r)
    tf.debugging.assert_equal(state_1.E_bar, state_2.E_bar)
    tf.debugging.assert_equal(state_1.njt, state_2.njt)
    tf.debugging.assert_equal(state_1.gamma, state_2.gamma)

    tf.debugging.assert_equal(results_1.beta_accept, results_2.beta_accept)
    tf.debugging.assert_equal(results_1.r_accept, results_2.r_accept)
    tf.debugging.assert_equal(results_1.E_bar_accept, results_2.E_bar_accept)
    tf.debugging.assert_equal(results_1.njt_accept, results_2.njt_accept)


# -----------------------------------------------------------------------------
# Public chain entrypoint
# -----------------------------------------------------------------------------
def test_run_chain_raises_on_q0t_length_mismatch():
    problem = _make_problem()
    posterior_config = _make_posterior_config(n_draws=25, seed=123)
    shrinkage_config = _make_shrinkage_config()

    bad_q0t = _tf([20.0])

    with pytest.raises(ValueError, match="q0t must have length T"):
        run_chain(
            pjt=problem["pjt"],
            wjt=problem["wjt"],
            qjt=problem["qjt"],
            q0t=bad_q0t,
            posterior_config=posterior_config,
            shrinkage_config=shrinkage_config,
            seed=_seed(3, 4),
        )


def test_run_chain_raises_on_bad_seed_shape():
    problem = _make_problem()
    posterior_config = _make_posterior_config(n_draws=25, seed=123)
    shrinkage_config = _make_shrinkage_config()

    bad_seed = tf.constant([1, 2, 3], dtype=tf.int32)

    with pytest.raises(ValueError, match="seed must have shape \\(2,\\)"):
        run_chain(
            pjt=problem["pjt"],
            wjt=problem["wjt"],
            qjt=problem["qjt"],
            q0t=problem["q0t"],
            posterior_config=posterior_config,
            shrinkage_config=shrinkage_config,
            seed=bad_seed,
        )


def test_run_chain_returns_retained_trace_with_num_results_draws():
    problem = _make_problem()
    posterior_config = _make_posterior_config(n_draws=25, seed=123)
    shrinkage_config = _make_shrinkage_config(
        num_results=3,
        num_burnin_steps=2,
        chunk_size=2,
    )

    with _patched_tuning_identity():
        samples = run_chain(
            pjt=problem["pjt"],
            wjt=problem["wjt"],
            qjt=problem["qjt"],
            q0t=problem["q0t"],
            posterior_config=posterior_config,
            shrinkage_config=shrinkage_config,
            seed=_seed(5, 6),
        )

    assert isinstance(samples, LuShrinkageState)
    _assert_trace_shapes(samples, N=3, T=problem["T"], J=problem["J"])

    _assert_all_finite_tf(
        samples.beta_p,
        samples.beta_w,
        samples.r,
        samples.E_bar,
        samples.njt,
        samples.gamma,
    )
    _assert_binary_01_tf(samples.gamma)


# -----------------------------------------------------------------------------
# Posterior-summary output
# -----------------------------------------------------------------------------
def test_summarize_samples_schema_shapes_and_identities():
    samples = _make_fake_trace(N=4, T=2, J=3)

    res = summarize_samples(samples)

    expected_keys = {
        "beta_p_hat",
        "beta_w_hat",
        "sigma_hat",
        "int_hat",
        "E_hat",
        "E_bar_hat",
        "njt_hat",
        "gamma_hat",
    }
    assert set(res.keys()) == expected_keys

    assert res["beta_p_hat"].shape == ()
    assert res["beta_w_hat"].shape == ()
    assert res["sigma_hat"].shape == ()
    assert res["int_hat"].shape == ()
    assert tuple(res["E_bar_hat"].shape) == (2,)
    assert tuple(res["njt_hat"].shape) == (2, 3)
    assert tuple(res["gamma_hat"].shape) == (2, 3)
    assert tuple(res["E_hat"].shape) == (2, 3)

    _assert_all_finite_tf(
        res["beta_p_hat"],
        res["beta_w_hat"],
        res["sigma_hat"],
        res["int_hat"],
        res["E_hat"],
        res["E_bar_hat"],
        res["njt_hat"],
        res["gamma_hat"],
    )

    expected_beta_p_hat = tf.reduce_mean(samples.beta_p, axis=0)
    expected_beta_w_hat = tf.reduce_mean(samples.beta_w, axis=0)
    expected_sigma_hat = tf.reduce_mean(tf.exp(samples.r), axis=0)
    expected_E_bar_hat = tf.reduce_mean(samples.E_bar, axis=0)
    expected_njt_hat = tf.reduce_mean(samples.njt, axis=0)
    expected_gamma_hat = tf.reduce_mean(samples.gamma, axis=0)
    expected_int_hat = tf.reduce_mean(expected_E_bar_hat)
    expected_E_hat = expected_E_bar_hat[:, None] + expected_njt_hat

    tf.debugging.assert_near(
        res["beta_p_hat"], expected_beta_p_hat, atol=ATOL, rtol=RTOL
    )
    tf.debugging.assert_near(
        res["beta_w_hat"], expected_beta_w_hat, atol=ATOL, rtol=RTOL
    )
    tf.debugging.assert_near(res["sigma_hat"], expected_sigma_hat, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["E_bar_hat"], expected_E_bar_hat, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["njt_hat"], expected_njt_hat, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["gamma_hat"], expected_gamma_hat, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["int_hat"], expected_int_hat, atol=ATOL, rtol=RTOL)
    tf.debugging.assert_near(res["E_hat"], expected_E_hat, atol=ATOL, rtol=RTOL)
