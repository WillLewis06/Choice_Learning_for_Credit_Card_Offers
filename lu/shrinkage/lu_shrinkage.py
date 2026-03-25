"""Run the Lu shrinkage MCMC chain and summarize retained samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from lu.lu_gibbs import gibbs_gamma
from lu.shrinkage.lu_diagnostics import (
    LuChunkSummary,
    LuChunkTrace,
    report_chunk_progress,
    report_run_summary,
)
from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF
from lu.shrinkage.lu_tuning import tune_shrinkage
from lu.shrinkage.lu_updates import (
    E_bar_one_step,
    beta_one_step,
    njt_one_step,
    r_one_step,
)
from lu.shrinkage.lu_validate_input import run_chain_validate_input


@dataclass(frozen=True)
class LuShrinkageConfig:
    """Store sampler controls, proposal scales, and tuning parameters."""

    num_results: int
    num_burnin_steps: int
    chunk_size: int
    k_beta: float
    k_r: float
    k_E_bar: float
    k_njt: float
    pilot_length: int
    target_low: float
    target_high: float
    max_rounds: int
    factor: float


class LuShrinkageState(NamedTuple):
    """Store the current state of the Lu shrinkage chain."""

    beta_p: tf.Tensor
    beta_w: tf.Tensor
    r: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    gamma: tf.Tensor


class LuHybridKernelResults(NamedTuple):
    """Store the acceptance diagnostics for one hybrid transition step."""

    beta_accept: tf.Tensor
    r_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


def _num_chunks(n_steps: int, chunk_size: int) -> int:
    """Return the number of chunks needed for a given step count."""

    if n_steps <= 0:
        return 0
    return (n_steps + chunk_size - 1) // chunk_size


def _last_state(samples: LuShrinkageState) -> LuShrinkageState:
    """Extract the terminal state from a chunk of sampled states."""

    return LuShrinkageState(
        beta_p=samples.beta_p[-1],
        beta_w=samples.beta_w[-1],
        r=samples.r[-1],
        E_bar=samples.E_bar[-1],
        njt=samples.njt[-1],
        gamma=samples.gamma[-1],
    )


def _concat_sample_chunks(chunks: list[LuShrinkageState]) -> LuShrinkageState:
    """Concatenate retained sample chunks into one state trace."""

    if len(chunks) == 0:
        raise ValueError("No retained sample chunks were produced.")

    # Concatenate retained chunks along the draw dimension.
    return LuShrinkageState(
        beta_p=tf.concat([chunk.beta_p for chunk in chunks], axis=0),
        beta_w=tf.concat([chunk.beta_w for chunk in chunks], axis=0),
        r=tf.concat([chunk.r for chunk in chunks], axis=0),
        E_bar=tf.concat([chunk.E_bar for chunk in chunks], axis=0),
        njt=tf.concat([chunk.njt for chunk in chunks], axis=0),
        gamma=tf.concat([chunk.gamma for chunk in chunks], axis=0),
    )


def _raw_trace_to_dataclass(raw_trace: dict[str, tf.Tensor]) -> LuChunkTrace:
    """Convert the raw trace dictionary into a LuChunkTrace."""

    return LuChunkTrace(
        beta_p=raw_trace["beta_p"],
        beta_w=raw_trace["beta_w"],
        sigma=raw_trace["sigma"],
        mean_E_bar=raw_trace["mean_E_bar"],
        norm_E_bar=raw_trace["norm_E_bar"],
        norm_njt=raw_trace["norm_njt"],
        gamma_active_share=raw_trace["gamma_active_share"],
        joint_logpost=raw_trace["joint_logpost"],
        beta_accept=raw_trace["beta_accept"],
        r_accept=raw_trace["r_accept"],
        E_bar_accept=raw_trace["E_bar_accept"],
        njt_accept=raw_trace["njt_accept"],
    )


def build_initial_state(
    pjt: tf.Tensor,
    posterior: LuPosteriorTF,
) -> LuShrinkageState:
    """Construct the default initial state for the shrinkage chain."""

    T = tf.shape(pjt)[0]
    J = tf.shape(pjt)[1]

    # Start from zero coefficients, zero product shocks, zero inclusions, and a flat market effect.
    return LuShrinkageState(
        beta_p=tf.constant(0.0, dtype=tf.float64),
        beta_w=tf.constant(0.0, dtype=tf.float64),
        r=tf.constant(0.0, dtype=tf.float64),
        E_bar=tf.fill([T], posterior.E_bar_mean),
        njt=tf.zeros([T, J], dtype=tf.float64),
        gamma=tf.zeros([T, J], dtype=tf.float64),
    )


class LuHybridKernel(tfp.mcmc.TransitionKernel):
    """Implement one hybrid transition of the Lu shrinkage sampler."""

    def __init__(
        self,
        posterior: LuPosteriorTF,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        config: LuShrinkageConfig,
    ):
        """Store the posterior object, observed data, and proposal scales."""

        self._posterior = posterior
        self._qjt = qjt
        self._q0t = q0t
        self._pjt = pjt
        self._wjt = wjt
        self._config = config

        # Cache the proposal scales used by the continuous update blocks.
        self._k_beta = tf.constant(config.k_beta, dtype=tf.float64)
        self._k_r = tf.constant(config.k_r, dtype=tf.float64)
        self._k_E_bar = tf.constant(config.k_E_bar, dtype=tf.float64)
        self._k_njt = tf.constant(config.k_njt, dtype=tf.float64)

        self._parameters = {
            "posterior": posterior,
            "qjt": qjt,
            "q0t": q0t,
            "pjt": pjt,
            "wjt": wjt,
            "config": config,
        }

    @property
    def parameters(self):
        """Return the kernel parameters required by the TFP interface."""

        return self._parameters

    @property
    def is_calibrated(self) -> bool:
        """Return whether the kernel is treated as calibrated."""

        return True

    def bootstrap_results(
        self,
        current_state: LuShrinkageState,
    ) -> LuHybridKernelResults:
        """Initialize the acceptance diagnostics for the first step."""

        del current_state
        zero = tf.constant(0.0, dtype=tf.float64)

        # Start all acceptance summaries at zero.
        return LuHybridKernelResults(
            beta_accept=zero,
            r_accept=zero,
            E_bar_accept=zero,
            njt_accept=zero,
        )

    def one_step(
        self,
        current_state: LuShrinkageState,
        previous_kernel_results: LuHybridKernelResults,
        seed: tf.Tensor,
    ) -> tuple[LuShrinkageState, LuHybridKernelResults]:
        """Apply one full hybrid transition of the shrinkage chain."""

        del previous_kernel_results

        # Allocate one seed to each sub-update in the hybrid step.
        seeds = tf.random.experimental.stateless_split(seed, num=5)

        # Update the joint beta block conditional on the current remaining state.
        beta_p_new, beta_w_new, beta_accept = beta_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            pjt=self._pjt,
            wjt=self._wjt,
            beta_p=current_state.beta_p,
            beta_w=current_state.beta_w,
            r=current_state.r,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_beta=self._k_beta,
            seed=seeds[0],
        )

        # Update the random-coefficient scale given the new beta block.
        r_new, r_accept = r_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            pjt=self._pjt,
            wjt=self._wjt,
            beta_p=beta_p_new,
            beta_w=beta_w_new,
            r=current_state.r,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_r=self._k_r,
            seed=seeds[1],
        )

        # Sweep across market effects given the updated global parameters.
        E_bar_new, E_bar_accept = E_bar_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            pjt=self._pjt,
            wjt=self._wjt,
            beta_p=beta_p_new,
            beta_w=beta_w_new,
            r=r_new,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_E_bar=self._k_E_bar,
            seed=seeds[2],
        )

        # Sweep across market-product shocks conditional on the current inclusion indicators.
        njt_new, njt_accept = njt_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            pjt=self._pjt,
            wjt=self._wjt,
            beta_p=beta_p_new,
            beta_w=beta_w_new,
            r=r_new,
            E_bar=E_bar_new,
            njt=current_state.njt,
            gamma=current_state.gamma,
            k_njt=self._k_njt,
            seed=seeds[3],
        )

        # Run the Gibbs sweep for the inclusion indicators conditional on the updated shocks.
        gamma_new = gibbs_gamma(
            njt=njt_new,
            gamma=current_state.gamma,
            a_phi=self._posterior.a_phi,
            b_phi=self._posterior.b_phi,
            T0_sq=self._posterior.T0_sq,
            T1_sq=self._posterior.T1_sq,
            seed=seeds[4],
        )

        # Package the updated chain state and acceptance diagnostics for the next iteration.
        new_state = LuShrinkageState(
            beta_p=beta_p_new,
            beta_w=beta_w_new,
            r=r_new,
            E_bar=E_bar_new,
            njt=njt_new,
            gamma=gamma_new,
        )
        new_results = LuHybridKernelResults(
            beta_accept=beta_accept,
            r_accept=r_accept,
            E_bar_accept=E_bar_accept,
            njt_accept=njt_accept,
        )
        return new_state, new_results

    def copy(self, **kwargs):
        """Return a copy of the kernel with updated parameters."""

        parameters = dict(self.parameters)
        parameters.update(kwargs)
        return type(self)(**parameters)


def _make_trace(
    posterior: LuPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    current_state: LuShrinkageState,
    kernel_results: LuHybridKernelResults,
) -> dict[str, tf.Tensor]:
    """Build the per-draw diagnostics recorded during sampling."""

    # Record the current joint log posterior together with state summaries and acceptance indicators.
    joint_lp = posterior.joint_logpost(
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        beta_p=current_state.beta_p,
        beta_w=current_state.beta_w,
        r=current_state.r,
        E_bar=current_state.E_bar,
        njt=current_state.njt,
        gamma=current_state.gamma,
    )

    return {
        "beta_p": current_state.beta_p,
        "beta_w": current_state.beta_w,
        "sigma": tf.exp(current_state.r),
        "mean_E_bar": tf.reduce_mean(current_state.E_bar),
        "norm_E_bar": tf.linalg.norm(current_state.E_bar),
        "norm_njt": tf.linalg.norm(current_state.njt),
        "gamma_active_share": tf.reduce_mean(current_state.gamma),
        "joint_logpost": joint_lp,
        "beta_accept": kernel_results.beta_accept,
        "r_accept": kernel_results.r_accept,
        "E_bar_accept": kernel_results.E_bar_accept,
        "njt_accept": kernel_results.njt_accept,
    }


def run_chain(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config: LuPosteriorConfig,
    shrinkage_config: LuShrinkageConfig,
    seed: tf.Tensor,
) -> LuShrinkageState:
    """Validate inputs, tune scales, run the chain, and return retained draws."""

    # Validate all external inputs before constructing posterior or sampler objects.
    run_chain_validate_input(
        pjt=pjt,
        wjt=wjt,
        qjt=qjt,
        q0t=q0t,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=seed,
    )

    # Initialize the posterior object and the default starting state.
    posterior = LuPosteriorTF(posterior_config)
    initial_state = build_initial_state(pjt=pjt, posterior=posterior)

    # Use separate seed streams for tuning and retained sampling.
    seeds = tf.random.experimental.stateless_split(seed, num=2)
    tuning_seed = seeds[0]
    sampling_seed = seeds[1]

    # Tune the proposal scales for the continuous blocks before the main run.
    tuned_config = tune_shrinkage(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        initial_state=initial_state,
        shrinkage_config=shrinkage_config,
        pilot_length=shrinkage_config.pilot_length,
        target_low=shrinkage_config.target_low,
        target_high=shrinkage_config.target_high,
        max_rounds=shrinkage_config.max_rounds,
        factor=shrinkage_config.factor,
        seed=tuning_seed,
    )

    # Build the transition kernel for the main run using the tuned config.
    kernel = LuHybridKernel(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        config=tuned_config,
    )

    def trace_fn(
        current_state: LuShrinkageState,
        kernel_results: LuHybridKernelResults,
    ) -> dict[str, tf.Tensor]:
        """Record the per-draw diagnostics reported at chunk boundaries."""

        return _make_trace(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            current_state=current_state,
            kernel_results=kernel_results,
        )

    # Run one compiled chunk of the chain at a time.
    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_chunk(
        current_state: LuShrinkageState,
        previous_kernel_results: LuHybridKernelResults,
        num_steps: tf.Tensor,
        chunk_seed: tf.Tensor,
    ):
        """Run one chunk of the chain and return samples, trace, and kernel results."""

        return tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=0,
            current_state=current_state,
            previous_kernel_results=previous_kernel_results,
            kernel=kernel,
            trace_fn=trace_fn,
            return_final_kernel_results=True,
            seed=chunk_seed,
        )

    # Start the main run from the default initial state and empty acceptance diagnostics.
    state = initial_state
    kernel_results = kernel.bootstrap_results(initial_state)

    # Execute burn-in and retained draws in chunked form.
    burnin_chunks = _num_chunks(tuned_config.num_burnin_steps, tuned_config.chunk_size)
    result_chunks = _num_chunks(tuned_config.num_results, tuned_config.chunk_size)
    total_chunks = burnin_chunks + result_chunks

    chunk_seeds = tf.random.experimental.stateless_split(
        sampling_seed, num=total_chunks
    )

    # Store reported chunk summaries separately from retained posterior draws.
    chunk_idx = 0
    chunk_summaries: list[LuChunkSummary] = []
    retained_chunks: list[LuShrinkageState] = []

    def run_phase(remaining_steps: int, retain: bool) -> int:
        """Execute one chunked phase of the chain."""

        nonlocal chunk_idx, state, kernel_results

        while remaining_steps > 0:
            this_chunk = min(tuned_config.chunk_size, remaining_steps)

            # Advance the chain by one chunk.
            chunk_samples, raw_trace, kernel_results = _run_chunk(
                current_state=state,
                previous_kernel_results=kernel_results,
                num_steps=tf.constant(this_chunk, dtype=tf.int32),
                chunk_seed=chunk_seeds[chunk_idx],
            )

            # Start the next chunk from the terminal state of the current chunk.
            state = _last_state(chunk_samples)

            # Append only retained-phase chunks to posterior output.
            if retain:
                retained_chunks.append(chunk_samples)

            # Convert and report the chunk diagnostics after each chunk.
            trace = _raw_trace_to_dataclass(raw_trace)
            summary = report_chunk_progress(
                trace=trace,
                chunk_idx=chunk_idx + 1,
                total_chunks=total_chunks,
            )
            chunk_summaries.append(summary)

            remaining_steps -= this_chunk
            chunk_idx += 1

        return remaining_steps

    # Run the chain in two stages: burn-in first, then retained sampling.
    run_phase(tuned_config.num_burnin_steps, retain=False)
    run_phase(tuned_config.num_results, retain=True)

    # Report the final run-level summary once chunk processing is complete.
    if len(chunk_summaries) > 0:
        report_run_summary(chunk_summaries)

    # Concatenate all retained sample chunks into one state trace.
    return _concat_sample_chunks(retained_chunks)


def summarize_samples(
    samples: LuShrinkageState,
) -> dict[str, tf.Tensor]:
    """Compute posterior mean summaries from retained chain output."""

    # Compute posterior means of the sampled parameter blocks.
    beta_p_hat = tf.reduce_mean(samples.beta_p, axis=0)
    beta_w_hat = tf.reduce_mean(samples.beta_w, axis=0)
    sigma_hat = tf.reduce_mean(tf.exp(samples.r), axis=0)

    E_bar_hat = tf.reduce_mean(samples.E_bar, axis=0)
    njt_hat = tf.reduce_mean(samples.njt, axis=0)
    gamma_hat = tf.reduce_mean(samples.gamma, axis=0)

    # Derive the reported intercept and the full shock estimate from posterior means.
    int_hat = tf.reduce_mean(E_bar_hat)
    E_hat = E_bar_hat[:, None] + njt_hat

    return {
        "beta_p_hat": beta_p_hat,
        "beta_w_hat": beta_w_hat,
        "sigma_hat": sigma_hat,
        "int_hat": int_hat,
        "E_hat": E_hat,
        "E_bar_hat": E_bar_hat,
        "njt_hat": njt_hat,
        "gamma_hat": gamma_hat,
    }
