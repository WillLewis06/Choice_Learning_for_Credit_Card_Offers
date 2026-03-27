"""Run the choice-learn shrinkage MCMC chain and summarize retained samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from lu.choice_learn.cl_diagnostics import (
    ChoiceLearnChunkSummary,
    ChoiceLearnChunkTrace,
    report_chunk_progress,
    report_run_summary,
)
from lu.choice_learn.cl_posterior import (
    ChoiceLearnPosteriorConfig,
    ChoiceLearnPosteriorTF,
)
from lu.choice_learn.cl_tuning import tune_shrinkage
from lu.choice_learn.cl_updates import (
    E_bar_one_step,
    alpha_one_step,
    gamma_one_step,
    njt_one_step,
)
from lu.choice_learn.cl_validate_input import run_chain_validate_input


@dataclass(frozen=True)
class ChoiceLearnShrinkageConfig:
    """Store sampler controls, proposal scales, and tuning parameters."""

    num_results: int
    num_burnin_steps: int
    chunk_size: int
    k_alpha: float
    k_E_bar: float
    k_njt: float
    pilot_length: int
    target_low: float
    target_high: float
    max_rounds: int
    factor: float


class ChoiceLearnShrinkageState(NamedTuple):
    """Store the current state of the choice-learn shrinkage chain."""

    alpha: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    gamma: tf.Tensor


class ChoiceLearnHybridKernelResults(NamedTuple):
    """Store the acceptance diagnostics for one hybrid transition step."""

    alpha_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


def _num_chunks(n_steps: int, chunk_size: int) -> int:
    """Return the number of chunks needed for a given step count."""

    if n_steps <= 0:
        return 0
    return (n_steps + chunk_size - 1) // chunk_size


def _last_state(samples: ChoiceLearnShrinkageState) -> ChoiceLearnShrinkageState:
    """Extract the terminal state from a chunk of sampled states."""

    return ChoiceLearnShrinkageState(
        alpha=samples.alpha[-1],
        E_bar=samples.E_bar[-1],
        njt=samples.njt[-1],
        gamma=samples.gamma[-1],
    )


def _concat_sample_chunks(
    chunks: list[ChoiceLearnShrinkageState],
) -> ChoiceLearnShrinkageState:
    """Concatenate retained sample chunks into one state trace."""

    if len(chunks) == 0:
        raise ValueError("No retained sample chunks were produced.")

    return ChoiceLearnShrinkageState(
        alpha=tf.concat([chunk.alpha for chunk in chunks], axis=0),
        E_bar=tf.concat([chunk.E_bar for chunk in chunks], axis=0),
        njt=tf.concat([chunk.njt for chunk in chunks], axis=0),
        gamma=tf.concat([chunk.gamma for chunk in chunks], axis=0),
    )


def _raw_trace_to_dataclass(
    raw_trace: dict[str, tf.Tensor],
) -> ChoiceLearnChunkTrace:
    """Convert the raw trace dictionary into a ChoiceLearnChunkTrace."""

    return ChoiceLearnChunkTrace(
        alpha=raw_trace["alpha"],
        mean_E_bar=raw_trace["mean_E_bar"],
        norm_E_bar=raw_trace["norm_E_bar"],
        norm_njt=raw_trace["norm_njt"],
        gamma_active_share=raw_trace["gamma_active_share"],
        joint_logpost=raw_trace["joint_logpost"],
        alpha_accept=raw_trace["alpha_accept"],
        E_bar_accept=raw_trace["E_bar_accept"],
        njt_accept=raw_trace["njt_accept"],
    )


def build_initial_state(
    delta_cl: tf.Tensor,
    posterior: ChoiceLearnPosteriorTF,
) -> ChoiceLearnShrinkageState:
    """Construct the default initial state for the shrinkage chain."""

    T = tf.shape(delta_cl)[0]
    J = tf.shape(delta_cl)[1]

    return ChoiceLearnShrinkageState(
        alpha=tf.constant(0.0, dtype=tf.float64),
        E_bar=tf.fill([T], posterior.E_bar_mean),
        njt=tf.zeros([T, J], dtype=tf.float64),
        gamma=tf.zeros([T, J], dtype=tf.float64),
    )


class ChoiceLearnHybridKernel(tfp.mcmc.TransitionKernel):
    """Implement one hybrid transition of the choice-learn shrinkage sampler."""

    def __init__(
        self,
        posterior: ChoiceLearnPosteriorTF,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        config: ChoiceLearnShrinkageConfig,
    ):
        """Store the posterior object, observed data, and proposal scales."""

        self._posterior = posterior
        self._qjt = qjt
        self._q0t = q0t
        self._delta_cl = delta_cl
        self._config = config

        self._k_alpha = tf.constant(config.k_alpha, dtype=tf.float64)
        self._k_E_bar = tf.constant(config.k_E_bar, dtype=tf.float64)
        self._k_njt = tf.constant(config.k_njt, dtype=tf.float64)

        self._parameters = {
            "posterior": posterior,
            "qjt": qjt,
            "q0t": q0t,
            "delta_cl": delta_cl,
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
        current_state: ChoiceLearnShrinkageState,
    ) -> ChoiceLearnHybridKernelResults:
        """Initialize the acceptance diagnostics for the first step."""

        del current_state
        zero = tf.constant(0.0, dtype=tf.float64)

        return ChoiceLearnHybridKernelResults(
            alpha_accept=zero,
            E_bar_accept=zero,
            njt_accept=zero,
        )

    def one_step(
        self,
        current_state: ChoiceLearnShrinkageState,
        previous_kernel_results: ChoiceLearnHybridKernelResults,
        seed: tf.Tensor,
    ) -> tuple[ChoiceLearnShrinkageState, ChoiceLearnHybridKernelResults]:
        """Apply one full hybrid transition of the shrinkage chain."""

        del previous_kernel_results

        seeds = tf.random.experimental.stateless_split(seed, num=4)

        alpha_new, alpha_accept = alpha_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            delta_cl=self._delta_cl,
            alpha=current_state.alpha,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_alpha=self._k_alpha,
            seed=seeds[0],
        )

        E_bar_new, E_bar_accept = E_bar_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            delta_cl=self._delta_cl,
            alpha=alpha_new,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_E_bar=self._k_E_bar,
            seed=seeds[1],
        )

        njt_new, njt_accept = njt_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            delta_cl=self._delta_cl,
            alpha=alpha_new,
            E_bar=E_bar_new,
            njt=current_state.njt,
            gamma=current_state.gamma,
            k_njt=self._k_njt,
            seed=seeds[2],
        )

        gamma_new = gamma_one_step(
            posterior=self._posterior,
            njt=njt_new,
            gamma=current_state.gamma,
            seed=seeds[3],
        )

        new_state = ChoiceLearnShrinkageState(
            alpha=alpha_new,
            E_bar=E_bar_new,
            njt=njt_new,
            gamma=gamma_new,
        )
        new_results = ChoiceLearnHybridKernelResults(
            alpha_accept=alpha_accept,
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
    posterior: ChoiceLearnPosteriorTF,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    delta_cl: tf.Tensor,
    current_state: ChoiceLearnShrinkageState,
    kernel_results: ChoiceLearnHybridKernelResults,
) -> dict[str, tf.Tensor]:
    """Build the per-draw diagnostics recorded during sampling."""

    joint_lp = posterior.joint_logpost(
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        alpha=current_state.alpha,
        E_bar=current_state.E_bar,
        njt=current_state.njt,
        gamma=current_state.gamma,
    )

    return {
        "alpha": current_state.alpha,
        "mean_E_bar": tf.reduce_mean(current_state.E_bar),
        "norm_E_bar": tf.linalg.norm(current_state.E_bar),
        "norm_njt": tf.linalg.norm(current_state.njt),
        "gamma_active_share": tf.reduce_mean(current_state.gamma),
        "joint_logpost": joint_lp,
        "alpha_accept": kernel_results.alpha_accept,
        "E_bar_accept": kernel_results.E_bar_accept,
        "njt_accept": kernel_results.njt_accept,
    }


def run_chain(
    delta_cl: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config: ChoiceLearnPosteriorConfig,
    shrinkage_config: ChoiceLearnShrinkageConfig,
    seed: tf.Tensor,
) -> ChoiceLearnShrinkageState:
    """Validate inputs, tune scales, run the chain, and return retained draws."""

    run_chain_validate_input(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=seed,
    )

    posterior = ChoiceLearnPosteriorTF(posterior_config)
    initial_state = build_initial_state(delta_cl=delta_cl, posterior=posterior)

    seeds = tf.random.experimental.stateless_split(seed, num=2)
    tuning_seed = seeds[0]
    sampling_seed = seeds[1]

    tuned_config = tune_shrinkage(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        initial_state=initial_state,
        shrinkage_config=shrinkage_config,
        pilot_length=shrinkage_config.pilot_length,
        target_low=shrinkage_config.target_low,
        target_high=shrinkage_config.target_high,
        max_rounds=shrinkage_config.max_rounds,
        factor=shrinkage_config.factor,
        seed=tuning_seed,
    )

    kernel = ChoiceLearnHybridKernel(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        config=tuned_config,
    )

    def trace_fn(
        current_state: ChoiceLearnShrinkageState,
        kernel_results: ChoiceLearnHybridKernelResults,
    ) -> dict[str, tf.Tensor]:
        """Record the per-draw diagnostics reported at chunk boundaries."""

        return _make_trace(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            delta_cl=delta_cl,
            current_state=current_state,
            kernel_results=kernel_results,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_chunk(
        current_state: ChoiceLearnShrinkageState,
        previous_kernel_results: ChoiceLearnHybridKernelResults,
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

    state = initial_state
    kernel_results = kernel.bootstrap_results(initial_state)

    burnin_chunks = _num_chunks(tuned_config.num_burnin_steps, tuned_config.chunk_size)
    result_chunks = _num_chunks(tuned_config.num_results, tuned_config.chunk_size)
    total_chunks = burnin_chunks + result_chunks

    chunk_seeds = tf.random.experimental.stateless_split(
        sampling_seed,
        num=total_chunks,
    )

    chunk_idx = 0
    chunk_summaries: list[ChoiceLearnChunkSummary] = []
    retained_chunks: list[ChoiceLearnShrinkageState] = []

    def run_phase(remaining_steps: int, retain: bool) -> int:
        """Execute one chunked phase of the chain."""

        nonlocal chunk_idx, state, kernel_results

        while remaining_steps > 0:
            this_chunk = min(tuned_config.chunk_size, remaining_steps)

            chunk_samples, raw_trace, kernel_results = _run_chunk(
                current_state=state,
                previous_kernel_results=kernel_results,
                num_steps=tf.constant(this_chunk, dtype=tf.int32),
                chunk_seed=chunk_seeds[chunk_idx],
            )

            state = _last_state(chunk_samples)

            if retain:
                retained_chunks.append(chunk_samples)

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

    run_phase(tuned_config.num_burnin_steps, retain=False)
    run_phase(tuned_config.num_results, retain=True)

    if len(chunk_summaries) > 0:
        report_run_summary(chunk_summaries)

    return _concat_sample_chunks(retained_chunks)


def summarize_samples(
    samples: ChoiceLearnShrinkageState,
) -> dict[str, tf.Tensor]:
    """Compute posterior mean summaries from retained chain output."""

    alpha_hat = tf.reduce_mean(samples.alpha, axis=0)
    E_bar_hat = tf.reduce_mean(samples.E_bar, axis=0)
    njt_hat = tf.reduce_mean(samples.njt, axis=0)
    gamma_hat = tf.reduce_mean(samples.gamma, axis=0)
    E_hat = E_bar_hat[:, None] + njt_hat

    return {
        "alpha_hat": alpha_hat,
        "E_hat": E_hat,
        "E_bar_hat": E_bar_hat,
        "njt_hat": njt_hat,
        "gamma_hat": gamma_hat,
    }
