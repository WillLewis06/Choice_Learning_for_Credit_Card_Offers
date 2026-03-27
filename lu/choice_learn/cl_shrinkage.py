"""Run the choice-learn shrinkage MCMC chain and summarize retained samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from lu.choice_learn.cl_diagnostics import (
    ChoiceLearnChunkSummary,
    format_chunk_progress_line,
    format_run_summary_line,
    summarize_chunk_trace,
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
    """Store chain controls and pilot-tuning controls for shrinkage sampling."""

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
    """Store the full chain state for the choice-learn shrinkage sampler."""

    alpha: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    gamma: tf.Tensor


class ChoiceLearnHybridKernelResults(NamedTuple):
    """Store acceptance diagnostics for one hybrid transition."""

    alpha_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


def _last_state(samples: ChoiceLearnShrinkageState) -> ChoiceLearnShrinkageState:
    """Return the terminal state from one sampled chunk."""

    return ChoiceLearnShrinkageState(
        alpha=samples.alpha[-1],
        E_bar=samples.E_bar[-1],
        njt=samples.njt[-1],
        gamma=samples.gamma[-1],
    )


def _concat_sample_chunks(
    chunks: list[ChoiceLearnShrinkageState],
) -> ChoiceLearnShrinkageState:
    """Concatenate retained chunks into one full retained sample trace."""

    if len(chunks) == 0:
        raise ValueError("No retained sample chunks were produced.")

    return ChoiceLearnShrinkageState(
        alpha=tf.concat([chunk.alpha for chunk in chunks], axis=0),
        E_bar=tf.concat([chunk.E_bar for chunk in chunks], axis=0),
        njt=tf.concat([chunk.njt for chunk in chunks], axis=0),
        gamma=tf.concat([chunk.gamma for chunk in chunks], axis=0),
    )


def build_initial_state(
    delta_cl: tf.Tensor,
    E_bar_mean: tf.Tensor,
) -> ChoiceLearnShrinkageState:
    """Construct the default chain state before tuning and sampling."""

    num_markets = tf.shape(delta_cl)[0]
    num_products = tf.shape(delta_cl)[1]

    return ChoiceLearnShrinkageState(
        alpha=tf.constant(0.0, dtype=tf.float64),
        E_bar=tf.fill([num_markets], E_bar_mean),
        njt=tf.zeros([num_markets, num_products], dtype=tf.float64),
        gamma=tf.zeros([num_markets, num_products], dtype=tf.float64),
    )


class ChoiceLearnHybridKernel(tfp.mcmc.TransitionKernel):
    """Implement one hybrid transition of the shrinkage sampler."""

    def __init__(
        self,
        posterior: ChoiceLearnPosteriorTF,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        delta_cl: tf.Tensor,
        k_alpha: float | tf.Tensor,
        k_E_bar: float | tf.Tensor,
        k_njt: float | tf.Tensor,
    ):
        """Store observed data, posterior object, and tuned proposal scales."""

        self._posterior = posterior
        self._qjt = qjt
        self._q0t = q0t
        self._delta_cl = delta_cl

        self._k_alpha = tf.constant(k_alpha, dtype=tf.float64)
        self._k_E_bar = tf.constant(k_E_bar, dtype=tf.float64)
        self._k_njt = tf.constant(k_njt, dtype=tf.float64)

        self._parameters = {
            "posterior": posterior,
            "qjt": qjt,
            "q0t": q0t,
            "delta_cl": delta_cl,
            "k_alpha": k_alpha,
            "k_E_bar": k_E_bar,
            "k_njt": k_njt,
        }

    @property
    def parameters(self):
        """Return the kernel parameters required by the TFP interface."""

        return self._parameters

    @property
    def is_calibrated(self) -> bool:
        """Report that this kernel is treated as calibrated."""

        return True

    def bootstrap_results(
        self,
        current_state: ChoiceLearnShrinkageState,
    ) -> ChoiceLearnHybridKernelResults:
        """Initialize acceptance diagnostics before the first transition."""

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
        """Apply one full hybrid transition."""

        del previous_kernel_results

        # Split one transition seed across the four block updates.
        block_seeds = tf.random.experimental.stateless_split(seed, num=4)

        alpha_new, alpha_accept = alpha_one_step(
            posterior=self._posterior,
            qjt=self._qjt,
            q0t=self._q0t,
            delta_cl=self._delta_cl,
            alpha=current_state.alpha,
            E_bar=current_state.E_bar,
            njt=current_state.njt,
            k_alpha=self._k_alpha,
            seed=block_seeds[0],
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
            seed=block_seeds[1],
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
            seed=block_seeds[2],
        )

        gamma_new = gamma_one_step(
            posterior=self._posterior,
            njt=njt_new,
            gamma=current_state.gamma,
            seed=block_seeds[3],
        )

        return (
            ChoiceLearnShrinkageState(
                alpha=alpha_new,
                E_bar=E_bar_new,
                njt=njt_new,
                gamma=gamma_new,
            ),
            ChoiceLearnHybridKernelResults(
                alpha_accept=alpha_accept,
                E_bar_accept=E_bar_accept,
                njt_accept=njt_accept,
            ),
        )

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
    """Build the per-draw diagnostics recorded during chunked sampling."""

    joint_logpost = posterior.joint_logpost(
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
        "joint_logpost": joint_logpost,
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
    """Validate inputs, tune proposal scales, run the chain, and return draws."""

    run_chain_validate_input(
        delta_cl=delta_cl,
        qjt=qjt,
        q0t=q0t,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=seed,
    )

    posterior = ChoiceLearnPosteriorTF(posterior_config)
    initial_state = build_initial_state(
        delta_cl=delta_cl,
        E_bar_mean=posterior.E_bar_mean,
    )

    # stateless_split returns one tensor of shape (2, 2), so index the split
    # seeds explicitly rather than tuple-unpacking a tensor.
    split_seeds = tf.random.experimental.stateless_split(seed, num=2)
    tuning_seed = split_seeds[0]
    sampling_seed = split_seeds[1]

    tuned_config = tune_shrinkage(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        initial_state=initial_state,
        shrinkage_config=shrinkage_config,
        seed=tuning_seed,
    )

    kernel = ChoiceLearnHybridKernel(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        delta_cl=delta_cl,
        k_alpha=tuned_config.k_alpha,
        k_E_bar=tuned_config.k_E_bar,
        k_njt=tuned_config.k_njt,
    )

    def trace_fn(
        current_state: ChoiceLearnShrinkageState,
        kernel_results: ChoiceLearnHybridKernelResults,
    ) -> dict[str, tf.Tensor]:
        """Record the per-draw diagnostics used by chunk reporting."""

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
        """Run one chunk of the chain and return samples, trace, and results."""

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

    burnin_chunks = (
        0
        if tuned_config.num_burnin_steps == 0
        else (tuned_config.num_burnin_steps + tuned_config.chunk_size - 1)
        // tuned_config.chunk_size
    )
    result_chunks = (tuned_config.num_results + tuned_config.chunk_size - 1) // (
        tuned_config.chunk_size
    )
    total_chunks = burnin_chunks + result_chunks

    chunk_seeds = tf.random.experimental.stateless_split(
        sampling_seed,
        num=total_chunks,
    )

    chunk_idx = 0
    chunk_summaries: list[ChoiceLearnChunkSummary] = []
    retained_chunks: list[ChoiceLearnShrinkageState] = []

    def run_phase(remaining_steps: int, retain: bool) -> None:
        """Run one chunked phase of the chain."""

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

            # Report diagnostics for every chunk, but only keep retained chunks
            # for posterior summarization.
            if retain:
                retained_chunks.append(chunk_samples)

            summary = summarize_chunk_trace(
                trace=raw_trace,
                chunk_idx=chunk_idx + 1,
                total_chunks=total_chunks,
            )
            print(format_chunk_progress_line(summary))
            chunk_summaries.append(summary)

            remaining_steps -= this_chunk
            chunk_idx += 1

    run_phase(tuned_config.num_burnin_steps, retain=False)
    run_phase(tuned_config.num_results, retain=True)

    if len(chunk_summaries) > 0:
        print(format_run_summary_line(chunk_summaries))

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
        "E_bar_hat": E_bar_hat,
        "njt_hat": njt_hat,
        "gamma_hat": gamma_hat,
        "E_hat": E_hat,
    }
