"""Run the Ching-style stockpiling MCMC chain and summarize draws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from ching.stockpiling_diagnostics import (
    StockpilingChunkSummary,
    StockpilingChunkTrace,
    build_mcmc_summary,
    report_chunk_progress,
    report_run_summary,
)
from ching.stockpiling_input_validation import run_chain_validate_input
from ching.stockpiling_model import unconstrained_to_theta
from ching.stockpiling_posterior import (
    StockpilingPosteriorConfig,
    StockpilingPosteriorTF,
)
from ching.stockpiling_updates import (
    alpha_one_step,
    beta_one_step,
    fc_one_step,
    u_scale_one_step,
    v_one_step,
)

__all__ = [
    "StockpilingConfig",
    "StockpilingState",
    "StockpilingKernelResults",
    "StockpilingRunResult",
    "StockpilingHybridKernel",
    "build_initial_state",
    "run_chain",
    "summarize_samples",
]


@dataclass(frozen=True)
class StockpilingConfig:
    """Store the sampler controls and proposal scales."""

    num_results: int
    num_burnin_steps: int
    chunk_size: int
    k_beta: tf.Tensor
    k_alpha: tf.Tensor
    k_v: tf.Tensor
    k_fc: tf.Tensor
    k_u_scale: tf.Tensor


class StockpilingState(NamedTuple):
    """Store the current unconstrained chain state."""

    z_beta: tf.Tensor
    z_alpha: tf.Tensor
    z_v: tf.Tensor
    z_fc: tf.Tensor
    z_u_scale: tf.Tensor


class StockpilingKernelResults(NamedTuple):
    """Store scalar block acceptance diagnostics for one transition step."""

    beta_accept: tf.Tensor
    alpha_accept: tf.Tensor
    v_accept: tf.Tensor
    fc_accept: tf.Tensor
    u_scale_accept: tf.Tensor


@dataclass(frozen=True)
class StockpilingRunResult:
    """Store retained samples and reporting summaries from one chain run."""

    samples: StockpilingState
    chunk_summaries: list[StockpilingChunkSummary]
    mcmc_summary: dict[str, Any]


def _num_chunks(n_steps: int, chunk_size: int) -> int:
    """Return the number of chunks required for n_steps transitions."""
    if n_steps <= 0:
        return 0
    return (n_steps + chunk_size - 1) // chunk_size


def _last_state(samples: StockpilingState) -> StockpilingState:
    """Return the terminal state from a sampled chunk."""
    return StockpilingState(
        z_beta=samples.z_beta[-1],
        z_alpha=samples.z_alpha[-1],
        z_v=samples.z_v[-1],
        z_fc=samples.z_fc[-1],
        z_u_scale=samples.z_u_scale[-1],
    )


def _concat_sample_chunks(chunks: list[StockpilingState]) -> StockpilingState:
    """Concatenate retained chunks into one stacked sample object."""
    if not chunks:
        raise ValueError("No retained sample chunks were produced.")

    return StockpilingState(
        z_beta=tf.concat([chunk.z_beta for chunk in chunks], axis=0),
        z_alpha=tf.concat([chunk.z_alpha for chunk in chunks], axis=0),
        z_v=tf.concat([chunk.z_v for chunk in chunks], axis=0),
        z_fc=tf.concat([chunk.z_fc for chunk in chunks], axis=0),
        z_u_scale=tf.concat([chunk.z_u_scale for chunk in chunks], axis=0),
    )


def _raw_trace_to_dataclass(raw_trace: dict[str, tf.Tensor]) -> StockpilingChunkTrace:
    """Convert the raw TFP trace dictionary into a diagnostics dataclass."""
    return StockpilingChunkTrace(
        beta=raw_trace["beta"],
        mean_alpha=raw_trace["mean_alpha"],
        mean_v=raw_trace["mean_v"],
        mean_fc=raw_trace["mean_fc"],
        mean_u_scale=raw_trace["mean_u_scale"],
        joint_logpost=raw_trace["joint_logpost"],
        beta_accept=raw_trace["beta_accept"],
        alpha_accept=raw_trace["alpha_accept"],
        v_accept=raw_trace["v_accept"],
        fc_accept=raw_trace["fc_accept"],
        u_scale_accept=raw_trace["u_scale_accept"],
    )


def build_initial_state(M: int, J: int) -> StockpilingState:
    """Build the default unconstrained initial state."""
    return StockpilingState(
        z_beta=tf.constant(0.0, dtype=tf.float64),
        z_alpha=tf.zeros((J,), dtype=tf.float64),
        z_v=tf.zeros((J,), dtype=tf.float64),
        z_fc=tf.zeros((J,), dtype=tf.float64),
        z_u_scale=tf.zeros((M,), dtype=tf.float64),
    )


class StockpilingHybridKernel(tfp.mcmc.TransitionKernel):
    """Implement one hybrid transition of the stockpiling sampler."""

    def __init__(
        self,
        posterior: StockpilingPosteriorTF,
        config: StockpilingConfig,
    ):
        """Store the posterior object and proposal scales."""
        self._posterior = posterior
        self._config = config

        self._k_beta = config.k_beta
        self._k_alpha = config.k_alpha
        self._k_v = config.k_v
        self._k_fc = config.k_fc
        self._k_u_scale = config.k_u_scale

        self._parameters = {
            "posterior": posterior,
            "config": config,
        }

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the kernel parameters required by the TFP interface."""
        return self._parameters

    @property
    def is_calibrated(self) -> bool:
        """Return whether the kernel is treated as calibrated."""
        return True

    def bootstrap_results(
        self,
        current_state: StockpilingState,
    ) -> StockpilingKernelResults:
        """Initialize acceptance diagnostics for the first transition."""
        del current_state
        zero = tf.constant(0.0, dtype=tf.float64)
        return StockpilingKernelResults(
            beta_accept=zero,
            alpha_accept=zero,
            v_accept=zero,
            fc_accept=zero,
            u_scale_accept=zero,
        )

    def one_step(
        self,
        current_state: StockpilingState,
        previous_kernel_results: StockpilingKernelResults,
        seed: tf.Tensor,
    ) -> tuple[StockpilingState, StockpilingKernelResults]:
        """Apply one full hybrid transition of the stockpiling chain."""
        del previous_kernel_results
        seeds = tf.random.experimental.stateless_split(seed, num=5)

        z_beta_new, beta_accept = beta_one_step(
            posterior=self._posterior,
            z_beta=current_state.z_beta,
            z_alpha=current_state.z_alpha,
            z_v=current_state.z_v,
            z_fc=current_state.z_fc,
            z_u_scale=current_state.z_u_scale,
            k_beta=self._k_beta,
            seed=seeds[0],
        )
        z_alpha_new, alpha_accept = alpha_one_step(
            posterior=self._posterior,
            z_beta=z_beta_new,
            z_alpha=current_state.z_alpha,
            z_v=current_state.z_v,
            z_fc=current_state.z_fc,
            z_u_scale=current_state.z_u_scale,
            k_alpha=self._k_alpha,
            seed=seeds[1],
        )
        z_v_new, v_accept = v_one_step(
            posterior=self._posterior,
            z_beta=z_beta_new,
            z_alpha=z_alpha_new,
            z_v=current_state.z_v,
            z_fc=current_state.z_fc,
            z_u_scale=current_state.z_u_scale,
            k_v=self._k_v,
            seed=seeds[2],
        )
        z_fc_new, fc_accept = fc_one_step(
            posterior=self._posterior,
            z_beta=z_beta_new,
            z_alpha=z_alpha_new,
            z_v=z_v_new,
            z_fc=current_state.z_fc,
            z_u_scale=current_state.z_u_scale,
            k_fc=self._k_fc,
            seed=seeds[3],
        )
        z_u_scale_new, u_scale_accept = u_scale_one_step(
            posterior=self._posterior,
            z_beta=z_beta_new,
            z_alpha=z_alpha_new,
            z_v=z_v_new,
            z_fc=z_fc_new,
            z_u_scale=current_state.z_u_scale,
            k_u_scale=self._k_u_scale,
            seed=seeds[4],
        )

        next_state = StockpilingState(
            z_beta=z_beta_new,
            z_alpha=z_alpha_new,
            z_v=z_v_new,
            z_fc=z_fc_new,
            z_u_scale=z_u_scale_new,
        )
        next_results = StockpilingKernelResults(
            beta_accept=beta_accept,
            alpha_accept=alpha_accept,
            v_accept=v_accept,
            fc_accept=fc_accept,
            u_scale_accept=u_scale_accept,
        )
        return next_state, next_results

    def copy(self, **kwargs):
        """Return a copy of the kernel with updated parameters."""
        parameters = dict(self.parameters)
        parameters.update(kwargs)
        return type(self)(**parameters)


def _make_trace(
    posterior: StockpilingPosteriorTF,
    current_state: StockpilingState,
    kernel_results: StockpilingKernelResults,
) -> dict[str, tf.Tensor]:
    """Build the per-draw diagnostics recorded during sampling."""
    theta = unconstrained_to_theta(
        {
            "z_beta": current_state.z_beta,
            "z_alpha": current_state.z_alpha,
            "z_v": current_state.z_v,
            "z_fc": current_state.z_fc,
            "z_u_scale": current_state.z_u_scale,
        }
    )
    joint_logpost = posterior.joint_logpost(
        z_beta=current_state.z_beta,
        z_alpha=current_state.z_alpha,
        z_v=current_state.z_v,
        z_fc=current_state.z_fc,
        z_u_scale=current_state.z_u_scale,
    )

    return {
        "beta": theta["beta"],
        "mean_alpha": tf.reduce_mean(theta["alpha"]),
        "mean_v": tf.reduce_mean(theta["v"]),
        "mean_fc": tf.reduce_mean(theta["fc"]),
        "mean_u_scale": tf.reduce_mean(theta["u_scale"]),
        "joint_logpost": joint_logpost,
        "beta_accept": kernel_results.beta_accept,
        "alpha_accept": kernel_results.alpha_accept,
        "v_accept": kernel_results.v_accept,
        "fc_accept": kernel_results.fc_accept,
        "u_scale_accept": kernel_results.u_scale_accept,
    }


def run_chain(
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    u_mj: tf.Tensor,
    P_price_mj: tf.Tensor,
    price_vals_mj: tf.Tensor,
    lambda_mn: tf.Tensor,
    waste_cost: tf.Tensor,
    inventory_maps,
    pi_I0: tf.Tensor,
    posterior_config: StockpilingPosteriorConfig,
    stockpiling_config: StockpilingConfig,
    initial_state: StockpilingState,
    seed: tf.Tensor,
) -> StockpilingRunResult:
    """Validate inputs, run the chain, and return samples plus summaries."""
    run_chain_validate_input(
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        u_mj=u_mj,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        inventory_maps=inventory_maps,
        pi_I0=pi_I0,
        posterior_config=posterior_config,
        sampler_config=stockpiling_config,
        z_beta=initial_state.z_beta,
        z_alpha=initial_state.z_alpha,
        z_v=initial_state.z_v,
        z_fc=initial_state.z_fc,
        z_u_scale=initial_state.z_u_scale,
        seed=seed,
    )

    posterior = StockpilingPosteriorTF(
        config=posterior_config,
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        u_mj=u_mj,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        lambda_mn=lambda_mn,
        waste_cost=waste_cost,
        inventory_maps=inventory_maps,
        pi_I0=pi_I0,
    )
    kernel = StockpilingHybridKernel(
        posterior=posterior,
        config=stockpiling_config,
    )

    @tf.function(jit_compile=True)
    def _run_chunk(
        current_state: StockpilingState,
        previous_kernel_results: StockpilingKernelResults,
        num_steps: tf.Tensor,
        chunk_seed: tf.Tensor,
    ) -> tuple[StockpilingState, dict[str, tf.Tensor], StockpilingKernelResults]:
        """Run one compiled chunk of transitions."""
        return tfp.mcmc.sample_chain(
            num_results=num_steps,
            current_state=current_state,
            previous_kernel_results=previous_kernel_results,
            kernel=kernel,
            trace_fn=lambda state, results: _make_trace(posterior, state, results),
            return_final_kernel_results=True,
            seed=chunk_seed,
        )

    total_burnin_chunks = _num_chunks(
        stockpiling_config.num_burnin_steps,
        stockpiling_config.chunk_size,
    )
    total_retained_chunks = _num_chunks(
        stockpiling_config.num_results,
        stockpiling_config.chunk_size,
    )
    total_chunks = total_burnin_chunks + total_retained_chunks

    chunk_seeds = tf.random.experimental.stateless_split(seed, num=total_chunks)

    state = initial_state
    kernel_results = kernel.bootstrap_results(initial_state)
    retained_chunks: list[StockpilingState] = []
    chunk_summaries: list[StockpilingChunkSummary] = []

    chunk_idx = 0

    def run_phase(num_steps: int, retain: bool) -> None:
        """Run either the burn-in or retained phase chunk by chunk."""
        nonlocal chunk_idx, state, kernel_results

        remaining = int(num_steps)
        while remaining > 0:
            chunk_len = min(stockpiling_config.chunk_size, remaining)
            samples, raw_trace, kernel_results = _run_chunk(
                current_state=state,
                previous_kernel_results=kernel_results,
                num_steps=tf.constant(chunk_len, dtype=tf.int32),
                chunk_seed=chunk_seeds[chunk_idx],
            )

            trace = _raw_trace_to_dataclass(raw_trace)
            summary = report_chunk_progress(
                trace=trace,
                chunk_idx=chunk_idx + 1,
                total_chunks=total_chunks,
            )
            chunk_summaries.append(summary)

            state = _last_state(samples)
            if retain:
                retained_chunks.append(samples)

            remaining -= chunk_len
            chunk_idx += 1

    run_phase(stockpiling_config.num_burnin_steps, retain=False)
    run_phase(stockpiling_config.num_results, retain=True)

    report_run_summary(chunk_summaries)

    samples = _concat_sample_chunks(retained_chunks)
    n_saved = int(samples.z_beta.shape[0])
    mcmc_summary = build_mcmc_summary(
        summaries=chunk_summaries,
        n_saved=n_saved,
    )

    return StockpilingRunResult(
        samples=samples,
        chunk_summaries=chunk_summaries,
        mcmc_summary=mcmc_summary,
    )


def summarize_samples(samples: StockpilingState) -> dict[str, tf.Tensor]:
    """Summarize retained draws by posterior means on the constrained scale."""
    theta = unconstrained_to_theta(
        {
            "z_beta": samples.z_beta,
            "z_alpha": samples.z_alpha,
            "z_v": samples.z_v,
            "z_fc": samples.z_fc,
            "z_u_scale": samples.z_u_scale,
        }
    )

    return {
        "beta": tf.reduce_mean(theta["beta"], axis=0),
        "alpha": tf.reduce_mean(theta["alpha"], axis=0),
        "v": tf.reduce_mean(theta["v"], axis=0),
        "fc": tf.reduce_mean(theta["fc"], axis=0),
        "u_scale": tf.reduce_mean(theta["u_scale"], axis=0),
    }
