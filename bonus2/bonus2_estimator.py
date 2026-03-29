"""Run MCMC estimation for the Bonus Q2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from bonus2 import bonus2_diagnostics as diagnostics
from bonus2 import bonus2_input_validation as validate_input
from bonus2 import bonus2_model as model
from bonus2 import bonus2_posterior as posterior_lib
from bonus2 import bonus2_updates as updates


@dataclass(frozen=True)
class Bonus2SamplerConfig:
    """Store MCMC run lengths and proposal scales."""

    num_results: int
    num_burnin_steps: int
    chunk_size: int

    k_beta_intercept: float
    k_beta_habit: float
    k_beta_peer: float
    k_beta_weekend: float
    k_a: float
    k_b: float


class Bonus2State(NamedTuple):
    """Store the full Bonus Q2 chain state."""

    z_beta_intercept_j: tf.Tensor
    z_beta_habit_j: tf.Tensor
    z_beta_peer_j: tf.Tensor
    z_beta_weekend_jw: tf.Tensor
    z_a_m: tf.Tensor
    z_b_m: tf.Tensor


class Bonus2HybridKernelResults(NamedTuple):
    """Store per-block acceptance indicators for one full transition."""

    beta_intercept_accept: tf.Tensor
    beta_habit_accept: tf.Tensor
    beta_peer_accept: tf.Tensor
    beta_weekend_accept: tf.Tensor
    a_accept: tf.Tensor
    b_accept: tf.Tensor


class Bonus2HybridKernel(tfp.mcmc.TransitionKernel):
    """Run one full Bonus Q2 transition with six RW-MH block updates."""

    def __init__(
        self,
        posterior: posterior_lib.Bonus2PosteriorTF,
        config: Bonus2SamplerConfig,
    ):
        self.posterior = posterior
        self.config = config

        self.k_beta_intercept = tf.constant(config.k_beta_intercept, dtype=tf.float64)
        self.k_beta_habit = tf.constant(config.k_beta_habit, dtype=tf.float64)
        self.k_beta_peer = tf.constant(config.k_beta_peer, dtype=tf.float64)
        self.k_beta_weekend = tf.constant(config.k_beta_weekend, dtype=tf.float64)
        self.k_a = tf.constant(config.k_a, dtype=tf.float64)
        self.k_b = tf.constant(config.k_b, dtype=tf.float64)

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "posterior": self.posterior,
            "config": self.config,
        }

    def bootstrap_results(
        self,
        init_state: Bonus2State,
    ) -> Bonus2HybridKernelResults:
        """Build initial kernel diagnostics."""
        del init_state
        zero = tf.constant(0.0, dtype=tf.float64)
        return Bonus2HybridKernelResults(
            beta_intercept_accept=zero,
            beta_habit_accept=zero,
            beta_peer_accept=zero,
            beta_weekend_accept=zero,
            a_accept=zero,
            b_accept=zero,
        )

    def one_step(
        self,
        current_state: Bonus2State,
        previous_kernel_results: Bonus2HybridKernelResults,
        seed: tf.Tensor | None = None,
    ) -> tuple[Bonus2State, Bonus2HybridKernelResults]:
        """Run one full sequential block update."""
        del previous_kernel_results
        del seed

        z_beta_intercept_j, beta_intercept_accept = updates.beta_intercept_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=current_state.z_beta_intercept_j,
            z_beta_habit_j=current_state.z_beta_habit_j,
            z_beta_peer_j=current_state.z_beta_peer_j,
            z_beta_weekend_jw=current_state.z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
            k_beta_intercept=self.k_beta_intercept,
        )

        z_beta_habit_j, beta_habit_accept = updates.beta_habit_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=current_state.z_beta_habit_j,
            z_beta_peer_j=current_state.z_beta_peer_j,
            z_beta_weekend_jw=current_state.z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
            k_beta_habit=self.k_beta_habit,
        )

        z_beta_peer_j, beta_peer_accept = updates.beta_peer_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=current_state.z_beta_peer_j,
            z_beta_weekend_jw=current_state.z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
            k_beta_peer=self.k_beta_peer,
        )

        z_beta_weekend_jw, beta_weekend_accept = updates.beta_weekend_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=current_state.z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
            k_beta_weekend=self.k_beta_weekend,
        )

        z_a_m, a_accept = updates.a_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
            k_a=self.k_a,
        )

        z_b_m, b_accept = updates.b_one_step(
            posterior=self.posterior,
            z_beta_intercept_j=z_beta_intercept_j,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_beta_weekend_jw=z_beta_weekend_jw,
            z_a_m=z_a_m,
            z_b_m=current_state.z_b_m,
            k_b=self.k_b,
        )

        return (
            Bonus2State(
                z_beta_intercept_j=z_beta_intercept_j,
                z_beta_habit_j=z_beta_habit_j,
                z_beta_peer_j=z_beta_peer_j,
                z_beta_weekend_jw=z_beta_weekend_jw,
                z_a_m=z_a_m,
                z_b_m=z_b_m,
            ),
            Bonus2HybridKernelResults(
                beta_intercept_accept=beta_intercept_accept,
                beta_habit_accept=beta_habit_accept,
                beta_peer_accept=beta_peer_accept,
                beta_weekend_accept=beta_weekend_accept,
                a_accept=a_accept,
                b_accept=b_accept,
            ),
        )


def build_initial_state(
    num_markets: int,
    num_products: int,
    num_harmonics: int,
) -> Bonus2State:
    """Build the initial chain state with zero-filled parameter blocks."""
    z_beta_intercept_j = tf.zeros([num_products], dtype=tf.float64)
    z_beta_habit_j = tf.zeros([num_products], dtype=tf.float64)
    z_beta_peer_j = tf.zeros([num_products], dtype=tf.float64)
    z_beta_weekend_jw = tf.zeros([num_products, 2], dtype=tf.float64)
    z_a_m = tf.zeros([num_markets, num_harmonics], dtype=tf.float64)
    z_b_m = tf.zeros([num_markets, num_harmonics], dtype=tf.float64)

    return Bonus2State(
        z_beta_intercept_j=z_beta_intercept_j,
        z_beta_habit_j=z_beta_habit_j,
        z_beta_peer_j=z_beta_peer_j,
        z_beta_weekend_jw=z_beta_weekend_jw,
        z_a_m=z_a_m,
        z_b_m=z_b_m,
    )


def _num_chunks(total_steps: int, chunk_size: int) -> int:
    """Return the number of chunks required for a total step count."""
    return (total_steps + chunk_size - 1) // chunk_size


def _last_state(samples: Bonus2State) -> Bonus2State:
    """Extract the terminal draw from one sampled chunk."""
    return Bonus2State(
        z_beta_intercept_j=samples.z_beta_intercept_j[-1],
        z_beta_habit_j=samples.z_beta_habit_j[-1],
        z_beta_peer_j=samples.z_beta_peer_j[-1],
        z_beta_weekend_jw=samples.z_beta_weekend_jw[-1],
        z_a_m=samples.z_a_m[-1],
        z_b_m=samples.z_b_m[-1],
    )


def _concat_sample_chunks(sample_chunks: list[Bonus2State]) -> Bonus2State:
    """Concatenate retained sample chunks along the draw axis."""
    return Bonus2State(
        z_beta_intercept_j=tf.concat(
            [chunk.z_beta_intercept_j for chunk in sample_chunks],
            axis=0,
        ),
        z_beta_habit_j=tf.concat(
            [chunk.z_beta_habit_j for chunk in sample_chunks],
            axis=0,
        ),
        z_beta_peer_j=tf.concat(
            [chunk.z_beta_peer_j for chunk in sample_chunks],
            axis=0,
        ),
        z_beta_weekend_jw=tf.concat(
            [chunk.z_beta_weekend_jw for chunk in sample_chunks],
            axis=0,
        ),
        z_a_m=tf.concat(
            [chunk.z_a_m for chunk in sample_chunks],
            axis=0,
        ),
        z_b_m=tf.concat(
            [chunk.z_b_m for chunk in sample_chunks],
            axis=0,
        ),
    )


def _trace_fn(
    posterior: posterior_lib.Bonus2PosteriorTF,
    current_state: Bonus2State,
    kernel_results: Bonus2HybridKernelResults,
) -> dict[str, tf.Tensor]:
    """Build the per-draw trace recorded by sample_chain."""
    return {
        "joint_logpost": posterior.joint_logpost(
            z_beta_intercept_j=current_state.z_beta_intercept_j,
            z_beta_habit_j=current_state.z_beta_habit_j,
            z_beta_peer_j=current_state.z_beta_peer_j,
            z_beta_weekend_jw=current_state.z_beta_weekend_jw,
            z_a_m=current_state.z_a_m,
            z_b_m=current_state.z_b_m,
        ),
        "beta_intercept_accept": kernel_results.beta_intercept_accept,
        "beta_habit_accept": kernel_results.beta_habit_accept,
        "beta_peer_accept": kernel_results.beta_peer_accept,
        "beta_weekend_accept": kernel_results.beta_weekend_accept,
        "a_accept": kernel_results.a_accept,
        "b_accept": kernel_results.b_accept,
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def _run_chunk(
    kernel: Bonus2HybridKernel,
    current_state: Bonus2State,
    previous_kernel_results: Bonus2HybridKernelResults,
    num_steps: tf.Tensor,
) -> tuple[Bonus2State, dict[str, tf.Tensor], Bonus2HybridKernelResults]:
    """Run one compiled sample_chain chunk."""
    samples, trace, final_kernel_results = tfp.mcmc.sample_chain(
        num_results=num_steps,
        current_state=current_state,
        kernel=kernel,
        previous_kernel_results=previous_kernel_results,
        trace_fn=lambda state, results: _trace_fn(kernel.posterior, state, results),
        return_final_kernel_results=True,
    )
    return samples, trace, final_kernel_results


def run_chain(
    y_mit: tf.Tensor,
    delta_mj: tf.Tensor,
    is_weekend_t: tf.Tensor,
    season_sin_kt: tf.Tensor,
    season_cos_kt: tf.Tensor,
    neighbors_m,
    lookback: int,
    decay: float,
    posterior_config: posterior_lib.Bonus2PosteriorConfig,
    sampler_config: Bonus2SamplerConfig,
) -> tuple[Bonus2State, list[diagnostics.Bonus2ChunkSummary]]:
    """Run the full Bonus Q2 MCMC chain."""
    y_mit = tf.convert_to_tensor(y_mit, dtype=tf.int32)
    delta_mj = tf.convert_to_tensor(delta_mj, dtype=tf.float64)
    is_weekend_t = tf.convert_to_tensor(is_weekend_t, dtype=tf.int32)
    season_sin_kt = tf.convert_to_tensor(season_sin_kt, dtype=tf.float64)
    season_cos_kt = tf.convert_to_tensor(season_cos_kt, dtype=tf.float64)

    num_markets = int(tf.shape(y_mit)[0].numpy())
    num_consumers = int(tf.shape(y_mit)[1].numpy())
    num_products = int(tf.shape(delta_mj)[1].numpy())
    num_harmonics = int(tf.shape(season_sin_kt)[0].numpy())

    peer_adj_m = model.build_peer_adjacency(
        neighbors_m=neighbors_m,
        n_consumers=num_consumers,
    )
    h_mntj, p_mntj = model.build_deterministic_states(
        y_mit=y_mit,
        n_products=tf.constant(num_products, dtype=tf.int32),
        peer_adj_m=peer_adj_m,
        lookback=tf.constant(lookback, dtype=tf.int32),
        decay=tf.constant(decay, dtype=tf.float64),
    )

    validate_input.run_chain_validate_input(
        y_mit=y_mit,
        delta_mj=delta_mj,
        is_weekend_t=is_weekend_t,
        season_sin_kt=season_sin_kt,
        season_cos_kt=season_cos_kt,
        h_mntj=h_mntj,
        p_mntj=p_mntj,
        posterior_config=posterior_config,
        sampler_config=sampler_config,
    )

    posterior = posterior_lib.Bonus2PosteriorTF(
        config=posterior_config,
        inputs={
            "y_mit": y_mit,
            "delta_mj": delta_mj,
            "is_weekend_t": is_weekend_t,
            "season_sin_kt": season_sin_kt,
            "season_cos_kt": season_cos_kt,
            "h_mntj": h_mntj,
            "p_mntj": p_mntj,
        },
    )

    current_state = build_initial_state(
        num_markets=num_markets,
        num_products=num_products,
        num_harmonics=num_harmonics,
    )
    kernel = Bonus2HybridKernel(
        posterior=posterior,
        config=sampler_config,
    )
    kernel_results = kernel.bootstrap_results(current_state)

    burnin_remaining = sampler_config.num_burnin_steps
    while burnin_remaining > 0:
        num_steps = min(sampler_config.chunk_size, burnin_remaining)
        samples, _, kernel_results = _run_chunk(
            kernel=kernel,
            current_state=current_state,
            previous_kernel_results=kernel_results,
            num_steps=tf.constant(num_steps, dtype=tf.int32),
        )
        current_state = _last_state(samples)
        burnin_remaining -= num_steps

    retained_chunks: list[Bonus2State] = []
    summaries: list[diagnostics.Bonus2ChunkSummary] = []

    num_result_chunks = _num_chunks(
        total_steps=sampler_config.num_results,
        chunk_size=sampler_config.chunk_size,
    )

    result_remaining = sampler_config.num_results
    chunk_idx = 0
    while result_remaining > 0:
        num_steps = min(sampler_config.chunk_size, result_remaining)
        samples, trace, kernel_results = _run_chunk(
            kernel=kernel,
            current_state=current_state,
            previous_kernel_results=kernel_results,
            num_steps=tf.constant(num_steps, dtype=tf.int32),
        )
        current_state = _last_state(samples)
        retained_chunks.append(samples)

        chunk_trace = diagnostics.build_chunk_trace(
            z_beta_intercept_j=samples.z_beta_intercept_j,
            z_beta_habit_j=samples.z_beta_habit_j,
            z_beta_peer_j=samples.z_beta_peer_j,
            z_beta_weekend_jw=samples.z_beta_weekend_jw,
            z_a_m=samples.z_a_m,
            z_b_m=samples.z_b_m,
            joint_logpost=trace["joint_logpost"],
            beta_intercept_accept=trace["beta_intercept_accept"],
            beta_habit_accept=trace["beta_habit_accept"],
            beta_peer_accept=trace["beta_peer_accept"],
            beta_weekend_accept=trace["beta_weekend_accept"],
            a_accept=trace["a_accept"],
            b_accept=trace["b_accept"],
        )
        summaries.append(
            diagnostics.report_chunk_progress(
                trace=chunk_trace,
                chunk_idx=chunk_idx + 1,
                total_chunks=num_result_chunks,
            )
        )

        result_remaining -= num_steps
        chunk_idx += 1

    diagnostics.report_run_summary(summaries)
    return _concat_sample_chunks(retained_chunks), summaries


def summarize_samples(samples: Bonus2State) -> dict[str, tf.Tensor]:
    """Summarize retained samples by posterior means."""
    return {
        "beta_intercept_j": tf.reduce_mean(samples.z_beta_intercept_j, axis=0),
        "beta_habit_j": tf.reduce_mean(samples.z_beta_habit_j, axis=0),
        "beta_peer_j": tf.reduce_mean(samples.z_beta_peer_j, axis=0),
        "beta_weekend_jw": tf.reduce_mean(samples.z_beta_weekend_jw, axis=0),
        "a_m": tf.reduce_mean(samples.z_a_m, axis=0),
        "b_m": tf.reduce_mean(samples.z_b_m, axis=0),
    }
