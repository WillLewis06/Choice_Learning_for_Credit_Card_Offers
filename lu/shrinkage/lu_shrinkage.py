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

tfmcmc = tfp.mcmc


@dataclass(frozen=True)
class LuShrinkageConfig:
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
    beta_p: tf.Tensor
    beta_w: tf.Tensor
    r: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    gamma: tf.Tensor


class LuHybridKernelResults(NamedTuple):
    beta_accept: tf.Tensor
    r_accept: tf.Tensor
    E_bar_accept: tf.Tensor
    njt_accept: tf.Tensor


def _normalize_seed(seed: tf.Tensor | int | None) -> tf.Tensor | None:
    if seed is None:
        return None

    seed_t = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed_t.shape.rank == 0:
        return tf.stack([seed_t, tf.constant(0, dtype=tf.int32)])
    return seed_t


def _fallback_seed(dtype: tf.dtypes.DType = tf.int32) -> tf.Tensor:
    maxval = tf.constant(2**31 - 1, dtype=dtype)
    return tf.random.uniform(
        shape=(2,),
        minval=0,
        maxval=maxval,
        dtype=dtype,
    )


def _split_seed(seed: tf.Tensor | None, num: int) -> list[tf.Tensor | None]:
    if num <= 0:
        return []

    if seed is None:
        return [None] * num

    seeds = tf.random.experimental.stateless_split(seed, num=num)
    return [seeds[i] for i in range(num)]


def _num_chunks(n_steps: int, chunk_size: int) -> int:
    if n_steps <= 0:
        return 0
    return (n_steps + chunk_size - 1) // chunk_size


def _last_state(samples: LuShrinkageState) -> LuShrinkageState:
    return LuShrinkageState(
        beta_p=samples.beta_p[-1],
        beta_w=samples.beta_w[-1],
        r=samples.r[-1],
        E_bar=samples.E_bar[-1],
        njt=samples.njt[-1],
        gamma=samples.gamma[-1],
    )


def _concat_sample_chunks(chunks: list[LuShrinkageState]) -> LuShrinkageState:
    if len(chunks) == 0:
        raise ValueError("No retained sample chunks were produced.")

    return LuShrinkageState(
        beta_p=tf.concat([chunk.beta_p for chunk in chunks], axis=0),
        beta_w=tf.concat([chunk.beta_w for chunk in chunks], axis=0),
        r=tf.concat([chunk.r for chunk in chunks], axis=0),
        E_bar=tf.concat([chunk.E_bar for chunk in chunks], axis=0),
        njt=tf.concat([chunk.njt for chunk in chunks], axis=0),
        gamma=tf.concat([chunk.gamma for chunk in chunks], axis=0),
    )


def _raw_trace_to_dataclass(raw_trace: dict[str, tf.Tensor]) -> LuChunkTrace:
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
    T = tf.shape(pjt)[0]
    J = tf.shape(pjt)[1]

    return LuShrinkageState(
        beta_p=tf.constant(0.0, dtype=posterior.dtype),
        beta_w=tf.constant(0.0, dtype=posterior.dtype),
        r=tf.constant(0.0, dtype=posterior.dtype),
        E_bar=tf.fill([T], posterior.E_bar_mean),
        njt=tf.zeros([T, J], dtype=posterior.dtype),
        gamma=tf.zeros([T, J], dtype=posterior.dtype),
    )


class LuHybridKernel(tfmcmc.TransitionKernel):
    def __init__(
        self,
        posterior: LuPosteriorTF,
        qjt: tf.Tensor,
        q0t: tf.Tensor,
        pjt: tf.Tensor,
        wjt: tf.Tensor,
        config: LuShrinkageConfig,
    ):
        self._posterior = posterior
        self._qjt = qjt
        self._q0t = q0t
        self._pjt = pjt
        self._wjt = wjt
        self._config = config
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
        return self._parameters

    @property
    def is_calibrated(self) -> bool:
        return True

    def bootstrap_results(
        self,
        current_state: LuShrinkageState,
    ) -> LuHybridKernelResults:
        zero = tf.constant(0.0, dtype=self._posterior.dtype)
        del current_state
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
        seed: tf.Tensor | None = None,
    ) -> tuple[LuShrinkageState, LuHybridKernelResults]:
        del previous_kernel_results

        if seed is None:
            seed = _fallback_seed(dtype=tf.int32)

        seeds = tf.random.experimental.stateless_split(seed, num=5)

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
            k_beta=tf.constant(self._config.k_beta, dtype=self._posterior.dtype),
            seed=seeds[0],
        )

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
            k_r=tf.constant(self._config.k_r, dtype=self._posterior.dtype),
            seed=seeds[1],
        )

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
            k_E_bar=tf.constant(self._config.k_E_bar, dtype=self._posterior.dtype),
            seed=seeds[2],
        )

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
            k_njt=tf.constant(self._config.k_njt, dtype=self._posterior.dtype),
            seed=seeds[3],
        )

        gamma_new = gibbs_gamma(
            njt=njt_new,
            gamma=current_state.gamma,
            a_phi=self._posterior.a_phi,
            b_phi=self._posterior.b_phi,
            T0_sq=self._posterior.T0_sq,
            T1_sq=self._posterior.T1_sq,
            seed=seeds[4],
        )

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
    seed: tf.Tensor | int | None = None,
) -> LuShrinkageState:
    run_chain_validate_input(
        pjt=pjt,
        wjt=wjt,
        qjt=qjt,
        q0t=q0t,
        posterior_config=posterior_config,
        shrinkage_config=shrinkage_config,
        seed=seed,
    )

    posterior = LuPosteriorTF(posterior_config)
    initial_state = build_initial_state(pjt=pjt, posterior=posterior)

    base_seed = _normalize_seed(seed)
    tuning_seed = None
    sampling_seed = None
    if base_seed is not None:
        tuning_seed, sampling_seed = _split_seed(base_seed, 2)

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
        return _make_trace(
            posterior=posterior,
            qjt=qjt,
            q0t=q0t,
            pjt=pjt,
            wjt=wjt,
            current_state=current_state,
            kernel_results=kernel_results,
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_chunk(
        current_state: LuShrinkageState,
        previous_kernel_results: LuHybridKernelResults,
        num_steps: tf.Tensor,
        chunk_seed: tf.Tensor | None,
    ):
        return tfmcmc.sample_chain(
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

    chunk_seeds = _split_seed(sampling_seed, total_chunks)

    chunk_idx = 0
    chunk_summaries: list[LuChunkSummary] = []
    retained_chunks: list[LuShrinkageState] = []

    burnin_remaining = tuned_config.num_burnin_steps
    while burnin_remaining > 0:
        this_chunk = min(tuned_config.chunk_size, burnin_remaining)
        chunk_seed = chunk_seeds[chunk_idx] if total_chunks > 0 else None

        chunk_samples, raw_trace, kernel_results = _run_chunk(
            current_state=state,
            previous_kernel_results=kernel_results,
            num_steps=tf.constant(this_chunk, dtype=tf.int32),
            chunk_seed=chunk_seed,
        )
        state = _last_state(chunk_samples)

        trace = _raw_trace_to_dataclass(raw_trace)
        summary = report_chunk_progress(
            trace=trace,
            chunk_idx=chunk_idx + 1,
            total_chunks=total_chunks,
        )
        chunk_summaries.append(summary)

        burnin_remaining -= this_chunk
        chunk_idx += 1

    results_remaining = tuned_config.num_results
    while results_remaining > 0:
        this_chunk = min(tuned_config.chunk_size, results_remaining)
        chunk_seed = chunk_seeds[chunk_idx] if total_chunks > 0 else None

        chunk_samples, raw_trace, kernel_results = _run_chunk(
            current_state=state,
            previous_kernel_results=kernel_results,
            num_steps=tf.constant(this_chunk, dtype=tf.int32),
            chunk_seed=chunk_seed,
        )
        state = _last_state(chunk_samples)
        retained_chunks.append(chunk_samples)

        trace = _raw_trace_to_dataclass(raw_trace)
        summary = report_chunk_progress(
            trace=trace,
            chunk_idx=chunk_idx + 1,
            total_chunks=total_chunks,
        )
        chunk_summaries.append(summary)

        results_remaining -= this_chunk
        chunk_idx += 1

    if len(chunk_summaries) > 0:
        report_run_summary(chunk_summaries)

    return _concat_sample_chunks(retained_chunks)


def summarize_samples(
    samples: LuShrinkageState,
) -> dict[str, tf.Tensor]:
    beta_p_hat = tf.reduce_mean(samples.beta_p, axis=0)
    beta_w_hat = tf.reduce_mean(samples.beta_w, axis=0)
    sigma_hat = tf.reduce_mean(tf.exp(samples.r), axis=0)

    E_bar_hat = tf.reduce_mean(samples.E_bar, axis=0)
    njt_hat = tf.reduce_mean(samples.njt, axis=0)
    gamma_hat = tf.reduce_mean(samples.gamma, axis=0)

    int_hat = tf.reduce_mean(E_bar_hat)
    E_full_hat = E_bar_hat[:, None] + njt_hat

    return {
        "beta_p_hat": beta_p_hat,
        "beta_w_hat": beta_w_hat,
        "sigma_hat": sigma_hat,
        "int_hat": int_hat,
        "E_hat": njt_hat,
        "E_full_hat": E_full_hat,
        "E_bar_hat": E_bar_hat,
        "njt_hat": njt_hat,
        "gamma_hat": gamma_hat,
    }
