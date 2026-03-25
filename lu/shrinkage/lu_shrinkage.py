from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from lu.lu_gibbs import gibbs_gamma, gibbs_phi
from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF

tfmcmc = tfp.mcmc
tfprandom = tfp.random


@dataclass(frozen=True)
class LuShrinkageConfig:
    num_results: int
    num_burnin_steps: int
    rw_scale: float


class LuShrinkageState(NamedTuple):
    beta_p: tf.Tensor
    beta_w: tf.Tensor
    r: tf.Tensor
    E_bar: tf.Tensor
    njt: tf.Tensor
    gamma: tf.Tensor
    phi: tf.Tensor


class LuHybridKernelResults(NamedTuple):
    continuous_kernel_results: object


def build_initial_state(
    pjt: tf.Tensor,
    posterior: LuPosteriorTF,
) -> LuShrinkageState:
    T = tf.shape(pjt)[0]
    J = tf.shape(pjt)[1]
    phi0 = posterior.a_phi / (posterior.a_phi + posterior.b_phi)

    return LuShrinkageState(
        beta_p=tf.constant(0.0, dtype=posterior.dtype),
        beta_w=tf.constant(0.0, dtype=posterior.dtype),
        r=tf.constant(0.0, dtype=posterior.dtype),
        E_bar=tf.fill(tf.stack([T]), posterior.E_bar_mean),
        njt=tf.zeros(tf.stack([T, J]), dtype=posterior.dtype),
        gamma=tf.zeros(tf.stack([T, J]), dtype=posterior.dtype),
        phi=tf.fill(tf.stack([T]), phi0),
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

    def _continuous_state(
        self,
        state: LuShrinkageState,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            state.beta_p,
            state.beta_w,
            state.r,
            state.E_bar,
            state.njt,
        )

    def _target_log_prob_fn(
        self,
        gamma: tf.Tensor,
    ):
        def target_log_prob_fn(
            beta_p: tf.Tensor,
            beta_w: tf.Tensor,
            r: tf.Tensor,
            E_bar: tf.Tensor,
            njt: tf.Tensor,
        ) -> tf.Tensor:
            return self._posterior.conditional_continuous_logpost(
                qjt=self._qjt,
                q0t=self._q0t,
                pjt=self._pjt,
                wjt=self._wjt,
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar=E_bar,
                njt=njt,
                gamma=gamma,
            )

        return target_log_prob_fn

    def _continuous_kernel(
        self,
        gamma: tf.Tensor,
    ) -> tfmcmc.RandomWalkMetropolis:
        return tfmcmc.RandomWalkMetropolis(
            target_log_prob_fn=self._target_log_prob_fn(gamma),
            new_state_fn=tfmcmc.random_walk_normal_fn(scale=self._config.rw_scale),
        )

    def bootstrap_results(
        self,
        current_state: LuShrinkageState,
    ) -> LuHybridKernelResults:
        continuous_kernel = self._continuous_kernel(current_state.gamma)
        continuous_kernel_results = continuous_kernel.bootstrap_results(
            self._continuous_state(current_state)
        )
        return LuHybridKernelResults(
            continuous_kernel_results=continuous_kernel_results,
        )

    def one_step(
        self,
        current_state: LuShrinkageState,
        previous_kernel_results: LuHybridKernelResults,
        seed: tf.Tensor | None = None,
    ) -> tuple[LuShrinkageState, LuHybridKernelResults]:
        seed = tfprandom.sanitize_seed(seed, salt="lu_hybrid_kernel")
        seeds = tfprandom.split_seed(seed, n=3)
        continuous_seed = seeds[0]
        gamma_seed = seeds[1]
        phi_seed = seeds[2]

        continuous_kernel = self._continuous_kernel(current_state.gamma)
        new_continuous_state, _ = continuous_kernel.one_step(
            current_state=self._continuous_state(current_state),
            previous_kernel_results=previous_kernel_results.continuous_kernel_results,
            seed=continuous_seed,
        )

        new_beta_p = new_continuous_state[0]
        new_beta_w = new_continuous_state[1]
        new_r = new_continuous_state[2]
        new_E_bar = new_continuous_state[3]
        new_njt = new_continuous_state[4]

        new_gamma = gibbs_gamma(
            njt=new_njt,
            phi=current_state.phi,
            T0_sq=self._posterior.T0_sq,
            T1_sq=self._posterior.T1_sq,
            eps=self._posterior.eps,
            seed=gamma_seed,
        )
        new_phi = gibbs_phi(
            gamma=new_gamma,
            a_phi=self._posterior.a_phi,
            b_phi=self._posterior.b_phi,
            eps=self._posterior.eps,
            seed=phi_seed,
        )

        new_state = LuShrinkageState(
            beta_p=new_beta_p,
            beta_w=new_beta_w,
            r=new_r,
            E_bar=new_E_bar,
            njt=new_njt,
            gamma=new_gamma,
            phi=new_phi,
        )

        next_continuous_kernel = self._continuous_kernel(new_state.gamma)
        next_continuous_kernel_results = next_continuous_kernel.bootstrap_results(
            self._continuous_state(new_state)
        )

        return new_state, LuHybridKernelResults(
            continuous_kernel_results=next_continuous_kernel_results,
        )

    def copy(self, **kwargs):
        parameters = dict(self.parameters)
        parameters.update(kwargs)
        return type(self)(**parameters)


def run_chain(
    pjt: tf.Tensor,
    wjt: tf.Tensor,
    qjt: tf.Tensor,
    q0t: tf.Tensor,
    posterior_config: LuPosteriorConfig,
    shrinkage_config: LuShrinkageConfig,
    seed: tf.Tensor | int | None = None,
) -> LuShrinkageState:
    posterior = LuPosteriorTF(posterior_config)
    initial_state = build_initial_state(pjt=pjt, posterior=posterior)
    kernel = LuHybridKernel(
        posterior=posterior,
        qjt=qjt,
        q0t=q0t,
        pjt=pjt,
        wjt=wjt,
        config=shrinkage_config,
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run():
        return tfmcmc.sample_chain(
            num_results=shrinkage_config.num_results,
            num_burnin_steps=shrinkage_config.num_burnin_steps,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=None,
            seed=seed,
        )

    return _run()


def summarize_samples(
    samples: LuShrinkageState,
) -> dict[str, tf.Tensor]:
    beta_p_hat = tf.reduce_mean(samples.beta_p, axis=0)
    beta_w_hat = tf.reduce_mean(samples.beta_w, axis=0)
    sigma_hat = tf.reduce_mean(tf.exp(samples.r), axis=0)

    E_bar_hat = tf.reduce_mean(samples.E_bar, axis=0)
    njt_hat = tf.reduce_mean(samples.njt, axis=0)
    gamma_hat = tf.reduce_mean(samples.gamma, axis=0)
    phi_hat = tf.reduce_mean(samples.phi, axis=0)

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
        "phi_hat": phi_hat,
    }
