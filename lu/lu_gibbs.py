"""Collapsed Gibbs updates for the Lu shrinkage inclusion indicators.

In Lu's paper, the latent sparsity block is formed by both the inclusion
indicators Gamma and the market-level inclusion probabilities phi.
Conditional on phi_t, the paper samples each gamma_jt from a Bernoulli
distribution, and then samples phi_t separately from its Beta full
conditional.

This implementation uses a different but mathematically consistent
formulation for the Gibbs step. To keep the sparsity update compatible with
TensorFlow's compiled execution model (jit_compile=True), we do not sample
phi as a separate state variable. Instead, we integrate phi_t out of
the Beta-Bernoulli hierarchy and update gamma directly from the resulting
collapsed conditional distribution.

As a result, this file preserves the same spike-and-slab sparsity structure as
Lu's model, but it is not an exact line-by-line replication of the paper's
Appendix Gibbs routine. The key difference is that the paper samples the
(Gamma, phi) block explicitly, whereas this implementation performs a
collapsed Gibbs sweep over Gamma only.
"""

from __future__ import annotations

import tensorflow as tf


@tf.function(jit_compile=True, reduce_retracing=True)
def gibbs_gamma(
    njt: tf.Tensor,
    gamma: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Run one collapsed Gibbs sweep for the market-product inclusion indicators.

    This function updates the spike-and-slab inclusion indicators gamma_jt
    for all markets t and products j. In Lu's paper, the sparsity block
    is sampled by first drawing the market-level inclusion probability
    phi_t and then drawing gamma_jt | phi_t. Here we instead collapse
    phi_t out analytically and sample gamma_jt directly from the
    conditional distribution implied by the Beta-Bernoulli hierarchy.

    The arguments map to Lu's notation as follows:
    - njt corresponds to the market-product shock deviation eta_jt.
    - gamma corresponds to the inclusion indicator gamma_jt.
    - a_phi and b_phi are the Beta prior hyperparameters for phi_t.
    - T0_sq and T1_sq are the spike/slab variances.

    The collapsed conditional combines two ingredients:
    1. a Beta-Bernoulli predictive prior term obtained after integrating out
       phi_t; and
    2. the spike/slab Gaussian density contribution for njt_j.

    This reformulation was adopted because a single collapsed Gibbs update is
    simpler to express as a stateless, XLA-compatible compiled TensorFlow step
    than a two-stage sample phi then sample gamma | phi routine.

    Therefore, this function preserves Lu's sparsity logic, but it is not an
    exact reproduction of the paper's original Gibbs block.
    """

    # The compiled sweep runs sequentially over product columns. Markets are
    # updated in parallel within each column. We keep only gamma in the
    # live sampler state; the Beta prior on phi_t is folded into the
    # collapsed conditional update through leave-one-out active counts.
    J_int = tf.shape(gamma)[-1]
    J = tf.cast(J_int, gamma.dtype)

    # Precompute the spike and slab log-variance terms. These appear in every
    # conditional update for gamma_jt through the Gaussian spike/slab prior
    # on njt_j.
    log_T0_sq = tf.math.log(T0_sq)
    log_T1_sq = tf.math.log(T1_sq)

    # In the collapsed formulation, the prior contribution for each
    # gamma_jt depends on how many other indicators are active in the same
    # market. We therefore carry the row sums
    #
    #     s_t = sum_j gamma_jt
    #
    # throughout the sweep so that leave-one-out counts can be formed cheaply.
    s_init = tf.reduce_sum(gamma, axis=-1)

    def cond(j, seed_curr, gamma_curr, s_curr):
        """Continue until all product indices have been updated."""
        return j < J_int

    def body(j, seed_curr, gamma_curr, s_curr):
        """Update one product column gamma[:, j] in the collapsed sweep.

        For a fixed product index j, this step updates all markets in
        parallel. The update is conditional on:
        - the current shock deviations njt[:, j]; and
        - the current values of all other inclusion indicators in the same
          market, summarized by the leave-one-out active counts.
        """

        gamma_j = gamma_curr[:, j]
        njt_j = njt[:, j]

        # Form the leave-one-out active count
        #
        #     s_{-j,t} = sum_{k != j} gamma_kt.
        #
        # After collapsing out phi_t, this count is the sufficient prior
        # summary entering the conditional update for gamma_jt. In the
        # paper's original formulation, the analogous dependence would instead
        # flow through an explicitly sampled phi_t.
        s_minus_j = s_curr - gamma_j

        # Compute the two conditional log posterior kernels for gamma_jt = 0
        # and gamma_jt = 1.
        #
        # Each kernel has two parts:
        #
        # 1. A collapsed Beta-Bernoulli predictive prior term:
        #       gamma_jt = 1  ->  a_phi + s_{-j,t}
        #       gamma_jt = 0  ->  b_phi + (J - 1 - s_{-j,t})
        #
        #    This is the direct consequence of integrating out phi_t from
        #    the Beta prior / Bernoulli indicator hierarchy.
        #
        # 2. A Gaussian spike/slab prior term for njt_j:
        #       gamma_jt = 0  ->  spike variance T0_sq
        #       gamma_jt = 1  ->  slab  variance T1_sq
        #
        # We work only with log kernels up to proportionality because the
        # normalizing constants that are common across the two states cancel in
        # the Bernoulli probability calculation below.
        logp0 = (
            tf.math.log(b_phi + (J - 1.0 - s_minus_j))
            - 0.5 * tf.square(njt_j) / T0_sq
            - 0.5 * log_T0_sq
        )
        logp1 = (
            tf.math.log(a_phi + s_minus_j)
            - 0.5 * tf.square(njt_j) / T1_sq
            - 0.5 * log_T1_sq
        )

        # Convert the log-kernel difference into the collapsed conditional
        # probability
        #
        #   P(gamma_jt = 1 | njt_j, gamma_{-j,t}, a_phi, b_phi, T0_sq, T1_sq).
        #
        # This differs from Lu's original Gibbs step, which conditions on an
        # explicitly sampled phi_t and evaluates
        # P(gamma_jt = 1 | njt_j, phi_t, ... ) instead.
        prob1 = tf.math.sigmoid(logp1 - logp0)

        # Draw the updated indicator with stateless randomness so that the full
        # collapsed Gibbs sweep remains compatible with compiled execution.
        # The seed is threaded through the while-loop explicitly rather than
        # relying on Python-side or stateful RNG.
        seeds = tf.random.experimental.stateless_split(seed_curr, num=2)
        next_seed = seeds[0]
        draw_seed = seeds[1]
        u = tf.random.stateless_uniform(
            shape=tf.shape(prob1),
            seed=draw_seed,
            dtype=gamma.dtype,
        )
        new_gamma_j = tf.cast(u < prob1, gamma.dtype)

        # Write the updated column back into gamma and update the running
        # row sums incrementally:
        #
        #     s_t^{new} = s_{-j,t} + gamma_jt^{new}.
        #
        # This avoids recomputing full row sums after each product update and is
        # the efficient way to propagate the collapsed prior information through
        # the sequential Gibbs sweep.
        one_hot_j = tf.one_hot(j, depth=J_int, dtype=gamma.dtype)
        gamma_next = (
            gamma_curr * (1.0 - one_hot_j[None, :])
            + new_gamma_j[:, None] * one_hot_j[None, :]
        )
        s_next = s_minus_j + new_gamma_j

        return j + 1, next_seed, gamma_next, s_next

    # Sweep sequentially across all product columns. This must be sequential in
    # j because each conditional update depends on the current values of the
    # remaining indicators through the leave-one-out counts. We use
    # tf.while_loop rather than a Python loop so that the entire collapsed
    # Gibbs sweep stays inside the compiled TensorFlow graph.
    _, _, gamma_out, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(
            tf.constant(0, dtype=J_int.dtype),
            seed,
            gamma,
            s_init,
        ),
    )
    return gamma_out
