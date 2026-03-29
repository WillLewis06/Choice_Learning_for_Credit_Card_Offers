"""Unit tests for the choice-learn posterior."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.choice_learn.cl_posterior import (
    ChoiceLearnPosteriorConfig,
    ChoiceLearnPosteriorTF,
)
from lu_conftest import (
    assert_all_finite_tf,
    assert_prob_simplex_tf,
    normal_logpdf_tf,
    permute_TJ_tf,
    permute_vec_tf,
)

DTYPE = tf.float64
ATOL = 1e-10
RTOL = 0.0
PROB_ATOL = 1e-12


def _assert_near(a: tf.Tensor, b: tf.Tensor, atol: float = ATOL) -> None:
    """Assert tensor near-equality."""
    tf.debugging.assert_near(a, b, atol=atol, rtol=RTOL)


def _make_tiny_inputs() -> dict:
    """
    Construct a canonical tiny (T=2, J=3) panel used throughout this test module.

    Keys
    - T, J: ints
    - delta_cl, qjt: (T, J) tf.float64
    - q0t, E_bar: (T,) tf.float64
    - alpha: scalar tf.float64
    - njt, gamma: (T, J) tf.float64
    """
    T, J = 2, 3

    delta_cl = tf.constant([[0.2, -0.1, 0.05], [0.0, 0.3, -0.2]], dtype=DTYPE)

    qjt = tf.constant([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=DTYPE)
    q0t = tf.constant([20.0, 15.0], dtype=DTYPE)

    alpha = tf.constant(1.2, dtype=DTYPE)
    E_bar = tf.constant([0.1, -0.2], dtype=DTYPE)

    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=DTYPE)
    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=DTYPE)

    return {
        "T": T,
        "J": J,
        "delta_cl": delta_cl,
        "qjt": qjt,
        "q0t": q0t,
        "alpha": alpha,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
    }


_POSTERIOR_CONFIG = ChoiceLearnPosteriorConfig(
    alpha_mean=1.0,
    alpha_var=1.0,
    E_bar_mean=0.0,
    E_bar_var=1.0,
    T0_sq=0.01,
    T1_sq=1.0,
    a_phi=2.0,
    b_phi=2.0,
)

posterior = ChoiceLearnPosteriorTF(config=_POSTERIOR_CONFIG)
tiny_inputs = _make_tiny_inputs()


# -----------------------------------------------------------------------------
# 1) Deterministic helpers
# -----------------------------------------------------------------------------
def test_utilities_shape_and_identity():
    """_utilities returns the expected affine utility construction."""
    J = 4
    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=DTYPE)
    njt_t = tf.constant([0.1, -0.2, 0.0, 0.3], dtype=DTYPE)

    alpha = tf.constant(1.7, dtype=DTYPE)
    E_bar_t = tf.constant(0.7, dtype=DTYPE)

    utilities = posterior._utilities(
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )
    assert tuple(utilities.shape) == (J,)

    expected = alpha * delta_cl_t + E_bar_t + njt_t
    _assert_near(utilities, expected, atol=PROB_ATOL)

    utilities_shift = posterior._utilities(
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t + 3.0,
        njt=njt_t,
    )
    d0 = utilities - utilities[0]
    d1 = utilities_shift - utilities_shift[0]
    _assert_near(d0, d1, atol=PROB_ATOL)


# -----------------------------------------------------------------------------
# 2) Choice probabilities
# -----------------------------------------------------------------------------
def test_log_choice_probs_shapes_simplex_bounds():
    """_log_choice_probs returns valid inside/outside probabilities."""
    J = 5
    utilities_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=DTYPE)

    log_pj, log_p0 = posterior._log_choice_probs(utilities=utilities_t)
    sjt_t = tf.exp(log_pj)
    s0t = tf.exp(log_p0)

    assert tuple(sjt_t.shape) == (J,)
    assert s0t.shape == ()
    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)


def test_log_choice_probs_extreme_negative_utilities_outside_near_one():
    """Very negative inside utilities should push almost all mass to outside."""
    J = 4
    utilities_t = tf.constant([-50.0] * J, dtype=DTYPE)

    log_pj, log_p0 = posterior._log_choice_probs(utilities=utilities_t)
    sjt_t = tf.exp(log_pj)
    s0t = tf.exp(log_p0)

    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)
    assert float(s0t.numpy()) > 1.0 - 1e-10
    assert np.max(sjt_t.numpy()) < 1e-10


def test_log_choice_probs_extreme_positive_utilities_outside_near_zero():
    """Very positive inside utilities should push almost all mass to inside goods."""
    J = 4
    utilities_t = tf.constant([50.0] * J, dtype=DTYPE)

    log_pj, log_p0 = posterior._log_choice_probs(utilities=utilities_t)
    sjt_t = tf.exp(log_pj)
    s0t = tf.exp(log_p0)

    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)
    assert float(s0t.numpy()) < 1e-10
    assert abs(float(tf.reduce_sum(sjt_t).numpy()) - 1.0) < 1e-10


def test_log_choice_probs_monotone_under_inside_shift():
    """Uniformly increasing inside utilities lowers outside mass."""
    utilities_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=DTYPE)

    log_pj0, log_p00 = posterior._log_choice_probs(utilities=utilities_t)
    log_pj1, log_p01 = posterior._log_choice_probs(utilities=utilities_t + 1.0)

    sjt0, s00 = tf.exp(log_pj0), tf.exp(log_p00)
    sjt1, s01 = tf.exp(log_pj1), tf.exp(log_p01)

    assert float(s01.numpy()) < float(s00.numpy())
    assert float(tf.reduce_sum(sjt1).numpy()) > float(tf.reduce_sum(sjt0).numpy())

    assert_prob_simplex_tf(sjt0, s00, atol=PROB_ATOL)
    assert_prob_simplex_tf(sjt1, s01, atol=PROB_ATOL)


# -----------------------------------------------------------------------------
# 3) Likelihood
# -----------------------------------------------------------------------------
def test_log_likelihood_terms_one_market_finite_reasonable_inputs():
    """_log_likelihood_terms returns a finite scalar for one market."""
    t = 0
    ll = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"][t],
        q0t=tiny_inputs["q0t"][t],
        delta_cl=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=tiny_inputs["njt"][t],
    )
    assert ll.shape == ()
    assert_all_finite_tf(ll)


def test_log_likelihood_terms_outside_only_identity():
    """If qjt is zero, one-market likelihood reduces to q0t * log(p0)."""
    J = 4
    qjt_t = tf.zeros((J,), dtype=DTYPE)
    q0t_t = tf.constant(100.0, dtype=DTYPE)

    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=DTYPE)
    alpha = tf.constant(1.0, dtype=DTYPE)
    E_bar_t = tf.constant(0.0, dtype=DTYPE)
    njt_t = tf.zeros((J,), dtype=DTYPE)

    ll = posterior._log_likelihood_terms(
        qjt=qjt_t,
        q0t=q0t_t,
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )

    utilities_t = posterior._utilities(
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )
    _, log_p0 = posterior._log_choice_probs(utilities=utilities_t)

    expected = q0t_t * log_p0
    _assert_near(ll, expected)


def test_log_likelihood_terms_inside_only_identity():
    """If q0t is zero, one-market likelihood reduces to sum_j qjt * log(pj)."""
    J = 4
    qjt_t = tf.constant([10.0, 5.0, 1.0, 2.0], dtype=DTYPE)
    q0t_t = tf.constant(0.0, dtype=DTYPE)

    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=DTYPE)
    alpha = tf.constant(1.0, dtype=DTYPE)
    E_bar_t = tf.constant(0.0, dtype=DTYPE)
    njt_t = tf.zeros((J,), dtype=DTYPE)

    ll = posterior._log_likelihood_terms(
        qjt=qjt_t,
        q0t=q0t_t,
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )

    utilities_t = posterior._utilities(
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )
    log_pj, _ = posterior._log_choice_probs(utilities=utilities_t)

    expected = tf.reduce_sum(qjt_t * log_pj)
    _assert_near(ll, expected)


def test_log_likelihood_terms_batched_shape_and_matches_one_market_stack():
    """Batched _log_likelihood_terms returns one contribution per market."""
    ll_vec = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    assert tuple(ll_vec.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(ll_vec)

    stacked = tf.stack(
        [
            posterior._log_likelihood_terms(
                qjt=tiny_inputs["qjt"][t],
                q0t=tiny_inputs["q0t"][t],
                delta_cl=tiny_inputs["delta_cl"][t],
                alpha=tiny_inputs["alpha"],
                E_bar=tiny_inputs["E_bar"][t],
                njt=tiny_inputs["njt"][t],
            )
            for t in range(tiny_inputs["T"])
        ]
    )
    _assert_near(ll_vec, stacked)


def test_loglik_equals_sum_of_log_likelihood_terms():
    """loglik equals the sum of the batched market contributions."""
    ll = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    ll_terms = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    assert ll.shape == ()
    assert_all_finite_tf(ll)
    _assert_near(ll, tf.reduce_sum(ll_terms))


# -----------------------------------------------------------------------------
# 4) Priors
# -----------------------------------------------------------------------------
def test_alpha_logprior_matches_closed_form_at_mean():
    """alpha_logprior matches the Gaussian closed form at the prior mean."""
    alpha = posterior.alpha_mean
    lp = posterior.alpha_logprior(alpha=alpha)

    expected = normal_logpdf_tf(alpha, posterior.alpha_mean, posterior.alpha_var)
    _assert_near(lp, expected, atol=PROB_ATOL)


def test_alpha_logprior_symmetry_around_mean():
    """Gaussian alpha prior is symmetric around its mean."""
    d = tf.constant(0.37, dtype=DTYPE)
    a_m = posterior.alpha_mean

    lp1 = posterior.alpha_logprior(alpha=a_m + d)
    lp2 = posterior.alpha_logprior(alpha=a_m - d)
    _assert_near(lp1, lp2, atol=PROB_ATOL)


def test_E_bar_logprior_shape_and_closed_form():
    """E_bar_logprior returns one Gaussian contribution per market."""
    lp_t = posterior.E_bar_logprior(E_bar=tiny_inputs["E_bar"])
    assert tuple(lp_t.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(lp_t)

    expected = normal_logpdf_tf(
        tiny_inputs["E_bar"],
        posterior.E_bar_mean,
        posterior.E_bar_var,
    )
    _assert_near(lp_t, expected)


def test_njt_logprior_given_gamma_shape_and_finite():
    """njt_logprior_given_gamma returns one conditional Gaussian contribution per market."""
    lp_t = posterior.njt_logprior_given_gamma(
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )
    assert tuple(lp_t.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(lp_t)


def test_njt_logprior_given_gamma_gamma_selects_variance():
    """For large |njt|, slab variance gives higher density than spike variance."""
    T, J = 2, 6
    njt = tf.ones((T, J), dtype=DTYPE) * 2.0
    gamma = tf.concat(
        [tf.ones((1, J), dtype=DTYPE), tf.zeros((1, J), dtype=DTYPE)],
        axis=0,
    )

    lp = posterior.njt_logprior_given_gamma(njt=njt, gamma=gamma)
    assert float(lp[0].numpy()) > float(lp[1].numpy())


def test_collapsed_gamma_prior_depends_only_on_marketwise_active_counts():
    """collapsed_gamma_prior depends only on row sums, not exact active columns."""
    gamma_a = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=DTYPE)
    gamma_b = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=DTYPE)

    lp_a = posterior.collapsed_gamma_prior(gamma=gamma_a)
    lp_b = posterior.collapsed_gamma_prior(gamma=gamma_b)
    _assert_near(lp_a, lp_b)


# -----------------------------------------------------------------------------
# 5) Posterior composition
# -----------------------------------------------------------------------------
def test_alpha_block_logpost_equals_loglik_plus_alpha_prior():
    """alpha_block_logpost equals loglik plus alpha prior."""
    block_lp = posterior.alpha_block_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    expected = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    ) + posterior.alpha_logprior(alpha=tiny_inputs["alpha"])

    _assert_near(block_lp, expected)


def test_E_bar_block_logpost_equals_one_market_likelihood_plus_E_bar_prior():
    """E_bar_block_logpost equals one-market likelihood plus E_bar_t prior."""
    t = 0
    block_lp = posterior.E_bar_block_logpost(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        delta_cl_t=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )

    expected = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"][t],
        q0t=tiny_inputs["q0t"][t],
        delta_cl=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=tiny_inputs["njt"][t],
    ) + posterior.E_bar_logprior(E_bar=tiny_inputs["E_bar"][t])

    _assert_near(block_lp, expected)


def test_njt_block_logpost_equals_one_market_likelihood_plus_njt_prior():
    """njt_block_logpost equals one-market likelihood plus njt_t conditional prior."""
    t = 0
    block_lp = posterior.njt_block_logpost(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        delta_cl_t=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
        gamma_t=tiny_inputs["gamma"][t],
    )

    expected = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"][t],
        q0t=tiny_inputs["q0t"][t],
        delta_cl=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=tiny_inputs["njt"][t],
    ) + posterior.njt_logprior_given_gamma(
        njt=tiny_inputs["njt"][t],
        gamma=tiny_inputs["gamma"][t],
    )

    _assert_near(block_lp, expected)


def test_joint_logpost_equals_likelihood_plus_all_prior_components():
    """joint_logpost equals likelihood plus all prior terms."""
    lp = posterior.joint_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    expected = (
        posterior.loglik(
            qjt=tiny_inputs["qjt"],
            q0t=tiny_inputs["q0t"],
            delta_cl=tiny_inputs["delta_cl"],
            alpha=tiny_inputs["alpha"],
            E_bar=tiny_inputs["E_bar"],
            njt=tiny_inputs["njt"],
        )
        + posterior.alpha_logprior(alpha=tiny_inputs["alpha"])
        + tf.reduce_sum(posterior.E_bar_logprior(E_bar=tiny_inputs["E_bar"]))
        + tf.reduce_sum(
            posterior.njt_logprior_given_gamma(
                njt=tiny_inputs["njt"],
                gamma=tiny_inputs["gamma"],
            )
        )
        + posterior.collapsed_gamma_prior(gamma=tiny_inputs["gamma"])
    )

    _assert_near(lp, expected)


# -----------------------------------------------------------------------------
# 6) Numerical edge cases
# -----------------------------------------------------------------------------
def test_log_likelihood_terms_finite_when_probabilities_extremely_small():
    """Likelihood stays finite when one inside probability is extremely small."""
    J = 3
    qjt_t = tf.constant([10.0, 10.0, 10.0], dtype=DTYPE)
    q0t_t = tf.constant(10.0, dtype=DTYPE)

    delta_cl_t = tf.zeros((J,), dtype=DTYPE)
    alpha = tf.constant(0.0, dtype=DTYPE)
    E_bar_t = tf.constant(0.0, dtype=DTYPE)
    njt_t = tf.constant([-100.0, 0.0, 0.0], dtype=DTYPE)

    ll = posterior._log_likelihood_terms(
        qjt=qjt_t,
        q0t=q0t_t,
        delta_cl=delta_cl_t,
        alpha=alpha,
        E_bar=E_bar_t,
        njt=njt_t,
    )
    assert_all_finite_tf(ll)


# -----------------------------------------------------------------------------
# 7) Permutation invariance / equivariance (products within market)
# -----------------------------------------------------------------------------
def test_utilities_equivariant_under_product_permutation():
    """_utilities is equivariant under product permutation."""
    t = 0
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    utilities = posterior._utilities(
        delta_cl=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=tiny_inputs["njt"][t],
    )

    utilities_p = posterior._utilities(
        delta_cl=permute_vec_tf(tiny_inputs["delta_cl"][t], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=permute_vec_tf(tiny_inputs["njt"][t], perm),
    )

    _assert_near(utilities_p, permute_vec_tf(utilities, perm), atol=PROB_ATOL)


def test_log_choice_probs_equivariant_under_product_permutation():
    """Inside log-probabilities are equivariant and outside log-probability is invariant."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    utilities_t = tf.constant([0.2, -0.1, 0.3], dtype=DTYPE)

    log_pj, log_p0 = posterior._log_choice_probs(utilities=utilities_t)
    log_pj_p, log_p0_p = posterior._log_choice_probs(
        utilities=permute_vec_tf(utilities_t, perm)
    )

    _assert_near(log_p0_p, log_p0, atol=PROB_ATOL)
    _assert_near(log_pj_p, permute_vec_tf(log_pj, perm), atol=PROB_ATOL)


def test_log_likelihood_terms_invariant_under_product_permutation_one_market():
    """One-market likelihood contribution is invariant to product permutation."""
    t = 0
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    ll = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"][t],
        q0t=tiny_inputs["q0t"][t],
        delta_cl=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=tiny_inputs["njt"][t],
    )

    ll_p = posterior._log_likelihood_terms(
        qjt=permute_vec_tf(tiny_inputs["qjt"][t], perm),
        q0t=tiny_inputs["q0t"][t],
        delta_cl=permute_vec_tf(tiny_inputs["delta_cl"][t], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"][t],
        njt=permute_vec_tf(tiny_inputs["njt"][t], perm),
    )

    _assert_near(ll_p, ll)


def test_log_likelihood_terms_batched_invariant_under_product_permutation():
    """Batched market contributions are invariant to joint product permutation."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    ll = posterior._log_likelihood_terms(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    ll_p = posterior._log_likelihood_terms(
        qjt=permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        delta_cl=permute_TJ_tf(tiny_inputs["delta_cl"], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
    )

    _assert_near(ll_p, ll)


def test_njt_logprior_given_gamma_invariant_under_product_permutation():
    """njt_logprior_given_gamma is invariant to joint permutation of njt and gamma."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.njt_logprior_given_gamma(
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    lp_p = posterior.njt_logprior_given_gamma(
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=permute_TJ_tf(tiny_inputs["gamma"], perm),
    )

    _assert_near(lp_p, lp)


def test_collapsed_gamma_prior_invariant_under_product_permutation():
    """collapsed_gamma_prior is invariant to product permutation within each market."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.collapsed_gamma_prior(gamma=tiny_inputs["gamma"])
    lp_p = posterior.collapsed_gamma_prior(
        gamma=permute_TJ_tf(tiny_inputs["gamma"], perm)
    )

    _assert_near(lp_p, lp)


def test_joint_logpost_invariant_under_product_permutation():
    """joint_logpost is invariant to consistent product permutation across all product arrays."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.joint_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    lp_p = posterior.joint_logpost(
        qjt=permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        delta_cl=permute_TJ_tf(tiny_inputs["delta_cl"], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=permute_TJ_tf(tiny_inputs["gamma"], perm),
    )

    _assert_near(lp_p, lp)
