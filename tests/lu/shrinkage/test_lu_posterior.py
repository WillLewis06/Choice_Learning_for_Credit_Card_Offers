"""
Unit tests for the refactored LuPosteriorTF.

This file targets the current posterior API:
- scalar total log-likelihood via loglik(...)
- separated continuous and collapsed discrete priors
- block-level log-posterior methods used by the sampler
- no explicit phi argument anywhere in the posterior interface
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF

DTYPE = tf.float64
ATOL = 1e-10
PROB_ATOL = 1e-12
RTOL = 0.0


def _tf(x) -> tf.Tensor:
    """Create a tf.float64 constant."""
    return tf.constant(x, dtype=DTYPE)


def _assert_near(a: tf.Tensor, b: tf.Tensor, atol: float = ATOL) -> None:
    """Assert tensor near-equality."""
    tf.debugging.assert_near(a, b, atol=atol, rtol=RTOL)


def _assert_all_finite_tf(x: tf.Tensor) -> None:
    """Assert all tensor entries are finite."""
    x_np = tf.convert_to_tensor(x).numpy()
    if not np.all(np.isfinite(x_np)):
        raise AssertionError("Tensor contains non-finite values.")


def _assert_prob_simplex_tf(
    sjt_t: tf.Tensor,
    s0t: tf.Tensor,
    atol: float = PROB_ATOL,
) -> None:
    """Assert inside and outside shares form a valid simplex."""
    sj_np = tf.convert_to_tensor(sjt_t).numpy()
    s0_np = float(tf.convert_to_tensor(s0t).numpy())

    if np.any(sj_np < -atol) or np.any(sj_np > 1.0 + atol):
        raise AssertionError("Inside shares are outside [0, 1].")
    if s0_np < -atol or s0_np > 1.0 + atol:
        raise AssertionError("Outside share is outside [0, 1].")

    total = float(np.sum(sj_np) + s0_np)
    if abs(total - 1.0) > atol:
        raise AssertionError(f"Shares do not sum to 1. Got total={total}.")


def _normal_logpdf_tf(x: tf.Tensor, mean: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
    """Closed-form scalar/vector normal log density."""
    two_pi = tf.constant(2.0 * np.pi, dtype=DTYPE)
    return -0.5 * (tf.math.log(two_pi) + tf.math.log(var) + tf.square(x - mean) / var)


def _permute_vec_tf(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    """Permute a length-J vector."""
    return tf.gather(x, perm, axis=0)


def _permute_TJ_tf(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    """Permute the product dimension of a (T, J) tensor."""
    return tf.gather(x, perm, axis=1)


def _default_global_params() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Common (beta_p, beta_w, r) used across multiple tests."""
    return _tf(-1.0), _tf(0.3), _tf(0.0)


def _make_tiny_inputs() -> dict:
    """Canonical tiny panel used across tests."""
    T, J = 2, 3

    pjt = _tf([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]])
    wjt = _tf([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]])

    qjt = _tf([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]])
    q0t = _tf([20.0, 15.0])

    E_bar = _tf([0.1, -0.2])
    njt = _tf([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]])
    gamma = _tf([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    return {
        "T": T,
        "J": J,
        "pjt": pjt,
        "wjt": wjt,
        "qjt": qjt,
        "q0t": q0t,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
    }


POSTERIOR_CONFIG = LuPosteriorConfig(
    n_draws=25,
    seed=123,
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

posterior = LuPosteriorTF(config=POSTERIOR_CONFIG)
tiny_inputs = _make_tiny_inputs()


# -----------------------------------------------------------------------------
# 1) Deterministic helpers
# -----------------------------------------------------------------------------
def test_mean_utility_jt_shape_and_identity():
    """_mean_utility_jt returns a length-J vector and matches its linear identity."""
    J = 4
    pjt_t = _tf([1.0, 1.2, 0.8, 0.9])
    wjt_t = _tf([0.4, 0.7, 0.6, 0.2])
    njt_t = _tf([0.1, -0.2, 0.0, 0.3])

    beta_p = _tf(-1.5)
    beta_w = _tf(0.2)
    E_bar_t = _tf(0.7)

    delta = posterior._mean_utility_jt(
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    assert tuple(delta.shape) == (J,)

    expected = beta_p * pjt_t + beta_w * wjt_t + E_bar_t + njt_t
    _assert_near(delta, expected)

    delta_shift = posterior._mean_utility_jt(
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t + _tf(2.0),
        njt_t=njt_t,
    )
    _assert_near(delta_shift - delta, tf.ones_like(delta) * _tf(2.0))


# -----------------------------------------------------------------------------
# 2) Choice probabilities
# -----------------------------------------------------------------------------
def test_choice_probs_t_shapes_simplex_bounds():
    """_choice_probs_t returns valid inside/outside shares on the simplex."""
    J = 5
    pjt_t = _tf([1.0, 1.1, 0.9, 1.3, 0.8])
    delta_t = _tf([0.2, -0.1, 0.0, 0.3, -0.2])
    r = _tf(0.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

    assert tuple(sjt_t.shape) == (J,)
    assert s0t.shape == ()
    _assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)


def test_choice_probs_t_extreme_negative_delta_outside_near_one():
    """Very negative inside utilities should push almost all mass to the outside good."""
    J = 4
    pjt_t = _tf([1.0, 1.2, 0.8, 1.1])
    delta_t = _tf([-50.0, -50.0, -50.0, -50.0])
    r = _tf(0.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

    _assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)
    assert float(s0t.numpy()) > 1.0 - ATOL
    assert float(tf.reduce_max(sjt_t).numpy()) < ATOL


def test_choice_probs_t_monotone_under_uniform_inside_shift():
    """Increasing all inside utilities increases total inside share and lowers outside share."""
    J = 5
    pjt_t = _tf([1.0, 1.1, 0.9, 1.3, 0.8])
    delta_t = _tf([0.2, -0.1, 0.0, 0.3, -0.2])
    r = _tf(0.0)

    sj0, s00 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    sj1, s01 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t + _tf(1.0), r=r)

    assert float(s01.numpy()) < float(s00.numpy())
    assert float(tf.reduce_sum(sj1).numpy()) > float(tf.reduce_sum(sj0).numpy())

    _assert_prob_simplex_tf(sj0, s00, atol=PROB_ATOL)
    _assert_prob_simplex_tf(sj1, s01, atol=PROB_ATOL)


def test_choice_probs_t_finite_for_large_r_moderate_prices():
    """_choice_probs_t remains finite when r is large but prices are moderate."""
    J = 5
    pjt_t = _tf([0.5, 0.8, 0.6, 0.7, 0.9])
    delta_t = _tf([0.0, 0.1, -0.1, 0.05, -0.05])
    r = _tf(5.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)

    _assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)
    _assert_all_finite_tf(sjt_t)
    _assert_all_finite_tf(s0t)


# -----------------------------------------------------------------------------
# 3) Likelihood
# -----------------------------------------------------------------------------
def test_market_loglik_impl_finite_reasonable_inputs():
    """_market_loglik_impl returns a finite scalar on a typical one-market input."""
    t = 0
    beta_p, beta_w, r = _default_global_params()

    ll = posterior._market_loglik_impl(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )

    assert ll.shape == ()
    _assert_all_finite_tf(ll)


def test_market_loglik_impl_outside_only_identity():
    """If qjt is zero, one-market log-likelihood reduces to q0t * log(s0t)."""
    J = 4
    qjt_t = tf.zeros((J,), dtype=DTYPE)
    q0t_t = _tf(100.0)

    pjt_t = _tf([1.0, 1.2, 0.8, 1.1])
    wjt_t = _tf([0.4, 0.7, 0.6, 0.2])
    beta_p = _tf(-1.0)
    beta_w = _tf(0.2)
    r = _tf(0.0)
    E_bar_t = _tf(0.0)
    njt_t = tf.zeros((J,), dtype=DTYPE)

    ll = posterior._market_loglik_impl(
        qjt_t=qjt_t,
        q0t_t=q0t_t,
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )

    delta_t = posterior._mean_utility_jt(
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    _, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    s0t_c = tf.clip_by_value(s0t, posterior.eps, 1.0)

    expected = q0t_t * tf.math.log(s0t_c)
    _assert_near(ll, expected)


def test_loglik_matches_sum_of_market_loglik_impl():
    """loglik equals the sum of one-market likelihood contributions."""
    beta_p, beta_w, r = _default_global_params()

    ll = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    assert ll.shape == ()
    _assert_all_finite_tf(ll)

    ll_sum = tf.add_n(
        [
            posterior._market_loglik_impl(
                qjt_t=tiny_inputs["qjt"][t],
                q0t_t=tiny_inputs["q0t"][t],
                pjt_t=tiny_inputs["pjt"][t],
                wjt_t=tiny_inputs["wjt"][t],
                beta_p=beta_p,
                beta_w=beta_w,
                r=r,
                E_bar_t=tiny_inputs["E_bar"][t],
                njt_t=tiny_inputs["njt"][t],
            )
            for t in range(tiny_inputs["T"])
        ]
    )
    _assert_near(ll, ll_sum)


def test_market_loglik_impl_finite_when_shares_tiny_due_to_clipping():
    """_market_loglik_impl remains finite when an inside share is extremely small."""
    J = 3
    qjt_t = _tf([10.0, 10.0, 10.0])
    q0t_t = _tf(10.0)

    pjt_t = _tf([1.0, 1.0, 1.0])
    wjt_t = tf.zeros((J,), dtype=DTYPE)

    beta_p = _tf(0.0)
    beta_w = _tf(0.0)
    r = _tf(0.0)
    E_bar_t = _tf(0.0)
    njt_t = _tf([-100.0, 0.0, 0.0])

    ll = posterior._market_loglik_impl(
        qjt_t=qjt_t,
        q0t_t=q0t_t,
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )

    _assert_all_finite_tf(ll)


# -----------------------------------------------------------------------------
# 4) Priors
# -----------------------------------------------------------------------------
def test_logprior_global_matches_closed_form_at_means():
    """logprior_global matches the sum of independent Gaussian log-densities."""
    beta_p = posterior.beta_p_mean
    beta_w = posterior.beta_w_mean
    r = posterior.r_mean

    lp = posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)

    expected = (
        _normal_logpdf_tf(beta_p, posterior.beta_p_mean, posterior.beta_p_var)
        + _normal_logpdf_tf(beta_w, posterior.beta_w_mean, posterior.beta_w_var)
        + _normal_logpdf_tf(r, posterior.r_mean, posterior.r_var)
    )
    _assert_near(lp, expected, atol=PROB_ATOL)


def test_logprior_global_symmetry_around_mean():
    """A Gaussian prior is symmetric around its mean."""
    d = _tf(0.37)

    beta_p_m = posterior.beta_p_mean
    beta_w_m = posterior.beta_w_mean
    r_m = posterior.r_mean

    _assert_near(
        posterior.logprior_global(beta_p=beta_p_m + d, beta_w=beta_w_m, r=r_m),
        posterior.logprior_global(beta_p=beta_p_m - d, beta_w=beta_w_m, r=r_m),
        atol=PROB_ATOL,
    )
    _assert_near(
        posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m + d, r=r_m),
        posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m - d, r=r_m),
        atol=PROB_ATOL,
    )
    _assert_near(
        posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m, r=r_m + d),
        posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m, r=r_m - d),
        atol=PROB_ATOL,
    )


def test_logprior_E_bar_vec_shape_and_closed_form():
    """logprior_E_bar_vec returns (T,) and matches the closed-form Gaussian prior."""
    E_bar = tiny_inputs["E_bar"]

    lp = posterior.logprior_E_bar_vec(E_bar=E_bar)
    assert tuple(lp.shape) == (tiny_inputs["T"],)
    _assert_all_finite_tf(lp)

    expected = _normal_logpdf_tf(E_bar, posterior.E_bar_mean, posterior.E_bar_var)
    _assert_near(lp, expected)


def test_logprior_njt_given_gamma_vec_shape_and_finite():
    """logprior_njt_given_gamma_vec returns one contribution per market."""
    lp = posterior.logprior_njt_given_gamma_vec(
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    assert tuple(lp.shape) == (tiny_inputs["T"],)
    _assert_all_finite_tf(lp)


def test_logprior_njt_given_gamma_vec_gamma_selects_variance():
    """For large |njt|, slab variance should give higher density than spike variance."""
    T, J = 2, 6
    njt = tf.ones((T, J), dtype=DTYPE) * _tf(2.0)
    gamma = tf.concat(
        [tf.ones((1, J), dtype=DTYPE), tf.zeros((1, J), dtype=DTYPE)],
        axis=0,
    )

    lp = posterior.logprior_njt_given_gamma_vec(njt=njt, gamma=gamma)

    assert float(lp[0].numpy()) > float(lp[1].numpy())


def test_continuous_prior_matches_sum_of_vector_components():
    """continuous_prior equals sum_t logprior_E_bar_vec[t] + logprior_njt_given_gamma_vec[t]."""
    lp = posterior.continuous_prior(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    expected = tf.reduce_sum(
        posterior.logprior_E_bar_vec(E_bar=tiny_inputs["E_bar"])
        + posterior.logprior_njt_given_gamma_vec(
            njt=tiny_inputs["njt"],
            gamma=tiny_inputs["gamma"],
        )
    )
    _assert_near(lp, expected)


def test_collapsed_gamma_prior_depends_only_on_marketwise_active_counts():
    """collapsed_gamma_prior depends only on row sums of gamma, not which columns are active."""
    gamma_a = _tf([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    gamma_b = _tf([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    lp_a = posterior.collapsed_gamma_prior(gamma=gamma_a)
    lp_b = posterior.collapsed_gamma_prior(gamma=gamma_b)

    _assert_near(lp_a, lp_b)


# -----------------------------------------------------------------------------
# 5) Block posteriors and joint posterior
# -----------------------------------------------------------------------------
def test_beta_block_logpost_equals_loglik_plus_beta_priors():
    """beta_block_logpost equals loglik plus the beta_p and beta_w priors only."""
    beta_p, beta_w, r = _default_global_params()

    block_lp = posterior.beta_block_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    ll = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    expected = (
        ll
        + _normal_logpdf_tf(beta_p, posterior.beta_p_mean, posterior.beta_p_var)
        + _normal_logpdf_tf(beta_w, posterior.beta_w_mean, posterior.beta_w_var)
    )
    _assert_near(block_lp, expected)


def test_r_block_logpost_equals_loglik_plus_r_prior():
    """r_block_logpost equals loglik plus the prior on r."""
    beta_p, beta_w, r = _default_global_params()

    block_lp = posterior.r_block_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    ll = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    expected = ll + _normal_logpdf_tf(r, posterior.r_mean, posterior.r_var)
    _assert_near(block_lp, expected)


def test_E_bar_block_logpost_equals_market_loglik_plus_E_bar_prior():
    """E_bar_block_logpost equals one-market likelihood plus the E_bar_t prior."""
    t = 0
    beta_p, beta_w, r = _default_global_params()

    block_lp = posterior.E_bar_block_logpost(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )

    expected = posterior._market_loglik_impl(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    ) + posterior._logprior_E_bar_t(E_bar_t=tiny_inputs["E_bar"][t])

    _assert_near(block_lp, expected)


def test_njt_block_logpost_equals_market_loglik_plus_njt_prior():
    """njt_block_logpost equals one-market likelihood plus the njt_t conditional prior."""
    t = 0
    beta_p, beta_w, r = _default_global_params()

    block_lp = posterior.njt_block_logpost(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
        gamma_t=tiny_inputs["gamma"][t],
    )

    expected = posterior._market_loglik_impl(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    ) + posterior._logprior_njt_t_given_gamma_t(
        njt_t=tiny_inputs["njt"][t],
        gamma_t=tiny_inputs["gamma"][t],
    )

    _assert_near(block_lp, expected)


def test_joint_logpost_equals_sum_of_all_components():
    """joint_logpost equals likelihood + global prior + continuous prior + collapsed gamma prior."""
    beta_p, beta_w, r = _default_global_params()

    lp = posterior.joint_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )

    expected = (
        posterior.loglik(
            qjt=tiny_inputs["qjt"],
            q0t=tiny_inputs["q0t"],
            pjt=tiny_inputs["pjt"],
            wjt=tiny_inputs["wjt"],
            beta_p=beta_p,
            beta_w=beta_w,
            r=r,
            E_bar=tiny_inputs["E_bar"],
            njt=tiny_inputs["njt"],
        )
        + posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)
        + posterior.continuous_prior(
            E_bar=tiny_inputs["E_bar"],
            njt=tiny_inputs["njt"],
            gamma=tiny_inputs["gamma"],
        )
        + posterior.collapsed_gamma_prior(gamma=tiny_inputs["gamma"])
    )

    _assert_near(lp, expected)


# -----------------------------------------------------------------------------
# 6) Permutation invariance / equivariance
# -----------------------------------------------------------------------------
def test_mean_utility_equivariant_under_product_permutation():
    """_mean_utility_jt is equivariant under product reordering."""
    t = 0
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, _ = _default_global_params()

    p = tiny_inputs["pjt"][t]
    w = tiny_inputs["wjt"][t]
    n = tiny_inputs["njt"][t]
    E_bar_t = tiny_inputs["E_bar"][t]

    delta = posterior._mean_utility_jt(
        pjt_t=p,
        wjt_t=w,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=n,
    )
    delta_perm = posterior._mean_utility_jt(
        pjt_t=_permute_vec_tf(p, perm),
        wjt_t=_permute_vec_tf(w, perm),
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=_permute_vec_tf(n, perm),
    )

    _assert_near(delta_perm, _permute_vec_tf(delta, perm))


def test_choice_probs_equivariant_under_product_permutation():
    """_choice_probs_t is equivariant in inside shares and invariant in outside share."""
    t = 0
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    p = tiny_inputs["pjt"][t]
    w = tiny_inputs["wjt"][t]
    n = tiny_inputs["njt"][t]
    E_bar_t = tiny_inputs["E_bar"][t]

    delta = posterior._mean_utility_jt(
        pjt_t=p,
        wjt_t=w,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=n,
    )
    sj, s0 = posterior._choice_probs_t(pjt_t=p, delta_t=delta, r=r)

    sj_p, s0_p = posterior._choice_probs_t(
        pjt_t=_permute_vec_tf(p, perm),
        delta_t=_permute_vec_tf(delta, perm),
        r=r,
    )

    _assert_near(s0_p, s0, atol=PROB_ATOL)
    _assert_near(sj_p, _permute_vec_tf(sj, perm), atol=PROB_ATOL)


def test_market_loglik_impl_invariant_under_product_permutation():
    """_market_loglik_impl is invariant to product reordering within a market."""
    t = 0
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    ll = posterior._market_loglik_impl(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=tiny_inputs["pjt"][t],
        wjt_t=tiny_inputs["wjt"][t],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )
    ll_p = posterior._market_loglik_impl(
        qjt_t=_permute_vec_tf(tiny_inputs["qjt"][t], perm),
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=_permute_vec_tf(tiny_inputs["pjt"][t], perm),
        wjt_t=_permute_vec_tf(tiny_inputs["wjt"][t], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=_permute_vec_tf(tiny_inputs["njt"][t], perm),
    )

    _assert_near(ll_p, ll)


def test_loglik_invariant_under_product_permutation():
    """loglik is invariant to consistent product permutation across markets."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    ll = posterior.loglik(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    ll_p = posterior.loglik(
        qjt=_permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=_permute_TJ_tf(tiny_inputs["pjt"], perm),
        wjt=_permute_TJ_tf(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ_tf(tiny_inputs["njt"], perm),
    )

    _assert_near(ll_p, ll)


def test_logprior_njt_given_gamma_vec_invariant_under_product_permutation():
    """logprior_njt_given_gamma_vec is invariant to permuting njt and gamma jointly."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.logprior_njt_given_gamma_vec(
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )
    lp_p = posterior.logprior_njt_given_gamma_vec(
        njt=_permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=_permute_TJ_tf(tiny_inputs["gamma"], perm),
    )

    _assert_near(lp_p, lp)


def test_collapsed_gamma_prior_invariant_under_product_permutation():
    """collapsed_gamma_prior is invariant to product permutation within each market."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.collapsed_gamma_prior(gamma=tiny_inputs["gamma"])
    lp_p = posterior.collapsed_gamma_prior(
        gamma=_permute_TJ_tf(tiny_inputs["gamma"], perm)
    )

    _assert_near(lp_p, lp)


def test_joint_logpost_invariant_under_product_permutation():
    """joint_logpost is invariant to consistent product permutation across all product-level arrays."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    lp = posterior.joint_logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        pjt=tiny_inputs["pjt"],
        wjt=tiny_inputs["wjt"],
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
    )
    lp_p = posterior.joint_logpost(
        qjt=_permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=_permute_TJ_tf(tiny_inputs["pjt"], perm),
        wjt=_permute_TJ_tf(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=_permute_TJ_tf(tiny_inputs["gamma"], perm),
    )

    _assert_near(lp_p, lp)
