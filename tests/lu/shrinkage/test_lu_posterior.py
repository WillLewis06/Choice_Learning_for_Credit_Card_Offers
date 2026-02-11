"""
Unit tests for the LuPosteriorTF (Lu shrinkage posterior).

This module is written without pytest fixture injection:
- All tests are plain functions with no fixture arguments.
- Shared assertion / permutation helpers are imported from `lu_conftest`.
- A single small posterior instance and a single tiny input panel are constructed
  at module import time and reused across tests.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorTF
from lu_conftest import (
    assert_all_finite_tf,
    assert_prob_simplex_tf,
    normal_logpdf_tf,
    permute_TJ_tf,
    permute_vec_tf,
)

DTYPE = tf.float64

# Numerical tolerances (float64, but avoid overly brittle exact-equality checks
# unless the test is explicitly verifying an identity/property).
ATOL = 1e-10
PROB_ATOL = 1e-12
RTOL = 0.0


def _tf(x) -> tf.Tensor:
    """Convenience: create a scalar/array constant with the module dtype."""
    return tf.constant(x, dtype=DTYPE)


def _assert_near(a: tf.Tensor, b: tf.Tensor, atol: float = ATOL) -> None:
    """Tensor-aware near equality with clean failure messages."""
    tf.debugging.assert_near(a, b, atol=atol, rtol=RTOL)


def _default_global_params() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Common (beta_p, beta_w, r) used across many tests."""
    return _tf(-1.0), _tf(0.3), _tf(0.0)


def _make_tiny_inputs() -> dict:
    """
    Construct a canonical tiny (T=2, J=3) panel used throughout this test module.

    Keys
    - T, J: ints
    - pjt, wjt, qjt: (T, J) tf.float64
    - q0t, E_bar, phi: (T,) tf.float64
    - njt, gamma: (T, J) tf.float64
    """
    T, J = 2, 3

    pjt = _tf([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]])
    wjt = _tf([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]])

    qjt = _tf([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]])
    q0t = _tf([20.0, 15.0])

    E_bar = _tf([0.1, -0.2])
    njt = _tf([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]])

    gamma = _tf([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    phi = _tf([0.6, 0.4])

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
        "phi": phi,
    }


posterior = LuPosteriorTF(n_draws=25, seed=123, dtype=DTYPE)
tiny_inputs = _make_tiny_inputs()


# -----------------------------------------------------------------------------
# 1) Deterministic helpers
# -----------------------------------------------------------------------------
def test_mean_utility_jt_shape_and_identity():
    """_mean_utility_jt returns a length-J vector and equals the linear identity."""
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

    # Adding a constant to E_bar_t shifts all components equally, so within-vector
    # differences are invariant.
    delta2 = posterior._mean_utility_jt(
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t + _tf(3.0),
        njt_t=njt_t,
    )
    _assert_near(delta - delta[0], delta2 - delta2[0])


# -----------------------------------------------------------------------------
# 2) Choice probabilities
# -----------------------------------------------------------------------------
def test_choice_probs_t_shapes_simplex_bounds():
    """_choice_probs_t returns (J,) inside shares and a scalar outside share on the simplex."""
    J = 5
    pjt_t = _tf([1.0, 1.1, 0.9, 1.3, 0.8])
    delta_t = _tf([0.2, -0.1, 0.0, 0.3, -0.2])
    r = _tf(0.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    assert tuple(sjt_t.shape) == (J,)
    assert s0t.shape == ()
    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)


def test_choice_probs_t_extreme_negative_delta_outside_near_one():
    """Very negative utilities should push almost all share to the outside option."""
    J = 4
    pjt_t = _tf([1.0, 1.2, 0.8, 1.1])
    delta_t = _tf([-50.0] * J)
    r = _tf(0.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)

    assert float(s0t.numpy()) > 1.0 - ATOL
    assert float(tf.reduce_max(sjt_t).numpy()) < ATOL


def test_choice_probs_t_extreme_positive_delta_outside_near_zero():
    """Very positive utilities should drive the outside share near zero."""
    J = 4
    pjt_t = _tf([1.0, 1.2, 0.8, 1.1])
    delta_t = _tf([50.0] * J)
    r = _tf(0.0)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)

    assert float(s0t.numpy()) < ATOL
    assert abs(float(tf.reduce_sum(sjt_t).numpy()) - 1.0) < ATOL


def test_choice_probs_t_monotone_under_inside_shift():
    """Increasing all inside utilities increases total inside share and reduces outside share."""
    J = 5
    pjt_t = _tf([1.0, 1.1, 0.9, 1.3, 0.8])
    delta_t = _tf([0.2, -0.1, 0.0, 0.3, -0.2])
    r = _tf(0.0)

    sjt0, s00 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    sjt1, s01 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t + _tf(1.0), r=r)

    assert float(s01.numpy()) < float(s00.numpy())
    assert float(tf.reduce_sum(sjt1).numpy()) > float(tf.reduce_sum(sjt0).numpy())

    assert_prob_simplex_tf(sjt0, s00, atol=PROB_ATOL)
    assert_prob_simplex_tf(sjt1, s01, atol=PROB_ATOL)


# -----------------------------------------------------------------------------
# 3) Likelihood
# -----------------------------------------------------------------------------
def test_market_loglik_finite_reasonable_inputs():
    """market_loglik returns a finite scalar for a typical tiny-market input."""
    t = 0
    beta_p, beta_w, r = _default_global_params()

    ll = posterior.market_loglik(
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
    assert_all_finite_tf(ll)


def test_market_loglik_outside_only_identity():
    """If qjt is all zeros, market_loglik reduces to q0t * log(s0t) (with clipping)."""
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

    ll = posterior.market_loglik(
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
    _assert_near(ll, expected, atol=ATOL)


def test_market_loglik_inside_only_identity():
    """If q0t is zero, market_loglik reduces to sum_j qjt * log(sjt) (with clipping)."""
    J = 4
    qjt_t = _tf([10.0, 5.0, 1.0, 2.0])
    q0t_t = _tf(0.0)

    pjt_t = _tf([1.0, 1.2, 0.8, 1.1])
    wjt_t = _tf([0.4, 0.7, 0.6, 0.2])

    beta_p = _tf(-1.0)
    beta_w = _tf(0.2)
    r = _tf(0.0)
    E_bar_t = _tf(0.0)
    njt_t = tf.zeros((J,), dtype=DTYPE)

    ll = posterior.market_loglik(
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
    sjt_t, _ = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    sjt_c = tf.clip_by_value(sjt_t, posterior.eps, 1.0)

    expected = tf.reduce_sum(qjt_t * tf.math.log(sjt_c))
    _assert_near(ll, expected, atol=ATOL)


def test_loglik_vec_shape_and_matches_market_loglik_stack():
    """loglik_vec equals stacking market_loglik over markets and has shape (T,)."""
    beta_p, beta_w, r = _default_global_params()

    ll_vec = posterior.loglik_vec(
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
    assert tuple(ll_vec.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(ll_vec)

    stacked = tf.stack(
        [
            posterior.market_loglik(
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
    _assert_near(ll_vec, stacked, atol=ATOL)


# -----------------------------------------------------------------------------
# 4) Priors
# -----------------------------------------------------------------------------
def test_logprior_global_matches_closed_form_at_means():
    """logprior_global matches the sum of independent normal log-densities at means."""
    beta_p = posterior.beta_p_mean
    beta_w = posterior.beta_w_mean
    r = posterior.r_mean

    lp = posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)

    expected = (
        normal_logpdf_tf(beta_p, posterior.beta_p_mean, posterior.beta_p_var)
        + normal_logpdf_tf(beta_w, posterior.beta_w_mean, posterior.beta_w_var)
        + normal_logpdf_tf(r, posterior.r_mean, posterior.r_var)
    )
    _assert_near(lp, expected, atol=PROB_ATOL)


def test_logprior_global_symmetry_around_mean():
    """A normal prior is symmetric: lp(mean+d) == lp(mean-d) for each scalar parameter."""
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


def test_logprior_market_vec_shapes_and_finite():
    """logprior_market_vec returns (T,) and is finite on the tiny canonical panel."""
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )
    assert tuple(lp_t.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(lp_t)


def test_logprior_market_vec_phi_clipping_at_endpoints():
    """phi at {0,1} should not produce NaNs/Infs due to internal clipping."""
    phi = _tf([0.0, 1.0])
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=phi,
    )
    assert_all_finite_tf(lp_t)


def test_logprior_market_vec_bernoulli_term_counts():
    """Difference in logprior across markets matches Bernoulli + variance-selection terms."""
    T, J = 2, 5
    E_bar = tf.zeros((T,), dtype=DTYPE)
    njt = tf.zeros((T, J), dtype=DTYPE)

    gamma = tf.concat(
        [tf.ones((1, J), dtype=DTYPE), tf.zeros((1, J), dtype=DTYPE)], axis=0
    )
    phi = _tf([0.7, 0.7])

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)

    phi_val = 0.7
    bern_diff = float(J) * (np.log(phi_val) - np.log(1.0 - phi_val))

    t1 = float(posterior.T1_sq.numpy())
    t0 = float(posterior.T0_sq.numpy())
    n_diff = -0.5 * float(J) * (np.log(t1) - np.log(t0))

    expected_diff = bern_diff + n_diff
    actual_diff = float((lp[0] - lp[1]).numpy())
    assert abs(actual_diff - expected_diff) < ATOL


def test_logprior_market_vec_gamma_selects_variance():
    """All-ones gamma should yield higher prior density than all-zeros for large njt."""
    T, J = 2, 6
    E_bar = tf.zeros((T,), dtype=DTYPE)
    njt = tf.ones((T, J), dtype=DTYPE) * _tf(2.0)

    gamma = tf.concat(
        [tf.ones((1, J), dtype=DTYPE), tf.zeros((1, J), dtype=DTYPE)], axis=0
    )
    phi = _tf([0.5, 0.5])

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)
    assert float(lp[0].numpy()) > float(lp[1].numpy())


# -----------------------------------------------------------------------------
# 5) Posterior composition
# -----------------------------------------------------------------------------
def test_logpost_vec_equals_loglik_vec_plus_logprior_market_vec():
    """logpost_vec equals loglik_vec + logprior_market_vec elementwise over markets."""
    beta_p, beta_w, r = _default_global_params()

    lp_vec = posterior.logpost_vec(
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
        phi=tiny_inputs["phi"],
    )

    ll_vec = posterior.loglik_vec(
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

    prior_vec = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    _assert_near(lp_vec, ll_vec + prior_vec, atol=ATOL)


def test_logpost_equals_sum_logpost_vec_plus_logprior_global():
    """logpost equals sum_t logpost_vec[t] plus the global prior."""
    beta_p, beta_w, r = _default_global_params()

    lp = posterior.logpost(
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
        phi=tiny_inputs["phi"],
    )

    lp_vec = posterior.logpost_vec(
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
        phi=tiny_inputs["phi"],
    )

    expected = tf.reduce_sum(lp_vec) + posterior.logprior_global(
        beta_p=beta_p, beta_w=beta_w, r=r
    )
    _assert_near(lp, expected, atol=ATOL)


# -----------------------------------------------------------------------------
# 6) Numerical edge cases
# -----------------------------------------------------------------------------
def test_market_loglik_finite_when_shares_extremely_small_due_to_clipping():
    """market_loglik remains finite when some inside shares are tiny (clipped)."""
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

    ll = posterior.market_loglik(
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
    assert_all_finite_tf(ll)


def test_choice_probs_t_finite_for_large_r_moderate_prices():
    """_choice_probs_t stays finite when r is large but prices are moderate."""
    J = 5
    pjt_t = _tf([0.5, 0.8, 0.6, 0.7, 0.9])
    delta_t = _tf([0.0, 0.1, -0.1, 0.05, -0.05])

    r = _tf(5.0)
    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    assert_prob_simplex_tf(sjt_t, s0t, atol=PROB_ATOL)


# -----------------------------------------------------------------------------
# 7) Permutation invariance / equivariance (products within market)
# -----------------------------------------------------------------------------
def test_mean_utility_equivariant_under_product_permutation():
    """_mean_utility_jt is equivariant to permuting product order within a market."""
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

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

    delta_perm_inputs = posterior._mean_utility_jt(
        pjt_t=permute_vec_tf(p, perm),
        wjt_t=permute_vec_tf(w, perm),
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=permute_vec_tf(n, perm),
    )

    _assert_near(delta_perm_inputs, permute_vec_tf(delta, perm))


def test_choice_probs_equivariant_under_product_permutation():
    """_choice_probs_t is equivariant in sjt and invariant in s0t under product permutation."""
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

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
        pjt_t=permute_vec_tf(p, perm),
        delta_t=permute_vec_tf(delta, perm),
        r=r,
    )

    _assert_near(s0_p, s0, atol=PROB_ATOL)
    _assert_near(sj_p, permute_vec_tf(sj, perm), atol=PROB_ATOL)


def test_market_loglik_invariant_under_product_permutation():
    """market_loglik is invariant to product reordering within a market."""
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    beta_p, beta_w, r = _default_global_params()

    ll = posterior.market_loglik(
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

    ll_p = posterior.market_loglik(
        qjt_t=permute_vec_tf(tiny_inputs["qjt"][t], perm),
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=permute_vec_tf(tiny_inputs["pjt"][t], perm),
        wjt_t=permute_vec_tf(tiny_inputs["wjt"][t], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=permute_vec_tf(tiny_inputs["njt"][t], perm),
    )

    _assert_near(ll_p, ll, atol=ATOL)


def test_loglik_vec_invariant_under_product_permutation():
    """loglik_vec is invariant to permuting products consistently across all markets."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    ll = posterior.loglik_vec(
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

    ll_p = posterior.loglik_vec(
        qjt=permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=permute_TJ_tf(tiny_inputs["pjt"], perm),
        wjt=permute_TJ_tf(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
    )

    _assert_near(ll_p, ll, atol=ATOL)


def test_logprior_market_vec_invariant_under_product_permutation():
    """logprior_market_vec is invariant to permuting product-specific terms (njt, gamma)."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    lp_p = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=permute_TJ_tf(tiny_inputs["gamma"], perm),
        phi=tiny_inputs["phi"],
    )

    _assert_near(lp_p, lp, atol=ATOL)


def test_logpost_vec_invariant_under_product_permutation():
    """logpost_vec is invariant to permuting products consistently across all markets."""
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    beta_p, beta_w, r = _default_global_params()

    lp = posterior.logpost_vec(
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
        phi=tiny_inputs["phi"],
    )

    lp_p = posterior.logpost_vec(
        qjt=permute_TJ_tf(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=permute_TJ_tf(tiny_inputs["pjt"], perm),
        wjt=permute_TJ_tf(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=permute_TJ_tf(tiny_inputs["njt"], perm),
        gamma=permute_TJ_tf(tiny_inputs["gamma"], perm),
        phi=tiny_inputs["phi"],
    )

    _assert_near(lp_p, lp, atol=ATOL)
