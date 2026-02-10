import numpy as np
import tensorflow as tf

from conftest import assert_all_finite_tf, assert_prob_simplex_tf


# -----------------------------------------------------------------------------
# Local helpers (not in conftest; specific to permutation tests / closed forms)
# -----------------------------------------------------------------------------
def _normal_logpdf(x, mean, var, two_pi) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    mean = tf.convert_to_tensor(mean, dtype=tf.float64)
    var = tf.convert_to_tensor(var, dtype=tf.float64)
    two_pi = tf.convert_to_tensor(two_pi, dtype=tf.float64)
    return -0.5 * tf.math.log(two_pi * var) - 0.5 * tf.square(x - mean) / var


def _permute_vec(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    return tf.gather(tf.convert_to_tensor(x), perm, axis=0)


def _permute_TJ(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    return tf.gather(tf.convert_to_tensor(x), perm, axis=1)


# -----------------------------------------------------------------------------
# 1) Deterministic helpers
# -----------------------------------------------------------------------------
def test_mean_utility_jt_shape_and_identity(posterior):
    J = 4
    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=tf.float64)
    njt_t = tf.constant([0.1, -0.2, 0.0, 0.3], dtype=tf.float64)

    alpha = tf.constant(1.7, dtype=tf.float64)
    E_bar_t = tf.constant(0.7, dtype=tf.float64)

    delta = posterior._mean_utility_jt(
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    assert tuple(delta.shape) == (J,)

    expected = alpha * delta_cl_t + E_bar_t + njt_t
    assert np.allclose(delta.numpy(), expected.numpy(), atol=0.0, rtol=0.0)

    # E_bar_t adds equally to all components: differences don't depend on it
    delta2 = posterior._mean_utility_jt(
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t + 3.0,
        njt_t=njt_t,
    )
    d = (delta - delta[0]).numpy()
    d2 = (delta2 - delta2[0]).numpy()
    assert np.allclose(d, d2, atol=1e-12, rtol=0.0)


# -----------------------------------------------------------------------------
# 2) Choice probabilities
# -----------------------------------------------------------------------------
def test_choice_probs_t_shapes_simplex_bounds(posterior):
    J = 5
    delta_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(delta_t=delta_t)
    assert tuple(sjt_t.shape) == (J,)
    assert s0t.shape == ()
    assert_prob_simplex_tf(sjt_t, s0t, atol=1e-12)


def test_choice_probs_t_extreme_negative_delta_outside_near_one(posterior):
    J = 4
    delta_t = tf.constant([-50.0] * J, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(delta_t=delta_t)
    assert_prob_simplex_tf(sjt_t, s0t, atol=1e-12)

    assert float(s0t.numpy()) > 1.0 - 1e-10
    assert np.max(sjt_t.numpy()) < 1e-10


def test_choice_probs_t_extreme_positive_delta_outside_near_zero(posterior):
    J = 4
    delta_t = tf.constant([50.0] * J, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(delta_t=delta_t)
    assert_prob_simplex_tf(sjt_t, s0t, atol=1e-12)

    assert float(s0t.numpy()) < 1e-10
    assert abs(float(tf.reduce_sum(sjt_t).numpy()) - 1.0) < 1e-10


def test_choice_probs_t_monotone_under_inside_shift(posterior):
    delta_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=tf.float64)

    sjt0, s00 = posterior._choice_probs_t(delta_t=delta_t)
    sjt1, s01 = posterior._choice_probs_t(delta_t=delta_t + 1.0)

    assert float(s01.numpy()) < float(s00.numpy())
    assert float(tf.reduce_sum(sjt1).numpy()) > float(tf.reduce_sum(sjt0).numpy())

    assert_prob_simplex_tf(sjt0, s00, atol=1e-12)
    assert_prob_simplex_tf(sjt1, s01, atol=1e-12)


# -----------------------------------------------------------------------------
# 3) Likelihood
# -----------------------------------------------------------------------------
def test_market_loglik_finite_reasonable_inputs(posterior, tiny_inputs):
    t = 0
    ll = posterior.market_loglik(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        delta_cl_t=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )
    assert ll.shape == ()
    assert_all_finite_tf(ll)


def test_market_loglik_outside_only_identity(posterior):
    J = 4
    qjt_t = tf.zeros((J,), dtype=tf.float64)
    q0t_t = tf.constant(100.0, dtype=tf.float64)

    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=tf.float64)
    alpha = tf.constant(1.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)
    njt_t = tf.zeros((J,), dtype=tf.float64)

    ll = posterior.market_loglik(
        qjt_t=qjt_t,
        q0t_t=q0t_t,
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )

    delta_t = posterior._mean_utility_jt(
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    _, s0t = posterior._choice_probs_t(delta_t=delta_t)
    s0t_c = tf.clip_by_value(s0t, posterior.eps, 1.0)

    expected = q0t_t * tf.math.log(s0t_c)
    assert np.allclose(ll.numpy(), expected.numpy(), atol=1e-10)


def test_market_loglik_inside_only_identity(posterior):
    J = 4
    qjt_t = tf.constant([10.0, 5.0, 1.0, 2.0], dtype=tf.float64)
    q0t_t = tf.constant(0.0, dtype=tf.float64)

    delta_cl_t = tf.constant([0.3, -0.2, 0.0, 0.1], dtype=tf.float64)
    alpha = tf.constant(1.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)
    njt_t = tf.zeros((J,), dtype=tf.float64)

    ll = posterior.market_loglik(
        qjt_t=qjt_t,
        q0t_t=q0t_t,
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )

    delta_t = posterior._mean_utility_jt(
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    sjt_t, _ = posterior._choice_probs_t(delta_t=delta_t)
    sjt_c = tf.clip_by_value(sjt_t, posterior.eps, 1.0)

    expected = tf.reduce_sum(qjt_t * tf.math.log(sjt_c))
    assert np.allclose(ll.numpy(), expected.numpy(), atol=1e-10)


def test_loglik_vec_shape_and_matches_market_loglik_stack(posterior, tiny_inputs):
    ll_vec = posterior.loglik_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )
    assert tuple(ll_vec.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(ll_vec)

    stacked = []
    for t in range(tiny_inputs["T"]):
        stacked.append(
            posterior.market_loglik(
                qjt_t=tiny_inputs["qjt"][t],
                q0t_t=tiny_inputs["q0t"][t],
                delta_cl_t=tiny_inputs["delta_cl"][t],
                alpha=tiny_inputs["alpha"],
                E_bar_t=tiny_inputs["E_bar"][t],
                njt_t=tiny_inputs["njt"][t],
            )
        )
    stacked = tf.stack(stacked)
    assert np.allclose(ll_vec.numpy(), stacked.numpy(), atol=1e-10)


# -----------------------------------------------------------------------------
# 4) Priors
# -----------------------------------------------------------------------------
def test_logprior_global_matches_closed_form_at_means(posterior):
    alpha = posterior.alpha_mean
    lp = posterior.logprior_global(alpha=alpha)

    expected = _normal_logpdf(
        alpha, posterior.alpha_mean, posterior.alpha_var, posterior.two_pi
    )
    assert np.allclose(lp.numpy(), expected.numpy(), atol=1e-12)


def test_logprior_global_symmetry_around_mean(posterior):
    d = tf.constant(0.37, dtype=tf.float64)
    a_m = posterior.alpha_mean

    lp1 = posterior.logprior_global(alpha=a_m + d)
    lp2 = posterior.logprior_global(alpha=a_m - d)
    assert np.allclose(lp1.numpy(), lp2.numpy(), atol=1e-12)


def test_logprior_market_vec_shapes_and_finite(posterior, tiny_inputs):
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )
    assert tuple(lp_t.shape) == (tiny_inputs["T"],)
    assert_all_finite_tf(lp_t)


def test_logprior_market_vec_phi_clipping_at_endpoints(posterior, tiny_inputs):
    phi = tf.constant([0.0, 1.0], dtype=tf.float64)
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=phi,
    )
    assert_all_finite_tf(lp_t)


def test_logprior_market_vec_bernoulli_term_counts(posterior):
    T, J = 2, 5
    E_bar = tf.zeros((T,), dtype=tf.float64)
    njt = tf.zeros((T, J), dtype=tf.float64)

    gamma = tf.concat(
        [tf.ones((1, J), tf.float64), tf.zeros((1, J), tf.float64)], axis=0
    )
    phi = tf.constant([0.7, 0.7], dtype=tf.float64)

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)

    phi_val = 0.7
    bern_diff = float(J) * (np.log(phi_val) - np.log(1.0 - phi_val))

    t1 = float(posterior.T1_sq.numpy())
    t0 = float(posterior.T0_sq.numpy())
    n_diff = -0.5 * float(J) * (np.log(t1) - np.log(t0))

    expected_diff = bern_diff + n_diff
    actual_diff = float((lp[0] - lp[1]).numpy())
    assert abs(actual_diff - expected_diff) < 1e-10


def test_logprior_market_vec_gamma_selects_variance(posterior):
    T, J = 2, 6
    E_bar = tf.zeros((T,), dtype=tf.float64)
    njt = tf.ones((T, J), dtype=tf.float64) * 2.0

    gamma = tf.concat(
        [tf.ones((1, J), tf.float64), tf.zeros((1, J), tf.float64)], axis=0
    )
    phi = tf.constant([0.5, 0.5], dtype=tf.float64)

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)
    assert float(lp[0].numpy()) > float(lp[1].numpy())


# -----------------------------------------------------------------------------
# 5) Posterior composition
# -----------------------------------------------------------------------------
def test_logpost_vec_equals_loglik_vec_plus_logprior_market_vec(posterior, tiny_inputs):
    lp_vec = posterior.logpost_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    ll_vec = posterior.loglik_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    prior_vec = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    assert np.allclose(lp_vec.numpy(), (ll_vec + prior_vec).numpy(), atol=1e-10)


def test_logpost_equals_sum_logpost_vec_plus_logprior_global(posterior, tiny_inputs):
    lp = posterior.logpost(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    lp_vec = posterior.logpost_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    expected = tf.reduce_sum(lp_vec) + posterior.logprior_global(
        alpha=tiny_inputs["alpha"]
    )
    assert np.allclose(lp.numpy(), expected.numpy(), atol=1e-10)


# -----------------------------------------------------------------------------
# 6) Numerical edge cases
# -----------------------------------------------------------------------------
def test_market_loglik_finite_when_shares_extremely_small_due_to_clipping(posterior):
    J = 3
    qjt_t = tf.constant([10.0, 10.0, 10.0], dtype=tf.float64)
    q0t_t = tf.constant(10.0, dtype=tf.float64)

    delta_cl_t = tf.zeros((J,), dtype=tf.float64)
    alpha = tf.constant(0.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)

    njt_t = tf.constant([-100.0, 0.0, 0.0], dtype=tf.float64)

    ll = posterior.market_loglik(
        qjt_t=qjt_t,
        q0t_t=q0t_t,
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )
    assert_all_finite_tf(ll)


# -----------------------------------------------------------------------------
# 7) Permutation invariance / equivariance (products within market)
# -----------------------------------------------------------------------------
def test_mean_utility_equivariant_under_product_permutation(posterior, tiny_inputs):
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    delta_cl_t = tiny_inputs["delta_cl"][t]
    njt_t = tiny_inputs["njt"][t]
    E_bar_t = tiny_inputs["E_bar"][t]
    alpha = tiny_inputs["alpha"]

    delta = posterior._mean_utility_jt(
        delta_cl_t=delta_cl_t,
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=njt_t,
    )

    delta_perm_inputs = posterior._mean_utility_jt(
        delta_cl_t=_permute_vec(delta_cl_t, perm),
        alpha=alpha,
        E_bar_t=E_bar_t,
        njt_t=_permute_vec(njt_t, perm),
    )

    assert np.allclose(
        delta_perm_inputs.numpy(),
        _permute_vec(delta, perm).numpy(),
        atol=0.0,
        rtol=0.0,
    )


def test_choice_probs_equivariant_under_product_permutation(posterior):
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    delta_t = tf.constant([0.2, -0.1, 0.3], dtype=tf.float64)

    sj, s0 = posterior._choice_probs_t(delta_t=delta_t)
    sj_p, s0_p = posterior._choice_probs_t(delta_t=_permute_vec(delta_t, perm))

    assert np.allclose(float(s0_p.numpy()), float(s0.numpy()), atol=1e-12, rtol=0.0)
    assert np.allclose(
        sj_p.numpy(), _permute_vec(sj, perm).numpy(), atol=1e-12, rtol=0.0
    )


def test_market_loglik_invariant_under_product_permutation(posterior, tiny_inputs):
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    ll = posterior.market_loglik(
        qjt_t=tiny_inputs["qjt"][t],
        q0t_t=tiny_inputs["q0t"][t],
        delta_cl_t=tiny_inputs["delta_cl"][t],
        alpha=tiny_inputs["alpha"],
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=tiny_inputs["njt"][t],
    )

    ll_p = posterior.market_loglik(
        qjt_t=_permute_vec(tiny_inputs["qjt"][t], perm),
        q0t_t=tiny_inputs["q0t"][t],
        delta_cl_t=_permute_vec(tiny_inputs["delta_cl"][t], perm),
        alpha=tiny_inputs["alpha"],
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=_permute_vec(tiny_inputs["njt"][t], perm),
    )

    assert np.allclose(ll_p.numpy(), ll.numpy(), atol=1e-10, rtol=0.0)


def test_loglik_vec_invariant_under_product_permutation(posterior, tiny_inputs):
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    ll = posterior.loglik_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
    )

    ll_p = posterior.loglik_vec(
        qjt=_permute_TJ(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        delta_cl=_permute_TJ(tiny_inputs["delta_cl"], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ(tiny_inputs["njt"], perm),
    )

    assert np.allclose(ll_p.numpy(), ll.numpy(), atol=1e-10, rtol=0.0)


def test_logprior_market_vec_invariant_under_product_permutation(
    posterior, tiny_inputs
):
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    lp_p = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ(tiny_inputs["njt"], perm),
        gamma=_permute_TJ(tiny_inputs["gamma"], perm),
        phi=tiny_inputs["phi"],
    )

    assert np.allclose(lp_p.numpy(), lp.numpy(), atol=1e-10, rtol=0.0)


def test_logpost_vec_invariant_under_product_permutation(posterior, tiny_inputs):
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    lp = posterior.logpost_vec(
        qjt=tiny_inputs["qjt"],
        q0t=tiny_inputs["q0t"],
        delta_cl=tiny_inputs["delta_cl"],
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )

    lp_p = posterior.logpost_vec(
        qjt=_permute_TJ(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        delta_cl=_permute_TJ(tiny_inputs["delta_cl"], perm),
        alpha=tiny_inputs["alpha"],
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ(tiny_inputs["njt"], perm),
        gamma=_permute_TJ(tiny_inputs["gamma"], perm),
        phi=tiny_inputs["phi"],
    )

    assert np.allclose(lp_p.numpy(), lp.numpy(), atol=1e-10, rtol=0.0)
