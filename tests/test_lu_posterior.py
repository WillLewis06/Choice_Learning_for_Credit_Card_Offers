import numpy as np
import pytest
import tensorflow as tf

from market_shock_estimators.lu_posterior import LuPosteriorTF


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _assert_all_finite(*tensors) -> None:
    for x in tensors:
        x = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x))
        assert bool(ok.numpy()), "Found non-finite values."


def _assert_prob_simplex(
    sjt_t: tf.Tensor, s0t: tf.Tensor, *, atol: float = 1e-12
) -> None:
    sjt_t = tf.convert_to_tensor(sjt_t)
    s0t = tf.convert_to_tensor(s0t)

    assert sjt_t.shape.rank == 1
    assert s0t.shape.rank == 0

    _assert_all_finite(sjt_t, s0t)

    sjt_np = sjt_t.numpy()
    s0_np = float(s0t.numpy())

    assert np.min(sjt_np) >= -atol
    assert np.max(sjt_np) <= 1.0 + atol
    assert s0_np >= -atol
    assert s0_np <= 1.0 + atol

    err = abs(s0_np + float(np.sum(sjt_np)) - 1.0)
    assert err <= atol, f"Simplex identity violated (err={err:.3e}, atol={atol:.3e})."


def _normal_logpdf(x, mean, var, two_pi) -> tf.Tensor:
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    mean = tf.convert_to_tensor(mean, dtype=tf.float64)
    var = tf.convert_to_tensor(var, dtype=tf.float64)
    two_pi = tf.convert_to_tensor(two_pi, dtype=tf.float64)
    return -0.5 * tf.math.log(two_pi * var) - 0.5 * tf.square(x - mean) / var


def _permute_vec(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.gather(x, perm, axis=0)


def _permute_TJ(x: tf.Tensor, perm: tf.Tensor) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.gather(x, perm, axis=1)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def posterior():
    # Small n_draws keeps tests fast; seed irrelevant for invariants
    return LuPosteriorTF(n_draws=25, seed=123, dtype=tf.float64)


@pytest.fixture
def tiny_inputs():
    """
    Small consistent batched inputs for T=2, J=3.
    """
    T, J = 2, 3
    pjt = tf.constant([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=tf.float64)
    wjt = tf.constant([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=tf.float64)

    # counts
    qjt = tf.constant([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=tf.float64)
    q0t = tf.constant([20.0, 15.0], dtype=tf.float64)

    # latent states
    E_bar = tf.constant([0.1, -0.2], dtype=tf.float64)
    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=tf.float64)

    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=tf.float64)
    phi = tf.constant([0.6, 0.4], dtype=tf.float64)

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


# -----------------------------------------------------------------------------
# 1) Deterministic helpers
# -----------------------------------------------------------------------------
def test_mean_utility_jt_shape_and_identity(posterior):
    J = 4
    pjt_t = tf.constant([1.0, 1.2, 0.8, 0.9], dtype=tf.float64)
    wjt_t = tf.constant([0.4, 0.7, 0.6, 0.2], dtype=tf.float64)
    njt_t = tf.constant([0.1, -0.2, 0.0, 0.3], dtype=tf.float64)

    beta_p = tf.constant(-1.5, dtype=tf.float64)
    beta_w = tf.constant(0.2, dtype=tf.float64)
    E_bar_t = tf.constant(0.7, dtype=tf.float64)

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
    assert np.allclose(delta.numpy(), expected.numpy(), atol=0.0, rtol=0.0)

    # E_bar_t adds equally to all components: differences don't depend on it
    delta2 = posterior._mean_utility_jt(
        pjt_t=pjt_t,
        wjt_t=wjt_t,
        beta_p=beta_p,
        beta_w=beta_w,
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
    pjt_t = tf.constant([1.0, 1.1, 0.9, 1.3, 0.8], dtype=tf.float64)
    delta_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    assert tuple(sjt_t.shape) == (J,)
    assert s0t.shape == ()
    _assert_prob_simplex(sjt_t, s0t, atol=1e-12)


def test_choice_probs_t_extreme_negative_delta_outside_near_one(posterior):
    J = 4
    pjt_t = tf.constant([1.0, 1.2, 0.8, 1.1], dtype=tf.float64)
    delta_t = tf.constant([-50.0] * J, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    _assert_prob_simplex(sjt_t, s0t, atol=1e-12)

    assert float(s0t.numpy()) > 1.0 - 1e-10
    assert np.max(sjt_t.numpy()) < 1e-10


def test_choice_probs_t_extreme_positive_delta_outside_near_zero(posterior):
    J = 4
    pjt_t = tf.constant([1.0, 1.2, 0.8, 1.1], dtype=tf.float64)
    delta_t = tf.constant([50.0] * J, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    _assert_prob_simplex(sjt_t, s0t, atol=1e-12)

    assert float(s0t.numpy()) < 1e-10
    assert abs(float(tf.reduce_sum(sjt_t).numpy()) - 1.0) < 1e-10


def test_choice_probs_t_monotone_under_inside_shift(posterior):
    J = 5
    pjt_t = tf.constant([1.0, 1.1, 0.9, 1.3, 0.8], dtype=tf.float64)
    delta_t = tf.constant([0.2, -0.1, 0.0, 0.3, -0.2], dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

    sjt0, s00 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    sjt1, s01 = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t + 1.0, r=r)

    assert float(s01.numpy()) < float(s00.numpy())
    assert float(tf.reduce_sum(sjt1).numpy()) > float(tf.reduce_sum(sjt0).numpy())

    _assert_prob_simplex(sjt0, s00, atol=1e-12)
    _assert_prob_simplex(sjt1, s01, atol=1e-12)


# -----------------------------------------------------------------------------
# 3) Likelihood
# -----------------------------------------------------------------------------
def test_market_loglik_finite_reasonable_inputs(posterior, tiny_inputs):
    t = 0
    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
    _assert_all_finite(ll)


def test_market_loglik_outside_only_identity(posterior):
    J = 4
    qjt_t = tf.zeros((J,), dtype=tf.float64)
    q0t_t = tf.constant(100.0, dtype=tf.float64)

    pjt_t = tf.constant([1.0, 1.2, 0.8, 1.1], dtype=tf.float64)
    wjt_t = tf.constant([0.4, 0.7, 0.6, 0.2], dtype=tf.float64)

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.2, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)
    njt_t = tf.zeros((J,), dtype=tf.float64)

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
    assert np.allclose(ll.numpy(), expected.numpy(), atol=1e-10)


def test_market_loglik_inside_only_identity(posterior):
    J = 4
    qjt_t = tf.constant([10.0, 5.0, 1.0, 2.0], dtype=tf.float64)
    q0t_t = tf.constant(0.0, dtype=tf.float64)

    pjt_t = tf.constant([1.0, 1.2, 0.8, 1.1], dtype=tf.float64)
    wjt_t = tf.constant([0.4, 0.7, 0.6, 0.2], dtype=tf.float64)

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.2, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)
    njt_t = tf.zeros((J,), dtype=tf.float64)

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
    assert np.allclose(ll.numpy(), expected.numpy(), atol=1e-10)


def test_loglik_vec_shape_and_matches_market_loglik_stack(posterior, tiny_inputs):
    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
    _assert_all_finite(ll_vec)

    # Stack of market_loglik
    stacked = []
    for t in range(tiny_inputs["T"]):
        stacked.append(
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
        )
    stacked = tf.stack(stacked)
    assert np.allclose(ll_vec.numpy(), stacked.numpy(), atol=1e-10)


# -----------------------------------------------------------------------------
# 4) Priors
# -----------------------------------------------------------------------------
def test_logprior_global_matches_closed_form_at_means(posterior):
    beta_p = posterior.beta_p_mean
    beta_w = posterior.beta_w_mean
    r = posterior.r_mean

    lp = posterior.logprior_global(beta_p=beta_p, beta_w=beta_w, r=r)

    expected = (
        _normal_logpdf(
            beta_p, posterior.beta_p_mean, posterior.beta_p_var, posterior.two_pi
        )
        + _normal_logpdf(
            beta_w, posterior.beta_w_mean, posterior.beta_w_var, posterior.two_pi
        )
        + _normal_logpdf(r, posterior.r_mean, posterior.r_var, posterior.two_pi)
    )

    assert np.allclose(lp.numpy(), expected.numpy(), atol=1e-12)


def test_logprior_global_symmetry_around_mean(posterior):
    d = tf.constant(0.37, dtype=tf.float64)

    beta_p_m = posterior.beta_p_mean
    beta_w_m = posterior.beta_w_mean
    r_m = posterior.r_mean

    # Symmetry in beta_p
    lp1 = posterior.logprior_global(beta_p=beta_p_m + d, beta_w=beta_w_m, r=r_m)
    lp2 = posterior.logprior_global(beta_p=beta_p_m - d, beta_w=beta_w_m, r=r_m)
    assert np.allclose(lp1.numpy(), lp2.numpy(), atol=1e-12)

    # Symmetry in beta_w
    lp1 = posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m + d, r=r_m)
    lp2 = posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m - d, r=r_m)
    assert np.allclose(lp1.numpy(), lp2.numpy(), atol=1e-12)

    # Symmetry in r
    lp1 = posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m, r=r_m + d)
    lp2 = posterior.logprior_global(beta_p=beta_p_m, beta_w=beta_w_m, r=r_m - d)
    assert np.allclose(lp1.numpy(), lp2.numpy(), atol=1e-12)


def test_logprior_market_vec_shapes_and_finite(posterior, tiny_inputs):
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=tiny_inputs["phi"],
    )
    assert tuple(lp_t.shape) == (tiny_inputs["T"],)
    _assert_all_finite(lp_t)


def test_logprior_market_vec_phi_clipping_at_endpoints(posterior, tiny_inputs):
    # Use endpoints exactly; function clips internally.
    phi = tf.constant([0.0, 1.0], dtype=tf.float64)
    lp_t = posterior.logprior_market_vec(
        E_bar=tiny_inputs["E_bar"],
        njt=tiny_inputs["njt"],
        gamma=tiny_inputs["gamma"],
        phi=phi,
    )
    _assert_all_finite(lp_t)


def test_logprior_market_vec_bernoulli_term_counts(posterior):
    """
    Isolate Bernoulli contribution by holding E_bar, njt, phi fixed and varying gamma.
    Compare market-wise difference against J*(log(phi)-log(1-phi)).
    """
    T, J = 2, 5
    E_bar = tf.zeros((T,), dtype=tf.float64)
    njt = tf.zeros((T, J), dtype=tf.float64)

    # Market 0: all ones, Market 1: all zeros
    gamma = tf.concat(
        [tf.ones((1, J), tf.float64), tf.zeros((1, J), tf.float64)], axis=0
    )
    phi = tf.constant([0.7, 0.7], dtype=tf.float64)

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)

    # Since everything else matches, difference should be Bernoulli-only.

    phi_val = 0.7
    bern_diff = float(J) * (np.log(phi_val) - np.log(1.0 - phi_val))

    # n-prior contributes even when njt == 0 because of the -0.5*log(var) term
    t1 = float(posterior.T1_sq.numpy())
    t0 = float(posterior.T0_sq.numpy())
    n_diff = -0.5 * float(J) * (np.log(t1) - np.log(t0))

    expected_diff = bern_diff + n_diff

    actual_diff = float((lp[0] - lp[1]).numpy())
    assert abs(actual_diff - expected_diff) < 1e-10


def test_logprior_market_vec_gamma_selects_variance(posterior):
    """
    With large |n|, Normal logpdf is higher (less negative) under larger variance.
    """
    T, J = 2, 6
    E_bar = tf.zeros((T,), dtype=tf.float64)

    # Large magnitude n
    njt = tf.ones((T, J), dtype=tf.float64) * 2.0

    # Market 0: gamma=1 selects T1_sq; Market 1: gamma=0 selects T0_sq
    gamma = tf.concat(
        [tf.ones((1, J), tf.float64), tf.zeros((1, J), tf.float64)], axis=0
    )

    # Fix phi so Bernoulli differs; remove Bernoulli analytically via difference adjustment.
    phi_val = 0.5
    phi = tf.constant([phi_val, phi_val], dtype=tf.float64)

    lp = posterior.logprior_market_vec(E_bar=E_bar, njt=njt, gamma=gamma, phi=phi)

    # Market 0 should have higher lp due to larger variance for njt
    assert float(lp[0].numpy()) > float(lp[1].numpy())


# -----------------------------------------------------------------------------
# 5) Posterior composition
# -----------------------------------------------------------------------------
def test_logpost_vec_equals_loglik_vec_plus_logprior_market_vec(posterior, tiny_inputs):
    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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

    assert np.allclose(lp_vec.numpy(), (ll_vec + prior_vec).numpy(), atol=1e-10)


def test_logpost_equals_sum_logpost_vec_plus_logprior_global(posterior, tiny_inputs):
    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
    assert np.allclose(lp.numpy(), expected.numpy(), atol=1e-10)


# -----------------------------------------------------------------------------
# 6) Numerical edge cases
# -----------------------------------------------------------------------------
def test_market_loglik_finite_when_shares_extremely_small_due_to_clipping(posterior):
    """
    Force one product to have extremely low utility to push its share near 0;
    ensure loglik stays finite via clipping.
    """
    J = 3
    qjt_t = tf.constant([10.0, 10.0, 10.0], dtype=tf.float64)
    q0t_t = tf.constant(10.0, dtype=tf.float64)

    pjt_t = tf.constant([1.0, 1.0, 1.0], dtype=tf.float64)
    wjt_t = tf.zeros((J,), dtype=tf.float64)

    beta_p = tf.constant(0.0, dtype=tf.float64)
    beta_w = tf.constant(0.0, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)
    E_bar_t = tf.constant(0.0, dtype=tf.float64)

    # Make product 0 extremely unattractive
    njt_t = tf.constant([-100.0, 0.0, 0.0], dtype=tf.float64)

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
    _assert_all_finite(ll)


def test_choice_probs_t_finite_for_large_r_moderate_prices(posterior):
    J = 5
    pjt_t = tf.constant([0.5, 0.8, 0.6, 0.7, 0.9], dtype=tf.float64)
    delta_t = tf.constant([0.0, 0.1, -0.1, 0.05, -0.05], dtype=tf.float64)

    # Large r => sigma = exp(r) large, but keep prices moderate to avoid overflow.
    r = tf.constant(5.0, dtype=tf.float64)

    sjt_t, s0t = posterior._choice_probs_t(pjt_t=pjt_t, delta_t=delta_t, r=r)
    _assert_prob_simplex(sjt_t, s0t, atol=1e-12)


# -----------------------------------------------------------------------------
# 7) Permutation invariance / equivariance (products within market)
# -----------------------------------------------------------------------------
def test_mean_utility_equivariant_under_product_permutation(posterior, tiny_inputs):
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)

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
        pjt_t=_permute_vec(p, perm),
        wjt_t=_permute_vec(w, perm),
        beta_p=beta_p,
        beta_w=beta_w,
        E_bar_t=E_bar_t,
        njt_t=_permute_vec(n, perm),
    )

    assert np.allclose(
        delta_perm_inputs.numpy(), _permute_vec(delta, perm).numpy(), atol=0.0, rtol=0.0
    )


def test_choice_probs_equivariant_under_product_permutation(posterior, tiny_inputs):
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
        pjt_t=_permute_vec(p, perm),
        delta_t=_permute_vec(delta, perm),
        r=r,
    )

    assert np.allclose(float(s0_p.numpy()), float(s0.numpy()), atol=1e-12, rtol=0.0)
    assert np.allclose(
        sj_p.numpy(), _permute_vec(sj, perm).numpy(), atol=1e-12, rtol=0.0
    )


def test_market_loglik_invariant_under_product_permutation(posterior, tiny_inputs):
    t = 0
    J = tiny_inputs["J"]
    perm = tf.constant([2, 0, 1], dtype=tf.int32)
    assert int(perm.shape[0]) == J

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
        qjt_t=_permute_vec(tiny_inputs["qjt"][t], perm),
        q0t_t=tiny_inputs["q0t"][t],
        pjt_t=_permute_vec(tiny_inputs["pjt"][t], perm),
        wjt_t=_permute_vec(tiny_inputs["wjt"][t], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar_t=tiny_inputs["E_bar"][t],
        njt_t=_permute_vec(tiny_inputs["njt"][t], perm),
    )

    assert np.allclose(ll_p.numpy(), ll.numpy(), atol=1e-10, rtol=0.0)


def test_loglik_vec_invariant_under_product_permutation(posterior, tiny_inputs):
    perm = tf.constant([2, 0, 1], dtype=tf.int32)

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
        qjt=_permute_TJ(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=_permute_TJ(tiny_inputs["pjt"], perm),
        wjt=_permute_TJ(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
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

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
        qjt=_permute_TJ(tiny_inputs["qjt"], perm),
        q0t=tiny_inputs["q0t"],
        pjt=_permute_TJ(tiny_inputs["pjt"], perm),
        wjt=_permute_TJ(tiny_inputs["wjt"], perm),
        beta_p=beta_p,
        beta_w=beta_w,
        r=r,
        E_bar=tiny_inputs["E_bar"],
        njt=_permute_TJ(tiny_inputs["njt"], perm),
        gamma=_permute_TJ(tiny_inputs["gamma"], perm),
        phi=tiny_inputs["phi"],
    )

    assert np.allclose(lp_p.numpy(), lp.numpy(), atol=1e-10, rtol=0.0)
