import numpy as np
import pytest
import tensorflow as tf

from conftest import (
    assert_all_finite_tf,
    assert_bool_like_tf,
    assert_binary_01_tf,
    assert_in_open_unit_interval_tf,
)

from toolbox.mcmc_kernels import (
    rw_mh_step,
    tmh_step,
    gibbs_gamma,
    gibbs_phi,
)


# -----------------------------------------------------------------------------
# Fixtures (kernel-only)
# -----------------------------------------------------------------------------
@pytest.fixture
def slab_spike_consts():
    # Any positive values with T1_sq > T0_sq.
    T0_sq = tf.constant(0.2, dtype=tf.float64)
    T1_sq = tf.constant(2.0, dtype=tf.float64)
    log_T0_sq = tf.math.log(T0_sq)
    log_T1_sq = tf.math.log(T1_sq)
    return T0_sq, T1_sq, log_T0_sq, log_T1_sq


# -----------------------------------------------------------------------------
# Deterministic helper: prob(gamma_j=1 | njt_j, phi) (no RNG)
# -----------------------------------------------------------------------------
def _gibbs_gamma_prob1(
    njt_t: tf.Tensor,
    phi_t: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    log_T0_sq: tf.Tensor,
    log_T1_sq: tf.Tensor,
) -> tf.Tensor:
    """
    Deterministic conditional inclusion probability used by gibbs_gamma.
    Used only for equivariance tests (no sampling).
    """
    eps = tf.constant(1e-30, dtype=tf.float64)

    logp0 = -0.5 * (njt_t * njt_t) / T0_sq - 0.5 * log_T0_sq
    logp1 = -0.5 * (njt_t * njt_t) / T1_sq - 0.5 * log_T1_sq

    log_a = tf.math.log(phi_t + eps) + logp1
    log_b = tf.math.log(1.0 - phi_t + eps) + logp0

    m = tf.maximum(log_a, log_b)
    prob1 = tf.exp(log_a - m) / (tf.exp(log_a - m) + tf.exp(log_b - m))
    return prob1


# -----------------------------------------------------------------------------
# rw_mh_step
# -----------------------------------------------------------------------------
def test_rw_mh_step_scalar_shapes_and_types(rng):
    theta0 = tf.constant(0.3, dtype=tf.float64)
    k = tf.constant(0.1, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.square(theta)

    theta_new, accepted = rw_mh_step(theta0=theta0, logp_fn=logp_fn, k=k, rng=rng)

    assert theta_new.shape == ()
    assert accepted.shape == ()
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(theta_new)


def test_rw_mh_step_vector_shapes_and_accept_vector(rng):
    T = 5
    theta0 = tf.cast(tf.linspace(-0.3, 0.4, T), tf.float64)
    k = tf.constant(0.1, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.square(theta)

    theta_new, accepted = rw_mh_step(theta0=theta0, logp_fn=logp_fn, k=k, rng=rng)

    assert tuple(theta_new.shape) == (T,)
    assert tuple(accepted.shape) == (T,)
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(theta_new)


def test_rw_mh_step_no_move_when_k_zero(rng):
    k = tf.constant(0.0, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.square(theta)

    theta0 = tf.constant(-0.8, dtype=tf.float64)
    theta_new, accepted = rw_mh_step(theta0=theta0, logp_fn=logp_fn, k=k, rng=rng)
    assert float(theta_new.numpy()) == float(theta0.numpy())
    assert bool(tf.convert_to_tensor(accepted).numpy()) is True

    theta0 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)
    theta_new, accepted = rw_mh_step(theta0=theta0, logp_fn=logp_fn, k=k, rng=rng)
    assert np.allclose(theta_new.numpy(), theta0.numpy(), atol=0.0, rtol=0.0)
    assert bool(tf.reduce_all(tf.cast(accepted, tf.bool)).numpy()) is True


# -----------------------------------------------------------------------------
# tmh_step
# -----------------------------------------------------------------------------
def test_tmh_step_rejects_non_rank1_theta(rng):
    theta0 = tf.constant(0.1, dtype=tf.float64)  # rank-0 (invalid)
    k = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.reduce_sum(tf.square(theta))

    with pytest.raises(Exception):
        tmh_step(theta0=theta0, logp_fn=logp_fn, ridge=ridge, rng=rng, k=k)


def test_tmh_step_rejects_nonpositive_k_or_negative_ridge(rng):
    theta0 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.reduce_sum(tf.square(theta))

    ridge_ok = tf.constant(0.0, dtype=tf.float64)
    k_bad = tf.constant(0.0, dtype=tf.float64)
    with pytest.raises(Exception):
        tmh_step(theta0=theta0, logp_fn=logp_fn, ridge=ridge_ok, rng=rng, k=k_bad)

    k_ok = tf.constant(0.1, dtype=tf.float64)
    ridge_bad = tf.constant(-1e-6, dtype=tf.float64)
    with pytest.raises(Exception):
        tmh_step(theta0=theta0, logp_fn=logp_fn, ridge=ridge_bad, rng=rng, k=k_ok)


def test_tmh_step_quadratic_logp_runs_and_returns_finite(rng):
    theta0 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)
    k = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.reduce_sum(tf.square(theta))

    theta_new, accepted = tmh_step(
        theta0=theta0, logp_fn=logp_fn, ridge=ridge, rng=rng, k=k
    )

    assert tuple(theta_new.shape) == (3,)
    assert accepted.shape == ()
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(theta_new)


def test_tmh_step_fallback_on_nonfinite_logp_returns_theta0_and_rejects(rng):
    theta0 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)
    k = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    def logp_fn(theta):
        return tf.reduce_sum(tf.math.log(theta))  # non-finite when theta has negatives

    theta_new, accepted = tmh_step(
        theta0=theta0, logp_fn=logp_fn, ridge=ridge, rng=rng, k=k
    )

    assert np.allclose(theta_new.numpy(), theta0.numpy(), atol=0.0, rtol=0.0)
    assert bool(tf.convert_to_tensor(accepted).numpy()) is False


def test_tmh_step_ridge_zero_and_positive_both_work_on_quadratic(rng):
    theta0 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)
    k = tf.constant(0.1, dtype=tf.float64)

    def logp_fn(theta):
        return -0.5 * tf.reduce_sum(tf.square(theta))

    theta_new0, accepted0 = tmh_step(
        theta0=theta0,
        logp_fn=logp_fn,
        ridge=tf.constant(0.0, tf.float64),
        rng=rng,
        k=k,
    )
    theta_new1, accepted1 = tmh_step(
        theta0=theta0,
        logp_fn=logp_fn,
        ridge=tf.constant(1e-6, tf.float64),
        rng=rng,
        k=k,
    )

    assert_all_finite_tf(theta_new0, theta_new1)
    assert_bool_like_tf(accepted0)
    assert_bool_like_tf(accepted1)


# -----------------------------------------------------------------------------
# gibbs_gamma
# -----------------------------------------------------------------------------
def test_gibbs_gamma_shape_and_binary_support(rng, slab_spike_consts):
    T0_sq, T1_sq, log_T0_sq, log_T1_sq = slab_spike_consts
    njt_t = tf.constant([0.1, -0.5, 0.0, 0.3], dtype=tf.float64)
    phi_t = tf.constant(0.6, dtype=tf.float64)

    gamma_t = gibbs_gamma(
        njt_t=njt_t,
        phi_t=phi_t,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        log_T0_sq=log_T0_sq,
        log_T1_sq=log_T1_sq,
        rng=rng,
    )

    assert tuple(gamma_t.shape) == (4,)
    assert gamma_t.dtype == tf.float64
    assert_all_finite_tf(gamma_t)
    assert_binary_01_tf(gamma_t)


def test_gibbs_gamma_extreme_phi_no_nan(rng, slab_spike_consts):
    T0_sq, T1_sq, log_T0_sq, log_T1_sq = slab_spike_consts
    njt_t = tf.constant([0.2, -0.2, 1.0, -1.0], dtype=tf.float64)

    for phi_val in [1e-12, 1.0 - 1e-12]:
        phi_t = tf.constant(phi_val, dtype=tf.float64)
        gamma_t = gibbs_gamma(
            njt_t=njt_t,
            phi_t=phi_t,
            T0_sq=T0_sq,
            T1_sq=T1_sq,
            log_T0_sq=log_T0_sq,
            log_T1_sq=log_T1_sq,
            rng=rng,
        )
        assert_all_finite_tf(gamma_t)
        assert_binary_01_tf(gamma_t)


# -----------------------------------------------------------------------------
# gibbs_phi
# -----------------------------------------------------------------------------
def test_gibbs_phi_batched_shape_and_support(rng):
    gamma = tf.constant([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=tf.float64)
    a_phi = tf.constant(1.5, dtype=tf.float64)
    b_phi = tf.constant(2.5, dtype=tf.float64)

    phi = gibbs_phi(gamma=gamma, a_phi=a_phi, b_phi=b_phi, rng=rng)

    assert tuple(phi.shape) == (2,)
    assert phi.dtype == tf.float64
    assert_all_finite_tf(phi)
    assert_in_open_unit_interval_tf(phi)


def test_gibbs_phi_single_market_shape_and_support(rng):
    gamma = tf.constant([1.0, 0.0, 1.0, 1.0], dtype=tf.float64)
    a_phi = tf.constant(1.5, dtype=tf.float64)
    b_phi = tf.constant(2.5, dtype=tf.float64)

    phi = gibbs_phi(gamma=gamma, a_phi=a_phi, b_phi=b_phi, rng=rng)

    assert phi.shape == ()
    assert phi.dtype == tf.float64
    assert_all_finite_tf(phi)
    assert_in_open_unit_interval_tf(phi)


# -----------------------------------------------------------------------------
# Deterministic equivariance (no RNG): prob1 permutes with product permutation
# -----------------------------------------------------------------------------
def test_gibbs_gamma_prob1_equivariant_under_product_permutation(slab_spike_consts):
    T0_sq, T1_sq, log_T0_sq, log_T1_sq = slab_spike_consts
    njt_t = tf.constant([0.7, -0.2, 1.3, -0.9, 0.0], dtype=tf.float64)
    phi_t = tf.constant(0.4, dtype=tf.float64)

    perm = tf.constant([3, 0, 4, 1, 2], dtype=tf.int32)

    prob1 = _gibbs_gamma_prob1(
        njt_t=njt_t,
        phi_t=phi_t,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        log_T0_sq=log_T0_sq,
        log_T1_sq=log_T1_sq,
    )

    prob1_perm_inputs = _gibbs_gamma_prob1(
        njt_t=tf.gather(njt_t, perm),
        phi_t=phi_t,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        log_T0_sq=log_T0_sq,
        log_T1_sq=log_T1_sq,
    )

    assert np.allclose(
        prob1_perm_inputs.numpy(),
        tf.gather(prob1, perm).numpy(),
        atol=0.0,
        rtol=0.0,
    )
