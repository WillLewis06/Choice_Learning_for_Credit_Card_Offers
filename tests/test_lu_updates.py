# tests/test_lu_updates.py
import tensorflow as tf

from conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
    assert_bool_like_tf,
    assert_in_open_unit_interval_tf,
)
from market_shock_estimators.lu_updates import (
    update_E_bar,
    update_beta,
    update_gamma,
    update_njt,
    update_phi,
    update_r,
)


def test_update_beta_returns_scalars_and_bool_and_is_finite(
    posterior, rng, tiny_problem
):
    k_beta = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    beta_p_new, beta_w_new, accepted = update_beta(
        posterior=posterior,
        rng=rng,
        qjt=tiny_problem["qjt"],
        q0t=tiny_problem["q0t"],
        pjt=tiny_problem["pjt"],
        wjt=tiny_problem["wjt"],
        beta_p=tiny_problem["beta_p"],
        beta_w=tiny_problem["beta_w"],
        r=tiny_problem["r"],
        E_bar=tiny_problem["E_bar"],
        njt=tiny_problem["njt"],
        k_beta=k_beta,
        ridge=ridge,
    )

    assert beta_p_new.shape == ()
    assert beta_w_new.shape == ()
    assert beta_p_new.dtype == tf.float64
    assert beta_w_new.dtype == tf.float64
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(beta_p_new, beta_w_new)


def test_update_r_returns_scalar_and_bool_and_is_finite(posterior, rng, tiny_problem):
    k_r = tf.constant(0.1, dtype=tf.float64)

    r_new, accepted = update_r(
        posterior=posterior,
        rng=rng,
        qjt=tiny_problem["qjt"],
        q0t=tiny_problem["q0t"],
        pjt=tiny_problem["pjt"],
        wjt=tiny_problem["wjt"],
        beta_p=tiny_problem["beta_p"],
        beta_w=tiny_problem["beta_w"],
        r=tiny_problem["r"],
        E_bar=tiny_problem["E_bar"],
        njt=tiny_problem["njt"],
        k_r=k_r,
    )

    assert r_new.shape == ()
    assert r_new.dtype == tf.float64
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(r_new)


def test_update_E_bar_returns_vector_and_accept_vector(posterior, rng, tiny_problem):
    T = tiny_problem["T"]
    k_E_bar = tf.constant(0.1, dtype=tf.float64)

    E_bar_new, accepted = update_E_bar(
        posterior=posterior,
        rng=rng,
        qjt=tiny_problem["qjt"],
        q0t=tiny_problem["q0t"],
        pjt=tiny_problem["pjt"],
        wjt=tiny_problem["wjt"],
        beta_p=tiny_problem["beta_p"],
        beta_w=tiny_problem["beta_w"],
        r=tiny_problem["r"],
        E_bar=tiny_problem["E_bar"],
        njt=tiny_problem["njt"],
        gamma=tiny_problem["gamma"],
        phi=tiny_problem["phi"],
        k_E_bar=k_E_bar,
    )

    assert tuple(E_bar_new.shape) == (T,)
    assert E_bar_new.dtype == tf.float64
    assert tuple(accepted.shape) == (T,)
    assert_bool_like_tf(accepted)
    assert_all_finite_tf(E_bar_new)


def test_update_njt_returns_matrix_and_acc_sum_bounds(posterior, rng, tiny_problem):
    T, J = tiny_problem["T"], tiny_problem["J"]
    k_njt = tf.constant(0.1, dtype=tf.float64)
    ridge = tf.constant(1e-6, dtype=tf.float64)

    njt_new, acc_sum = update_njt(
        posterior=posterior,
        rng=rng,
        qjt=tiny_problem["qjt"],
        q0t=tiny_problem["q0t"],
        pjt=tiny_problem["pjt"],
        wjt=tiny_problem["wjt"],
        beta_p=tiny_problem["beta_p"],
        beta_w=tiny_problem["beta_w"],
        r=tiny_problem["r"],
        E_bar=tiny_problem["E_bar"],
        njt=tiny_problem["njt"],
        gamma=tiny_problem["gamma"],
        phi=tiny_problem["phi"],
        k_njt=k_njt,
        ridge=ridge,
    )

    assert tuple(njt_new.shape) == (T, J)
    assert njt_new.dtype == tf.float64
    assert acc_sum.shape == ()
    assert acc_sum.dtype == tf.float64

    assert_all_finite_tf(njt_new, acc_sum)

    acc = float(acc_sum.numpy())
    assert 0.0 <= acc <= float(T), f"acc_sum out of bounds: {acc} not in [0, {T}]"


def test_update_gamma_returns_binary_matrix(posterior, rng, tiny_problem):
    T, J = tiny_problem["T"], tiny_problem["J"]

    gamma_new = update_gamma(
        posterior=posterior,
        rng=rng,
        njt=tiny_problem["njt"],
        phi=tiny_problem["phi"],
    )

    assert tuple(gamma_new.shape) == (T, J)
    assert gamma_new.dtype == tf.float64
    assert_all_finite_tf(gamma_new)
    assert_binary_01_tf(gamma_new)


def test_update_phi_returns_vector_in_open_unit_interval(posterior, rng, tiny_problem):
    T = tiny_problem["T"]

    phi_new = update_phi(
        posterior=posterior,
        rng=rng,
        gamma=tiny_problem["gamma"],
    )

    assert tuple(phi_new.shape) == (T,)
    assert phi_new.dtype == tf.float64
    assert_all_finite_tf(phi_new)
    assert_in_open_unit_interval_tf(phi_new)
