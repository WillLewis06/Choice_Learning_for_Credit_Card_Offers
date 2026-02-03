import numpy as np
import pytest
import tensorflow as tf

from market_shock_estimators.lu_posterior import LuPosteriorTF
from market_shock_estimators.lu_updates import (
    update_beta,
    update_r,
    update_E_bar,
    update_njt,
    update_gamma,
    update_phi,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _assert_all_finite(*tensors) -> None:
    for x in tensors:
        x = tf.convert_to_tensor(x)
        ok = tf.reduce_all(tf.math.is_finite(x))
        assert bool(ok.numpy()), "Found non-finite values."


def _is_bool_like_tensor(x: tf.Tensor) -> bool:
    x = tf.convert_to_tensor(x)
    if x.dtype == tf.bool:
        return True
    # allow numeric 0/1
    if x.dtype.is_floating or x.dtype.is_integer:
        xv = x.numpy()
        uniq = np.unique(xv)
        return np.all(np.isin(uniq, [0, 1]))
    return False


def _assert_bool_like(x: tf.Tensor) -> None:
    assert _is_bool_like_tensor(
        x
    ), f"Expected bool-like (bool or 0/1), got dtype={x.dtype}."


def _assert_binary_01(x: tf.Tensor) -> None:
    x = tf.convert_to_tensor(x)
    xv = x.numpy()
    uniq = np.unique(xv)
    assert np.all(
        np.isin(uniq, [0.0, 1.0])
    ), f"Expected exactly 0/1 values, got {uniq}."


def _assert_in_open_unit_interval(x: tf.Tensor) -> None:
    x = tf.convert_to_tensor(x)
    xv = x.numpy()
    assert np.all(xv > 0.0), f"Expected > 0, got min={xv.min()}"
    assert np.all(xv < 1.0), f"Expected < 1, got max={xv.max()}"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def posterior():
    # Small n_draws for speed; correctness here is about wiring/contracts.
    return LuPosteriorTF(n_draws=25, seed=123, dtype=tf.float64)


@pytest.fixture
def rng():
    return tf.random.Generator.from_seed(123)


@pytest.fixture
def tiny_problem():
    """
    Minimal consistent tensors (T=2, J=3) with non-degenerate counts.
    """
    T, J = 2, 3

    pjt = tf.constant([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=tf.float64)
    wjt = tf.constant([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=tf.float64)

    qjt = tf.constant([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=tf.float64)
    q0t = tf.constant([20.0, 15.0], dtype=tf.float64)

    beta_p = tf.constant(-1.0, dtype=tf.float64)
    beta_w = tf.constant(0.3, dtype=tf.float64)
    r = tf.constant(0.0, dtype=tf.float64)

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
        "beta_p": beta_p,
        "beta_w": beta_w,
        "r": r,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
        "phi": phi,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
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
    _assert_bool_like(accepted)
    _assert_all_finite(beta_p_new, beta_w_new)


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
    _assert_bool_like(accepted)
    _assert_all_finite(r_new)


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
    _assert_bool_like(accepted)
    _assert_all_finite(E_bar_new)


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

    _assert_all_finite(njt_new, acc_sum)

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
    _assert_all_finite(gamma_new)
    _assert_binary_01(gamma_new)


def test_update_phi_returns_vector_in_open_unit_interval(posterior, rng, tiny_problem):
    T = tiny_problem["T"]

    phi_new = update_phi(
        posterior=posterior,
        rng=rng,
        gamma=tiny_problem["gamma"],
    )

    assert tuple(phi_new.shape) == (T,)
    assert phi_new.dtype == tf.float64
    _assert_all_finite(phi_new)
    _assert_in_open_unit_interval(phi_new)
