"""
Unit tests for lu.shrinkage.lu_updates.

Constraints
- No pytest fixture injection: all tests are plain functions with no fixture args.
- Shared assertion helpers are imported from `lu_conftest` (a normal Python module).
- Each test constructs its own tf.random.Generator to avoid cross-test coupling.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF
from lu.shrinkage.lu_updates import (
    update_E_bar,
    update_beta,
    update_gamma,
    update_njt,
    update_phi,
    update_r,
)
from lu_conftest import (
    assert_all_finite_tf,
    assert_binary_01_tf,
    assert_in_open_unit_interval_tf,
)

DTYPE = tf.float64
K = tf.constant(0.1, dtype=DTYPE)
RIDGE = tf.constant(1e-6, dtype=DTYPE)


def _posterior_config(n_draws: int, seed: int) -> LuPosteriorConfig:
    return LuPosteriorConfig(
        n_draws=int(n_draws),
        seed=int(seed),
        dtype=DTYPE,
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


def _posterior() -> LuPosteriorTF:
    return LuPosteriorTF(config=_posterior_config(n_draws=25, seed=123))


def _rng(seed: int) -> tf.random.Generator:
    return tf.random.Generator.from_seed(seed)


def _tiny_problem() -> dict:
    """
    Canonical tiny panel used in update-step tests.

    Shapes
    - pjt, wjt, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    - beta_p, beta_w, r: scalars
    - E_bar: (T,)
    - njt, gamma: (T, J)
    - phi: (T,)
    """
    T, J = 2, 3

    pjt = tf.constant([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]], dtype=DTYPE)
    wjt = tf.constant([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]], dtype=DTYPE)

    qjt = tf.constant([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]], dtype=DTYPE)
    q0t = tf.constant([20.0, 15.0], dtype=DTYPE)

    beta_p = tf.constant(-1.0, dtype=DTYPE)
    beta_w = tf.constant(0.3, dtype=DTYPE)
    r = tf.constant(0.0, dtype=DTYPE)

    E_bar = tf.constant([0.1, -0.2], dtype=DTYPE)
    njt = tf.constant([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]], dtype=DTYPE)

    gamma = tf.cast(njt > 0.0, DTYPE)
    phi = tf.constant([0.6, 0.4], dtype=DTYPE)

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


def _assert_bool_like_tf(x: tf.Tensor) -> None:
    """
    Accept either tf.bool tensors, or numeric tensors taking values in {0,1}.
    Works for scalars and vectors.
    """
    x_t = tf.convert_to_tensor(x)
    if x_t.dtype == tf.bool:
        return

    xv = x_t.numpy()
    if xv.size == 0:
        raise AssertionError("Expected non-empty bool-like tensor.")

    xv_r = np.round(xv)
    if not np.all(np.isfinite(xv)):
        raise AssertionError("Bool-like tensor contains non-finite values.")
    if not np.all(np.abs(xv - xv_r) <= 1e-12):
        raise AssertionError("Bool-like tensor contains values not close to {0,1}.")
    if not np.all((xv_r == 0.0) | (xv_r == 1.0)):
        raise AssertionError("Bool-like tensor contains values outside {0,1}.")


def test_update_beta_returns_scalars_and_bool_and_is_finite():
    posterior = _posterior()
    rng = _rng(1)
    tiny = _tiny_problem()

    beta_p_new, beta_w_new, accepted = update_beta(
        posterior=posterior,
        rng=rng,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_beta=K,
        ridge=RIDGE,
    )

    assert beta_p_new.shape == ()
    assert beta_w_new.shape == ()
    assert beta_p_new.dtype == DTYPE
    assert beta_w_new.dtype == DTYPE
    _assert_bool_like_tf(accepted)
    assert_all_finite_tf(beta_p_new, beta_w_new)


def test_update_r_returns_scalar_and_bool_and_is_finite():
    posterior = _posterior()
    rng = _rng(2)
    tiny = _tiny_problem()

    r_new, accepted = update_r(
        posterior=posterior,
        rng=rng,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_r=K,
    )

    assert r_new.shape == ()
    assert r_new.dtype == DTYPE
    _assert_bool_like_tf(accepted)
    assert_all_finite_tf(r_new)


def test_update_E_bar_returns_vector_and_accept_vector():
    posterior = _posterior()
    rng = _rng(3)
    tiny = _tiny_problem()
    T = tiny["T"]

    E_bar_new, accepted = update_E_bar(
        posterior=posterior,
        rng=rng,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        phi=tiny["phi"],
        k_E_bar=K,
    )

    assert tuple(E_bar_new.shape) == (T,)
    assert E_bar_new.dtype == DTYPE
    assert tuple(accepted.shape) == (T,)
    _assert_bool_like_tf(accepted)
    assert_all_finite_tf(E_bar_new)


def test_update_njt_returns_matrix_and_acc_sum_bounds():
    posterior = _posterior()
    rng = _rng(4)
    tiny = _tiny_problem()
    T, J = tiny["T"], tiny["J"]

    njt_new, acc_sum = update_njt(
        posterior=posterior,
        rng=rng,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        phi=tiny["phi"],
        k_njt=K,
        ridge=RIDGE,
    )

    assert tuple(njt_new.shape) == (T, J)
    assert njt_new.dtype == DTYPE
    assert acc_sum.shape == ()
    assert acc_sum.dtype == DTYPE

    assert_all_finite_tf(njt_new, acc_sum)

    acc = float(acc_sum.numpy())
    assert 0.0 <= acc <= float(T), f"acc_sum out of bounds: {acc} not in [0, {T}]"


def test_update_gamma_returns_binary_matrix():
    posterior = _posterior()
    rng = _rng(5)
    tiny = _tiny_problem()
    T, J = tiny["T"], tiny["J"]

    gamma_new = update_gamma(
        posterior=posterior,
        rng=rng,
        njt=tiny["njt"],
        phi=tiny["phi"],
    )

    assert tuple(gamma_new.shape) == (T, J)
    assert gamma_new.dtype == DTYPE
    assert_all_finite_tf(gamma_new)
    assert_binary_01_tf(gamma_new)


def test_update_phi_returns_vector_in_open_unit_interval():
    posterior = _posterior()
    rng = _rng(6)
    tiny = _tiny_problem()
    T = tiny["T"]

    phi_new = update_phi(
        posterior=posterior,
        rng=rng,
        gamma=tiny["gamma"],
    )

    assert tuple(phi_new.shape) == (T,)
    assert phi_new.dtype == DTYPE
    assert_all_finite_tf(phi_new)
    assert_in_open_unit_interval_tf(phi_new)
