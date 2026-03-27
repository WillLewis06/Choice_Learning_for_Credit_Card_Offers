"""
Unit tests for lu.shrinkage.lu_updates.

This file targets the refactored update API:
- beta_one_step
- r_one_step
- E_bar_one_step
- njt_one_step

The old handwritten update functions, phi update, and gamma update are no
longer part of this module and are not tested here.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.shrinkage.lu_posterior import LuPosteriorConfig, LuPosteriorTF
from lu.shrinkage.lu_updates import (
    E_bar_one_step,
    beta_one_step,
    njt_one_step,
    r_one_step,
)

DTYPE = tf.float64
SEED_DTYPE = tf.int32

K_BETA = tf.constant(0.1, dtype=DTYPE)
K_R = tf.constant(0.1, dtype=DTYPE)
K_E_BAR = tf.constant(0.1, dtype=DTYPE)
K_NJT = tf.constant(0.1, dtype=DTYPE)


def _tf(x) -> tf.Tensor:
    """Create a tf.float64 constant."""
    return tf.constant(x, dtype=DTYPE)


def _seed(a: int, b: int) -> tf.Tensor:
    """Create a stateless seed tensor."""
    return tf.constant([a, b], dtype=SEED_DTYPE)


def _posterior_config(n_draws: int, seed: int) -> LuPosteriorConfig:
    """Build a small posterior config for update-step tests."""
    return LuPosteriorConfig(
        n_draws=int(n_draws),
        seed=int(seed),
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
    """Create the refactored posterior object."""
    return LuPosteriorTF(config=_posterior_config(n_draws=25, seed=123))


def _tiny_problem() -> dict:
    """
    Canonical tiny panel used in one-step update tests.

    Shapes
    - pjt, wjt, qjt: (T, J) with T=2, J=3
    - q0t: (T,)
    - beta_p, beta_w, r: scalars
    - E_bar: (T,)
    - njt, gamma: (T, J)
    """
    T, J = 2, 3

    pjt = _tf([[1.0, 1.2, 0.8], [0.9, 1.1, 1.3]])
    wjt = _tf([[0.5, 0.7, 0.6], [0.4, 0.9, 0.3]])

    qjt = _tf([[10.0, 5.0, 2.0], [3.0, 7.0, 1.0]])
    q0t = _tf([20.0, 15.0])

    beta_p = _tf(-1.0)
    beta_w = _tf(0.3)
    r = _tf(0.0)

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
        "beta_p": beta_p,
        "beta_w": beta_w,
        "r": r,
        "E_bar": E_bar,
        "njt": njt,
        "gamma": gamma,
    }


def _assert_all_finite_tf(*xs: tf.Tensor) -> None:
    """Assert that all supplied tensors contain only finite values."""
    for x in xs:
        x_np = tf.convert_to_tensor(x).numpy()
        if not np.all(np.isfinite(x_np)):
            raise AssertionError("Tensor contains non-finite values.")


def _assert_binary_accept_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance indicator in {0.0, 1.0}."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(
            f"Expected scalar acceptance indicator, got shape {x_t.shape}."
        )
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")

    x_val = float(x_t.numpy())
    if x_val not in (0.0, 1.0):
        raise AssertionError(
            f"Expected acceptance indicator in {{0.0, 1.0}}, got {x_val}."
        )


def _assert_accept_rate_scalar(x: tf.Tensor) -> None:
    """Assert a scalar acceptance rate in [0.0, 1.0]."""
    x_t = tf.convert_to_tensor(x)
    if x_t.shape != ():
        raise AssertionError(f"Expected scalar acceptance rate, got shape {x_t.shape}.")
    if x_t.dtype != DTYPE:
        raise AssertionError(f"Expected dtype {DTYPE}, got {x_t.dtype}.")

    x_val = float(x_t.numpy())
    if not np.isfinite(x_val):
        raise AssertionError("Acceptance rate is not finite.")
    if not (0.0 <= x_val <= 1.0):
        raise AssertionError(f"Acceptance rate out of bounds: {x_val} not in [0, 1].")


def test_beta_one_step_returns_scalars_and_binary_accept_indicator():
    posterior = _posterior()
    tiny = _tiny_problem()

    beta_p_new, beta_w_new, accepted = beta_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_beta=K_BETA,
        seed=_seed(1, 2),
    )

    assert beta_p_new.shape == ()
    assert beta_w_new.shape == ()
    assert beta_p_new.dtype == DTYPE
    assert beta_w_new.dtype == DTYPE
    _assert_all_finite_tf(beta_p_new, beta_w_new)
    _assert_binary_accept_scalar(accepted)


def test_beta_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(11, 12)

    out_1 = beta_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_beta=K_BETA,
        seed=seed,
    )
    out_2 = beta_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_beta=K_BETA,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])
    tf.debugging.assert_equal(out_1[2], out_2[2])


def test_r_one_step_returns_scalar_and_binary_accept_indicator():
    posterior = _posterior()
    tiny = _tiny_problem()

    r_new, accepted = r_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_r=K_R,
        seed=_seed(3, 4),
    )

    assert r_new.shape == ()
    assert r_new.dtype == DTYPE
    _assert_all_finite_tf(r_new)
    _assert_binary_accept_scalar(accepted)


def test_r_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(13, 14)

    out_1 = r_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_r=K_R,
        seed=seed,
    )
    out_2 = r_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_r=K_R,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])


def test_E_bar_one_step_returns_vector_and_accept_rate():
    posterior = _posterior()
    tiny = _tiny_problem()
    T = tiny["T"]

    E_bar_new, accept_rate = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=_seed(5, 6),
    )

    assert tuple(E_bar_new.shape) == (T,)
    assert E_bar_new.dtype == DTYPE
    _assert_all_finite_tf(E_bar_new)
    _assert_accept_rate_scalar(accept_rate)


def test_E_bar_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(15, 16)

    out_1 = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=seed,
    )
    out_2 = E_bar_one_step(
        posterior=posterior,
        qjt=tiny["qjt"],
        q0t=tiny["q0t"],
        pjt=tiny["pjt"],
        wjt=tiny["wjt"],
        beta_p=tiny["beta_p"],
        beta_w=tiny["beta_w"],
        r=tiny["r"],
        E_bar=tiny["E_bar"],
        njt=tiny["njt"],
        k_E_bar=K_E_BAR,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])


def test_njt_one_step_returns_matrix_and_accept_rate():
    posterior = _posterior()
    tiny = _tiny_problem()
    T, J = tiny["T"], tiny["J"]

    njt_new, accept_rate = njt_one_step(
        posterior=posterior,
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
        k_njt=K_NJT,
        seed=_seed(7, 8),
    )

    assert tuple(njt_new.shape) == (T, J)
    assert njt_new.dtype == DTYPE
    _assert_all_finite_tf(njt_new)
    _assert_accept_rate_scalar(accept_rate)


def test_njt_one_step_is_deterministic_for_fixed_seed():
    posterior = _posterior()
    tiny = _tiny_problem()
    seed = _seed(17, 18)

    out_1 = njt_one_step(
        posterior=posterior,
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
        k_njt=K_NJT,
        seed=seed,
    )
    out_2 = njt_one_step(
        posterior=posterior,
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
        k_njt=K_NJT,
        seed=seed,
    )

    tf.debugging.assert_equal(out_1[0], out_2[0])
    tf.debugging.assert_equal(out_1[1], out_2[1])
