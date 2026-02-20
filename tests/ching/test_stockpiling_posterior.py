# tests/ching/test_stockpiling_posterior.py
"""
Unit tests for `ching.stockpiling_posterior` (Phase-3 posterior utilities).

Updated posterior API (stateless):
  - logprior_normal(z, sigma_z)
  - loglik_mnj_from_theta(theta, inputs) -> (M,N,J)
  - predict_p_buy_mnjt_from_theta(theta, inputs) -> (M,N,J,T)

Key input names (updated):
  - a_mnjt        (M,N,J,T) int32
  - s_mjt         (M,J,T)   int32
  - inventory_maps: tuple returned by build_inventory_maps(I_max)
  - tol: float (python)
  - max_iter: int (python)
"""

from __future__ import annotations

import math
import os
import sys
from unittest.mock import patch

import numpy as np

# Ensure the local conftest helper is importable and can set env vars before TF import.
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import ching_conftest as cc  # noqa: F401  (side-effect: may set TF_CPP_MIN_LOG_LEVEL)
import tensorflow as tf

import ching.stockpiling_model as sm
import ching.stockpiling_posterior as sp


def test_logprior_normal_matches_closed_form() -> None:
    x = tf.constant([[0.0, 1.5], [-2.0, 3.0]], dtype=tf.float64)
    sigma = tf.constant(2.0, dtype=tf.float64)

    out = sp.logprior_normal(x, sigma).numpy()

    const = -0.5 * math.log(2.0 * math.pi)
    expected = const - math.log(2.0) - 0.5 * (x.numpy() / 2.0) ** 2
    expected = expected.astype(np.float64)

    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


def _minimal_inputs_Imax0(
    a_mnjt: np.ndarray,
    s_mjt: np.ndarray,
    ccp_by_state: np.ndarray,
) -> tuple[dict[str, object], tf.Tensor]:
    """
    Build posterior inputs for the special case I_max=0 (inventory always 0),
    and a patched CCP tensor of shape (M,N,J,S,1).

    Returns:
      inputs: dict for stockpiling_posterior
      ccp_buy: tf.Tensor (M,N,J,S,1)
    """
    assert a_mnjt.ndim == 4  # (M,N,J,T)
    assert s_mjt.ndim == 3  # (M,J,T)

    M, N, J, T = a_mnjt.shape
    assert s_mjt.shape == (M, J, T)

    S = int(ccp_by_state.shape[0])
    assert ccp_by_state.shape == (S,)
    assert np.all((ccp_by_state > 0.0) & (ccp_by_state < 1.0))

    # CCP tensor: (M,N,J,S,I) with I=1 since I_max=0
    ccp = np.zeros((M, N, J, S, 1), dtype=np.float64)
    for s in range(S):
        ccp[:, :, :, s, 0] = ccp_by_state[s]

    ccp_buy = tf.convert_to_tensor(ccp, dtype=tf.float64)

    # Dummy price process arrays (unused when solve_ccp_buy is patched, but must be well-shaped)
    price_vals_mj = tf.zeros((M, J, S), dtype=tf.float64)
    P_price_mj = tf.eye(S, dtype=tf.float64)[None, None, :, :]
    P_price_mj = tf.broadcast_to(P_price_mj, (M, J, S, S))

    inputs: dict[str, object] = {
        "a_mnjt": tf.convert_to_tensor(a_mnjt, dtype=tf.int32),
        "s_mjt": tf.convert_to_tensor(s_mjt, dtype=tf.int32),
        "u_mj": tf.zeros((M, J), dtype=tf.float64),
        "P_price_mj": P_price_mj,
        "price_vals_mj": price_vals_mj,
        "lambda_mn": tf.fill((M, N), tf.constant(0.5, dtype=tf.float64)),
        "waste_cost": tf.constant(0.0, dtype=tf.float64),
        "inventory_maps": sm.build_inventory_maps(I_max=0),
        "tol": 1e-10,
        "max_iter": 10,
        "pi_I0": tf.constant([1.0], dtype=tf.float64),
    }
    return inputs, ccp_buy


def test_loglik_mnj_from_theta_Imax0_matches_bernoulli_loglik() -> None:
    # Tiny deterministic panel with I_max=0 so inventory is degenerate.
    # Then the likelihood reduces to sum_t log( p(s_t) ) or log(1-p(s_t)).
    M, N, J, T, S = 1, 1, 1, 6, 2

    s_path = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int32)
    a_path = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)

    s_mjt = s_path.reshape(M, J, T)
    a_mnjt = a_path.reshape(M, N, J, T)

    p0, p1 = 0.2, 0.8
    inputs, ccp_buy = _minimal_inputs_Imax0(
        a_mnjt=a_mnjt, s_mjt=s_mjt, ccp_by_state=np.asarray([p0, p1])
    )

    theta_dummy = {
        "beta": tf.ones((M, J), dtype=tf.float64),
        "alpha": tf.ones((M, J), dtype=tf.float64),
        "v": tf.ones((M, J), dtype=tf.float64),
        "fc": tf.ones((M, J), dtype=tf.float64),
        "u_scale": tf.ones((M,), dtype=tf.float64),
    }

    with patch.object(sp, "solve_ccp_buy", return_value=(ccp_buy, None, None)):
        ll_mnj = sp.loglik_mnj_from_theta(theta=theta_dummy, inputs=inputs)

    assert tuple(ll_mnj.shape) == (M, N, J)

    # Closed form log-likelihood for degenerate inventory:
    p_by_t = np.where(s_path == 0, p0, p1)
    expected = np.sum(a_path * np.log(p_by_t) + (1 - a_path) * np.log(1.0 - p_by_t))
    np.testing.assert_allclose(
        ll_mnj.numpy().reshape(-1)[0], expected, rtol=0.0, atol=1e-12
    )


def test_predict_p_buy_mnjt_from_theta_Imax0_equals_state_ccp() -> None:
    # With I_max=0, predicted P(buy) at t equals CCP(s_t, I=0) regardless of history.
    M, N, J, T, S = 1, 1, 1, 5, 2

    s_path = np.asarray([1, 1, 0, 0, 1], dtype=np.int32)
    a_path = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)

    s_mjt = s_path.reshape(M, J, T)
    a_mnjt = a_path.reshape(M, N, J, T)

    p0, p1 = 0.3, 0.7
    inputs, ccp_buy = _minimal_inputs_Imax0(
        a_mnjt=a_mnjt, s_mjt=s_mjt, ccp_by_state=np.asarray([p0, p1])
    )

    theta_dummy = {
        "beta": tf.ones((M, J), dtype=tf.float64),
        "alpha": tf.ones((M, J), dtype=tf.float64),
        "v": tf.ones((M, J), dtype=tf.float64),
        "fc": tf.ones((M, J), dtype=tf.float64),
        "u_scale": tf.ones((M,), dtype=tf.float64),
    }

    with patch.object(sp, "solve_ccp_buy", return_value=(ccp_buy, None, None)):
        p_buy = sp.predict_p_buy_mnjt_from_theta(theta=theta_dummy, inputs=inputs)

    assert tuple(p_buy.shape) == (M, N, J, T)

    expected = np.where(s_path == 0, p0, p1).astype(np.float64)
    np.testing.assert_allclose(
        p_buy.numpy().reshape(-1), expected, rtol=0.0, atol=1e-12
    )
