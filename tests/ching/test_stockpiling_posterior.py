# tests/ching/test_stockpiling_posterior.py
"""Unit tests for `ching.stockpiling_posterior` (Phase-3 posterior utilities).

This file targets the refactored posterior API:
- StockpilingPosteriorConfig
- StockpilingPosteriorTF
- method-based prior, likelihood, block-posterior, and prediction evaluation

The main public methods exercised here are:
- logprior_beta
- logprior_alpha_vec
- logprior
- loglik_mnj
- loglik
- beta_block_logpost
- alpha_block_logpost
- v_block_logpost
- fc_block_logpost
- u_scale_block_logpost
- joint_logpost
- predict_p_buy_mnjt
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

import ching.stockpiling_posterior as sp

DTYPE = tf.float64
ATOL = 1e-10
RTOL = 0.0

STANDARD_BUNDLE = cc.posterior_bundle_tf()
STANDARD_POSTERIOR = STANDARD_BUNDLE["posterior"]
STANDARD_DIMS = STANDARD_BUNDLE["dims"]
STANDARD_Z = {
    k: tf.convert_to_tensor(v, dtype=DTYPE)
    for k, v in cc.z_blocks_np(STANDARD_DIMS).items()
}


def _normal_logpdf_zero_mean(x: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Closed-form zero-mean Normal log density."""
    two_pi = tf.constant(2.0 * math.pi, dtype=DTYPE)
    return -0.5 * (tf.math.log(two_pi) + 2.0 * tf.math.log(sigma)) - 0.5 * tf.square(
        x / sigma
    )


def _minimal_posterior_Imax0(
    a_mnjt: np.ndarray,
    s_mjt: np.ndarray,
    ccp_by_state: np.ndarray,
) -> tuple[sp.StockpilingPosteriorTF, tf.Tensor, dict[str, tf.Tensor]]:
    """
    Build a refactored posterior object for the special case I_max=0.

    Returns:
      posterior: StockpilingPosteriorTF
      ccp_buy:   patched CCP tensor of shape (M, N, J, S, 1)
      z_state:   dict with keys z_beta, z_alpha, z_v, z_fc, z_u_scale
    """
    assert a_mnjt.ndim == 4  # (M, N, J, T)
    assert s_mjt.ndim == 3  # (M, J, T)

    M, N, J, T = a_mnjt.shape
    assert s_mjt.shape == (M, J, T)

    ccp_by_state = np.asarray(ccp_by_state, dtype=np.float64)
    S = int(ccp_by_state.shape[0])
    assert ccp_by_state.shape == (S,)
    assert np.all((ccp_by_state > 0.0) & (ccp_by_state < 1.0))

    ccp = np.zeros((M, N, J, S, 1), dtype=np.float64)
    for s in range(S):
        ccp[:, :, :, s, 0] = ccp_by_state[s]
    ccp_buy = tf.convert_to_tensor(ccp, dtype=DTYPE)

    # These arrays are only required to satisfy the posterior constructor and
    # solve_ccp_buy signature. In the patched tests below, solve_ccp_buy is mocked.
    price_vals_mj = tf.zeros((M, J, S), dtype=DTYPE)
    P_price_mj = tf.eye(S, dtype=DTYPE)[None, None, :, :]
    P_price_mj = tf.broadcast_to(P_price_mj, (M, J, S, S))
    pi_I0 = tf.constant([1.0], dtype=DTYPE)

    config = sp.StockpilingPosteriorConfig(
        tol=1e-10,
        max_iter=10,
        eps=1e-12,
        sigma_z_beta=0.5,
        sigma_z_alpha=0.5,
        sigma_z_v=0.5,
        sigma_z_fc=0.5,
        sigma_z_u_scale=0.5,
    )

    posterior = sp.StockpilingPosteriorTF(
        config=config,
        a_mnjt=tf.convert_to_tensor(a_mnjt, dtype=tf.int32),
        s_mjt=tf.convert_to_tensor(s_mjt, dtype=tf.int32),
        u_mj=tf.zeros((M, J), dtype=DTYPE),
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        lambda_mn=tf.fill((M, N), tf.constant(0.5, dtype=DTYPE)),
        waste_cost=tf.constant(0.0, dtype=DTYPE),
        pi_I0=pi_I0,
        inventory_maps=cc.inventory_maps_tf(0),
    )

    z_state = {
        "z_beta": tf.constant(0.0, dtype=DTYPE),
        "z_alpha": tf.zeros((J,), dtype=DTYPE),
        "z_v": tf.zeros((J,), dtype=DTYPE),
        "z_fc": tf.zeros((J,), dtype=DTYPE),
        "z_u_scale": tf.zeros((M,), dtype=DTYPE),
    }
    return posterior, ccp_buy, z_state


def test_logprior_beta_matches_closed_form() -> None:
    """logprior_beta should match the closed-form scalar Normal prior."""
    z_beta = tf.constant(1.25, dtype=DTYPE)
    sigma = STANDARD_POSTERIOR.sigma_z_beta

    out = STANDARD_POSTERIOR.logprior_beta(z_beta)
    expected = _normal_logpdf_zero_mean(z_beta, sigma)

    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_logprior_alpha_vec_shape_and_closed_form() -> None:
    """logprior_alpha_vec should be elementwise Normal log density over products."""
    J = int(STANDARD_DIMS["J"])
    z_alpha = tf.constant(np.linspace(-0.4, 0.6, J), dtype=DTYPE)
    sigma = STANDARD_POSTERIOR.sigma_z_alpha

    out = STANDARD_POSTERIOR.logprior_alpha_vec(z_alpha)
    expected = _normal_logpdf_zero_mean(z_alpha, sigma)

    assert tuple(out.shape) == (J,)
    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_logprior_equals_sum_of_component_priors() -> None:
    """logprior should equal the sum of all block prior contributions."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.logprior(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    expected = (
        STANDARD_POSTERIOR.logprior_beta(z["z_beta"])
        + tf.reduce_sum(STANDARD_POSTERIOR.logprior_alpha_vec(z["z_alpha"]))
        + tf.reduce_sum(STANDARD_POSTERIOR.logprior_v_vec(z["z_v"]))
        + tf.reduce_sum(STANDARD_POSTERIOR.logprior_fc_vec(z["z_fc"]))
        + tf.reduce_sum(STANDARD_POSTERIOR.logprior_u_scale_vec(z["z_u_scale"]))
    )

    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_loglik_mnj_Imax0_matches_bernoulli_loglik() -> None:
    """
    With I_max=0, inventory is degenerate and the likelihood reduces to the
    Bernoulli log-likelihood implied by the observed price state path.
    """
    M, N, J, T, S = 1, 1, 1, 6, 2

    s_path = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int32)
    a_path = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32)

    s_mjt = s_path.reshape(M, J, T)
    a_mnjt = a_path.reshape(M, N, J, T)

    p0, p1 = 0.2, 0.8
    posterior, ccp_buy, z = _minimal_posterior_Imax0(
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        ccp_by_state=np.asarray([p0, p1], dtype=np.float64),
    )

    with patch.object(sp, "solve_ccp_buy", return_value=(ccp_buy, None, None)):
        ll_mnj = posterior.loglik_mnj(
            z_beta=z["z_beta"],
            z_alpha=z["z_alpha"],
            z_v=z["z_v"],
            z_fc=z["z_fc"],
            z_u_scale=z["z_u_scale"],
        )

    assert tuple(ll_mnj.shape) == (M, N, J)

    p_by_t = np.where(s_path == 0, p0, p1)
    expected = np.sum(a_path * np.log(p_by_t) + (1.0 - a_path) * np.log(1.0 - p_by_t))
    np.testing.assert_allclose(
        ll_mnj.numpy().reshape(-1)[0],
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_predict_p_buy_mnjt_Imax0_equals_state_ccp() -> None:
    """
    With I_max=0, filtered predicted P(buy) at each t equals CCP(s_t, I=0),
    regardless of purchase history.
    """
    M, N, J, T, S = 1, 1, 1, 5, 2

    s_path = np.asarray([1, 1, 0, 0, 1], dtype=np.int32)
    a_path = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)

    s_mjt = s_path.reshape(M, J, T)
    a_mnjt = a_path.reshape(M, N, J, T)

    p0, p1 = 0.3, 0.7
    posterior, ccp_buy, z = _minimal_posterior_Imax0(
        a_mnjt=a_mnjt,
        s_mjt=s_mjt,
        ccp_by_state=np.asarray([p0, p1], dtype=np.float64),
    )

    with patch.object(sp, "solve_ccp_buy", return_value=(ccp_buy, None, None)):
        p_buy = posterior.predict_p_buy_mnjt(
            z_beta=z["z_beta"],
            z_alpha=z["z_alpha"],
            z_v=z["z_v"],
            z_fc=z["z_fc"],
            z_u_scale=z["z_u_scale"],
        )

    assert tuple(p_buy.shape) == (M, N, J, T)

    expected = np.where(s_path == 0, p0, p1).astype(np.float64)
    np.testing.assert_allclose(
        p_buy.numpy().reshape(-1),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_loglik_equals_reduce_sum_of_loglik_mnj() -> None:
    """loglik should equal the sum over all (m, n, j) log-likelihood terms."""
    z = STANDARD_Z

    ll_mnj = STANDARD_POSTERIOR.loglik_mnj(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    ll = STANDARD_POSTERIOR.loglik(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    expected = tf.reduce_sum(ll_mnj)
    tf.debugging.assert_near(ll, expected, atol=ATOL, rtol=RTOL)


def test_beta_block_logpost_equals_loglik_plus_logprior_beta() -> None:
    """beta_block_logpost should equal total loglik plus the beta prior only."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.beta_block_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = STANDARD_POSTERIOR.loglik(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    ) + STANDARD_POSTERIOR.logprior_beta(z["z_beta"])

    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_alpha_block_logpost_equals_reduced_loglik_plus_logprior_alpha_vec() -> None:
    """alpha_block_logpost should equal per-product reduced loglik plus alpha prior."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.alpha_block_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    ll_mnj = STANDARD_POSTERIOR.loglik_mnj(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = tf.reduce_sum(
        ll_mnj, axis=[0, 1]
    ) + STANDARD_POSTERIOR.logprior_alpha_vec(z["z_alpha"])

    assert tuple(out.shape) == (int(STANDARD_DIMS["J"]),)
    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_v_block_logpost_equals_reduced_loglik_plus_logprior_v_vec() -> None:
    """v_block_logpost should equal per-product reduced loglik plus v prior."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.v_block_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    ll_mnj = STANDARD_POSTERIOR.loglik_mnj(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = tf.reduce_sum(ll_mnj, axis=[0, 1]) + STANDARD_POSTERIOR.logprior_v_vec(
        z["z_v"]
    )

    assert tuple(out.shape) == (int(STANDARD_DIMS["J"]),)
    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_fc_block_logpost_equals_reduced_loglik_plus_logprior_fc_vec() -> None:
    """fc_block_logpost should equal per-product reduced loglik plus fc prior."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.fc_block_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    ll_mnj = STANDARD_POSTERIOR.loglik_mnj(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = tf.reduce_sum(ll_mnj, axis=[0, 1]) + STANDARD_POSTERIOR.logprior_fc_vec(
        z["z_fc"]
    )

    assert tuple(out.shape) == (int(STANDARD_DIMS["J"]),)
    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_u_scale_block_logpost_equals_market_reduced_loglik_plus_logprior_u_scale_vec() -> (
    None
):
    """u_scale_block_logpost should equal per-market reduced loglik plus u_scale prior."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.u_scale_block_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    ll_mnj = STANDARD_POSTERIOR.loglik_mnj(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = tf.reduce_sum(
        ll_mnj, axis=[1, 2]
    ) + STANDARD_POSTERIOR.logprior_u_scale_vec(z["z_u_scale"])

    assert tuple(out.shape) == (int(STANDARD_DIMS["M"]),)
    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)


def test_joint_logpost_equals_loglik_plus_logprior() -> None:
    """joint_logpost should equal total loglik plus total prior."""
    z = STANDARD_Z

    out = STANDARD_POSTERIOR.joint_logpost(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )
    expected = STANDARD_POSTERIOR.loglik(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    ) + STANDARD_POSTERIOR.logprior(
        z_beta=z["z_beta"],
        z_alpha=z["z_alpha"],
        z_v=z["z_v"],
        z_fc=z["z_fc"],
        z_u_scale=z["z_u_scale"],
    )

    tf.debugging.assert_near(out, expected, atol=ATOL, rtol=RTOL)
