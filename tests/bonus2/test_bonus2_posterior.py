# tests/bonus2/test_bonus2_posterior.py
"""
Unit tests for `bonus2.bonus2_posterior`.

Coverage:
- Normal(0, sigma^2) block prior (dropping constants): `logprior_normal_sum`
- Likelihood wrappers: `loglik_mnt` and `loglik` (shape, finiteness, sum consistency)
- Block log-target: `log_target_block_normal_prior` combines likelihood + specified block prior only
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import bonus2_conftest as bc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

from bonus2 import bonus2_model as bm
from bonus2 import bonus2_posterior as bp


def _build_inputs_tf(panel_np: dict[str, object]) -> dict[str, object]:
    """Convert a canonical panel dict into PosteriorInputs tensors."""
    y_mit = tf.convert_to_tensor(
        np.asarray(panel_np["y_mit"], dtype=np.int32), tf.int32
    )
    delta_mj = tf.convert_to_tensor(
        np.asarray(panel_np["delta_mj"], dtype=np.float64), tf.float64
    )
    is_weekend_t = tf.convert_to_tensor(
        np.asarray(panel_np["is_weekend_t"], dtype=np.int32), tf.int32
    )
    season_sin_kt = tf.convert_to_tensor(
        np.asarray(panel_np["season_sin_kt"], dtype=np.float64), tf.float64
    )
    season_cos_kt = tf.convert_to_tensor(
        np.asarray(panel_np["season_cos_kt"], dtype=np.float64), tf.float64
    )

    lookback = tf.convert_to_tensor(int(panel_np["lookback"]), tf.int32)
    decay = tf.convert_to_tensor(float(panel_np["decay"]), tf.float64)

    _m, n, _t = (int(x) for x in y_mit.shape)
    peer_adj_m = bm.build_peer_adjacency(
        neighbors_m=panel_np["neighbors_m"], n_consumers=n
    )

    return {
        "y_mit": y_mit,
        "delta_mj": delta_mj,
        "is_weekend_t": is_weekend_t,
        "season_sin_kt": season_sin_kt,
        "season_cos_kt": season_cos_kt,
        "peer_adj_m": peer_adj_m,
        "lookback": lookback,
        "decay": decay,
    }


def _z_zero_tf(dims: dict[str, int]) -> dict[str, tf.Tensor]:
    """Zero-filled z dict with correct block shapes and float64 dtype."""
    m, j, k = int(dims["M"]), int(dims["J"]), int(dims["K"])
    return {
        "z_beta_intercept_j": tf.zeros((j,), tf.float64),
        "z_beta_habit_j": tf.zeros((j,), tf.float64),
        "z_beta_peer_j": tf.zeros((j,), tf.float64),
        "z_beta_weekend_jw": tf.zeros((j, 2), tf.float64),
        "z_a_m": tf.zeros((m, k), tf.float64),
        "z_b_m": tf.zeros((m, k), tf.float64),
    }


def _sigma_z_tf(keys: tuple[str, ...], value: float = 1.0) -> dict[str, tf.Tensor]:
    """Scalar float64 sigma per z-block key."""
    v = tf.convert_to_tensor(float(value), tf.float64)
    return {k: v for k in keys}


def test_logprior_normal_sum_matches_formula() -> None:
    z = tf.convert_to_tensor(np.asarray([1.0, -2.0, 3.0], dtype=np.float64), tf.float64)

    sigma_scalar = tf.convert_to_tensor(2.0, tf.float64)
    out_scalar = bp.logprior_normal_sum(z_block=z, sigma=sigma_scalar).numpy()
    expected_scalar = -0.5 * np.sum((z.numpy() / 2.0) ** 2)
    np.testing.assert_allclose(out_scalar, expected_scalar, rtol=0, atol=1e-12)

    sigma_vec = tf.convert_to_tensor(
        np.asarray([1.0, 2.0, 4.0], dtype=np.float64), tf.float64
    )
    out_vec = bp.logprior_normal_sum(z_block=z, sigma=sigma_vec).numpy()
    expected_vec = -0.5 * np.sum((z.numpy() / sigma_vec.numpy()) ** 2)
    np.testing.assert_allclose(out_vec, expected_vec, rtol=0, atol=1e-12)

    # Nontrivial shape reduction
    z2 = tf.convert_to_tensor(
        np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float64), tf.float64
    )
    sigma2 = tf.convert_to_tensor(0.5, tf.float64)
    out2 = bp.logprior_normal_sum(z_block=z2, sigma=sigma2).numpy()
    expected2 = -0.5 * np.sum((z2.numpy() / 0.5) ** 2)
    np.testing.assert_allclose(out2, expected2, rtol=0, atol=1e-12)


def test_loglik_mnt_runs_end_to_end_and_is_finite_and_loglik_is_sum() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )
    inputs = _build_inputs_tf(panel)
    z = _z_zero_tf(dims)

    ll_mnt = bp.loglik_mnt(z=z, inputs=inputs)
    assert tuple(ll_mnt.shape) == (dims["M"], dims["N"], dims["T"])
    ll_mnt_np = ll_mnt.numpy()
    assert np.isfinite(ll_mnt_np).all()

    ll = bp.loglik(z=z, inputs=inputs)
    assert tuple(ll.shape) == ()
    ll_np = ll.numpy()
    assert np.isfinite(ll_np)

    np.testing.assert_allclose(ll_np, ll_mnt_np.sum(), rtol=0, atol=1e-10)


def test_loglik_runs_with_K0_no_seasonality() -> None:
    dims = bc.tiny_dims_k0()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="mixed",
        neighbor_pattern="ring",
        weekend_pattern="0101",
    )
    inputs = _build_inputs_tf(panel)
    z = _z_zero_tf(dims)

    ll_mnt = bp.loglik_mnt(z=z, inputs=inputs)
    assert tuple(ll_mnt.shape) == (dims["M"], dims["N"], dims["T"])
    assert np.isfinite(ll_mnt.numpy()).all()

    ll = bp.loglik(z=z, inputs=inputs)
    assert np.isfinite(ll.numpy())


def test_log_target_block_normal_prior_combines_ll_and_block_prior_with_patch() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())
    inputs = _build_inputs_tf(panel)

    z = _z_zero_tf(dims)
    sigma_z = _sigma_z_tf(bp.Z_KEYS, value=2.0)

    # Target block with nonzero entries.
    z_key = "z_beta_habit_j"
    z[z_key] = tf.convert_to_tensor(
        np.asarray([0.5, -1.0, 2.0], dtype=np.float64), tf.float64
    )

    # Make another block huge; should not matter because only the specified block prior is included.
    z["z_beta_peer_j"] = tf.fill((dims["J"],), tf.constant(1.0e6, tf.float64))

    ll_const = tf.constant(123.0, tf.float64)
    expected_lp = bp.logprior_normal_sum(z_block=z[z_key], sigma=sigma_z[z_key]).numpy()
    expected = ll_const.numpy() + expected_lp

    with patch("bonus2.bonus2_posterior.loglik", return_value=ll_const) as mocked:
        out = bp.log_target_block_normal_prior(
            z=z, z_key=z_key, inputs=inputs, sigma_z=sigma_z
        )
        out_np = out.numpy()

        np.testing.assert_allclose(out_np, expected, rtol=0, atol=1e-12)

        mocked.assert_called_once()
        # Ensure signature is respected.
        args, kwargs = mocked.call_args
        assert kwargs["z"] is z
        assert kwargs["inputs"] is inputs

    # Guard: changing other blocks should not change the target (since their priors are omitted).
    z2 = dict(z)
    z2["z_beta_peer_j"] = tf.fill((dims["J"],), tf.constant(-1.0e6, tf.float64))
    with patch("bonus2.bonus2_posterior.loglik", return_value=ll_const):
        out2 = bp.log_target_block_normal_prior(
            z=z2, z_key=z_key, inputs=inputs, sigma_z=sigma_z
        )
        np.testing.assert_allclose(out2.numpy(), expected, rtol=0, atol=1e-12)
