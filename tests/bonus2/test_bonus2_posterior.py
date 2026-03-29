# tests/bonus2/test_bonus2_posterior.py
"""
Unit tests for `bonus2.bonus2_posterior`.

Coverage:
- `Bonus2PosteriorTF` likelihood wrappers: `loglik_mnt` and `loglik`
- Public block priors against the closed-form Gaussian quadratic prior
- `logprior_all` as the sum of block priors
- Block log posterior methods as likelihood plus the specified block prior only
- `joint_logpost` as `loglik + logprior_all`
"""

from __future__ import annotations

import numpy as np

import bonus2_conftest as bc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

from bonus2 import bonus2_posterior as bp

ATOL = 1e-12
RTOL = 0.0


def _make_posterior(
    panel_np: dict[str, object],
    config_overrides: dict[str, float] | None = None,
) -> tuple[bp.Bonus2PosteriorTF, bp.Bonus2PosteriorConfig, dict[str, object]]:
    """Build a Bonus2PosteriorTF and return it with its config and inputs."""
    config = bc.posterior_config(overrides=config_overrides)
    inputs = bc.posterior_inputs_tf(panel_np)
    posterior = bp.Bonus2PosteriorTF(config=config, inputs=inputs)
    return posterior, config, inputs


def _z_zero_tf(dims: dict[str, int]) -> dict[str, tf.Tensor]:
    """Zero-filled z dict with correct block shapes and float64 dtype."""
    return bc.z_blocks_tf(dims=dims, fill=0.0)


def _closed_form_logprior(z_block: tf.Tensor, sigma: float) -> float:
    """Closed-form Gaussian quadratic prior used by the posterior block priors."""
    z_np = tf.convert_to_tensor(z_block).numpy()
    return float(-0.5 * np.sum((z_np / float(sigma)) ** 2))


def test_logprior_beta_habit_matches_formula() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())
    posterior, config, _inputs = _make_posterior(panel)

    z_beta_habit_j = tf.convert_to_tensor(
        np.asarray([0.5, -1.0, 2.0], dtype=np.float64),
        dtype=tf.float64,
    )

    out = posterior.logprior_beta_habit(
        z_beta_habit_j=z_beta_habit_j,
    ).numpy()

    expected = _closed_form_logprior(
        z_block=z_beta_habit_j,
        sigma=config.sigma_z_beta_habit_j,
    )
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_logprior_beta_weekend_matches_formula() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())
    posterior, config, _inputs = _make_posterior(panel)

    z_beta_weekend_jw = tf.convert_to_tensor(
        np.asarray(
            [
                [0.25, -0.50],
                [1.00, 0.75],
                [-1.50, 0.10],
            ],
            dtype=np.float64,
        ),
        dtype=tf.float64,
    )

    out = posterior.logprior_beta_weekend(
        z_beta_weekend_jw=z_beta_weekend_jw,
    ).numpy()

    expected = _closed_form_logprior(
        z_block=z_beta_weekend_jw,
        sigma=config.sigma_z_beta_weekend_jw,
    )
    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_logprior_all_equals_sum_of_block_priors() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())
    posterior, _config, _inputs = _make_posterior(panel)

    z = _z_zero_tf(dims)
    z["z_beta_intercept_j"] = tf.convert_to_tensor(
        np.asarray([0.10, -0.20, 0.30], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_habit_j"] = tf.convert_to_tensor(
        np.asarray([0.40, 0.00, -0.50], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_peer_j"] = tf.convert_to_tensor(
        np.asarray([-0.10, 0.20, 0.30], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_weekend_jw"] = tf.convert_to_tensor(
        np.asarray(
            [
                [0.20, -0.10],
                [0.00, 0.30],
                [-0.40, 0.50],
            ],
            dtype=np.float64,
        ),
        dtype=tf.float64,
    )
    z["z_a_m"] = tf.convert_to_tensor(
        np.asarray([[-0.20], [0.15]], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_b_m"] = tf.convert_to_tensor(
        np.asarray([[0.05], [-0.25]], dtype=np.float64),
        dtype=tf.float64,
    )

    out = posterior.logprior_all(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    ).numpy()

    expected = (
        posterior.logprior_beta_intercept(
            z_beta_intercept_j=z["z_beta_intercept_j"],
        )
        + posterior.logprior_beta_habit(
            z_beta_habit_j=z["z_beta_habit_j"],
        )
        + posterior.logprior_beta_peer(
            z_beta_peer_j=z["z_beta_peer_j"],
        )
        + posterior.logprior_beta_weekend(
            z_beta_weekend_jw=z["z_beta_weekend_jw"],
        )
        + posterior.logprior_a(
            z_a_m=z["z_a_m"],
        )
        + posterior.logprior_b(
            z_b_m=z["z_b_m"],
        )
    ).numpy()

    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_loglik_mnt_runs_end_to_end_and_is_finite_and_loglik_is_sum() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )
    posterior, _config, _inputs = _make_posterior(panel)
    z = _z_zero_tf(dims)

    ll_mnt = posterior.loglik_mnt(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    )
    assert tuple(ll_mnt.shape) == (dims["M"], dims["N"], dims["T"])
    ll_mnt_np = ll_mnt.numpy()
    assert np.isfinite(ll_mnt_np).all()

    ll = posterior.loglik(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    )
    assert tuple(ll.shape) == ()
    ll_np = ll.numpy()
    assert np.isfinite(ll_np)

    np.testing.assert_allclose(ll_np, ll_mnt_np.sum(), rtol=RTOL, atol=1e-10)


def test_loglik_runs_with_K0_no_seasonality() -> None:
    dims = bc.tiny_dims_k0()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="mixed",
        neighbor_pattern="ring",
        weekend_pattern="0101",
    )
    posterior, _config, _inputs = _make_posterior(panel)
    z = _z_zero_tf(dims)

    ll_mnt = posterior.loglik_mnt(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    )
    assert tuple(ll_mnt.shape) == (dims["M"], dims["N"], dims["T"])
    assert np.isfinite(ll_mnt.numpy()).all()

    ll = posterior.loglik(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    )
    assert np.isfinite(ll.numpy())


def test_beta_habit_block_logpost_equals_loglik_plus_block_prior() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )
    posterior, _config, _inputs = _make_posterior(panel)
    z = _z_zero_tf(dims)
    z["z_beta_habit_j"] = tf.convert_to_tensor(
        np.asarray([0.5, -1.0, 2.0], dtype=np.float64),
        dtype=tf.float64,
    )

    out = posterior.beta_habit_block_logpost(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    ).numpy()

    expected = (
        posterior.loglik(
            z_beta_intercept_j=z["z_beta_intercept_j"],
            z_beta_habit_j=z["z_beta_habit_j"],
            z_beta_peer_j=z["z_beta_peer_j"],
            z_beta_weekend_jw=z["z_beta_weekend_jw"],
            z_a_m=z["z_a_m"],
            z_b_m=z["z_b_m"],
        )
        + posterior.logprior_beta_habit(
            z_beta_habit_j=z["z_beta_habit_j"],
        )
    ).numpy()

    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_a_block_logpost_equals_loglik_plus_block_prior() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="mixed",
        neighbor_pattern="ring",
        weekend_pattern="0101",
    )
    posterior, _config, _inputs = _make_posterior(panel)
    z = _z_zero_tf(dims)
    z["z_a_m"] = tf.convert_to_tensor(
        np.asarray([[-0.25], [0.75]], dtype=np.float64),
        dtype=tf.float64,
    )

    out = posterior.a_block_logpost(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    ).numpy()

    expected = (
        posterior.loglik(
            z_beta_intercept_j=z["z_beta_intercept_j"],
            z_beta_habit_j=z["z_beta_habit_j"],
            z_beta_peer_j=z["z_beta_peer_j"],
            z_beta_weekend_jw=z["z_beta_weekend_jw"],
            z_a_m=z["z_a_m"],
            z_b_m=z["z_b_m"],
        )
        + posterior.logprior_a(
            z_a_m=z["z_a_m"],
        )
    ).numpy()

    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)


def test_joint_logpost_equals_loglik_plus_logprior_all() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )
    posterior, _config, _inputs = _make_posterior(panel)
    z = _z_zero_tf(dims)
    z["z_beta_intercept_j"] = tf.convert_to_tensor(
        np.asarray([0.10, -0.15, 0.05], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_habit_j"] = tf.convert_to_tensor(
        np.asarray([0.20, 0.00, -0.30], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_peer_j"] = tf.convert_to_tensor(
        np.asarray([-0.10, 0.25, 0.15], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_beta_weekend_jw"] = tf.convert_to_tensor(
        np.asarray(
            [
                [0.10, -0.05],
                [0.00, 0.20],
                [-0.30, 0.40],
            ],
            dtype=np.float64,
        ),
        dtype=tf.float64,
    )
    z["z_a_m"] = tf.convert_to_tensor(
        np.asarray([[-0.20], [0.10]], dtype=np.float64),
        dtype=tf.float64,
    )
    z["z_b_m"] = tf.convert_to_tensor(
        np.asarray([[0.05], [-0.15]], dtype=np.float64),
        dtype=tf.float64,
    )

    out = posterior.joint_logpost(
        z_beta_intercept_j=z["z_beta_intercept_j"],
        z_beta_habit_j=z["z_beta_habit_j"],
        z_beta_peer_j=z["z_beta_peer_j"],
        z_beta_weekend_jw=z["z_beta_weekend_jw"],
        z_a_m=z["z_a_m"],
        z_b_m=z["z_b_m"],
    ).numpy()

    expected = (
        posterior.loglik(
            z_beta_intercept_j=z["z_beta_intercept_j"],
            z_beta_habit_j=z["z_beta_habit_j"],
            z_beta_peer_j=z["z_beta_peer_j"],
            z_beta_weekend_jw=z["z_beta_weekend_jw"],
            z_a_m=z["z_a_m"],
            z_b_m=z["z_b_m"],
        )
        + posterior.logprior_all(
            z_beta_intercept_j=z["z_beta_intercept_j"],
            z_beta_habit_j=z["z_beta_habit_j"],
            z_beta_peer_j=z["z_beta_peer_j"],
            z_beta_weekend_jw=z["z_beta_weekend_jw"],
            z_a_m=z["z_a_m"],
            z_b_m=z["z_b_m"],
        )
    ).numpy()

    np.testing.assert_allclose(out, expected, rtol=RTOL, atol=ATOL)
