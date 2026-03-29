# tests/bonus2/test_bonus2_model.py
"""
Unit tests for `bonus2.bonus2_model`.

These tests validate Bonus2 core mechanics:
- sparse adjacency construction from neighbor lists,
- deterministic state construction (inside-choice onehot, habit stock, peer exposure),
- market Fourier seasonality,
- MNL probabilities and log-likelihood invariants.

No pytest fixtures are used. Tests build tiny deterministic environments via
`bonus2_conftest` helper functions (called directly like normal Python functions).
"""

from __future__ import annotations

import math

import numpy as np

import bonus2_conftest as bc  # sets TF_CPP_MIN_LOG_LEVEL before importing TF
import tensorflow as tf

from bonus2 import bonus2_model as bm


def _theta_zeros(m: int, j: int, k: int) -> dict[str, tf.Tensor]:
    """All-zero theta dict with correct keys/shapes."""
    return {
        "beta_intercept_j": tf.zeros((j,), tf.float64),
        "beta_habit_j": tf.zeros((j,), tf.float64),
        "beta_peer_j": tf.zeros((j,), tf.float64),
        "beta_weekend_jw": tf.zeros((j, 2), tf.float64),
        "a_m": tf.zeros((m, k), tf.float64),
        "b_m": tf.zeros((m, k), tf.float64),
    }


def test_neighbors_to_sparse_adj_builds_correct_edges_and_shape() -> None:
    n = 4
    neighbors_by_consumer = [[1, 3], [], [0], [2]]

    adj = bm.neighbors_to_sparse_adj(
        neighbors_by_consumer=neighbors_by_consumer,
        n_consumers=n,
    )
    assert tuple(adj.dense_shape.numpy().tolist()) == (n, n)

    dense = tf.sparse.to_dense(adj).numpy()
    expected = np.zeros((n, n), dtype=np.float64)
    expected[0, 1] = 1.0
    expected[0, 3] = 1.0
    expected[2, 0] = 1.0
    expected[3, 2] = 1.0
    np.testing.assert_allclose(dense, expected, rtol=0, atol=0)


def test_build_peer_adjacency_returns_one_sparse_adj_per_market() -> None:
    n = 4
    neighbors_m = [
        [[1], [2], [3], []],
        [[2, 3], [], [0], [1]],
    ]

    peer_adj_m = bm.build_peer_adjacency(neighbors_m=neighbors_m, n_consumers=n)
    assert isinstance(peer_adj_m, tuple)
    assert len(peer_adj_m) == 2

    dense0 = tf.sparse.to_dense(peer_adj_m[0]).numpy()
    dense1 = tf.sparse.to_dense(peer_adj_m[1]).numpy()

    expected0 = np.zeros((n, n), dtype=np.float64)
    expected0[0, 1] = 1.0
    expected0[1, 2] = 1.0
    expected0[2, 3] = 1.0

    expected1 = np.zeros((n, n), dtype=np.float64)
    expected1[0, 2] = 1.0
    expected1[0, 3] = 1.0
    expected1[2, 0] = 1.0
    expected1[3, 1] = 1.0

    np.testing.assert_allclose(dense0, expected0, rtol=0, atol=0)
    np.testing.assert_allclose(dense1, expected1, rtol=0, atol=0)


def test_inside_choice_onehot_excludes_outside_option() -> None:
    m, n, t, j = 1, 2, 4, 3
    y = tf.constant(
        [
            [
                [0, 1, 0, 3],
                [2, 0, 1, 0],
            ]
        ],
        dtype=tf.int32,
    )

    x = bm._inside_choice_onehot(y_mit=y, n_products=tf.constant(j, tf.int32))
    assert tuple(x.shape) == (m, n, t, j)

    x_np = x.numpy()

    for ti in range(t):
        if int(y[0, 0, ti].numpy()) == 0:
            np.testing.assert_allclose(x_np[0, 0, ti, :], 0.0, rtol=0, atol=0)
        if int(y[0, 1, ti].numpy()) == 0:
            np.testing.assert_allclose(x_np[0, 1, ti, :], 0.0, rtol=0, atol=0)

    np.testing.assert_allclose(x_np[0, 0, 1, :], np.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(x_np[0, 0, 3, :], np.asarray([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(x_np[0, 1, 0, :], np.asarray([0.0, 1.0, 0.0]))

    sums = x_np.sum(axis=3)
    expected_sums = (y.numpy() > 0).astype(np.float64)
    np.testing.assert_allclose(sums, expected_sums, rtol=0, atol=0)


def test_habit_stock_pre_choice_matches_recurrence_and_timing() -> None:
    panel = bc.toy_habit_case()
    ptf = bc.model_state_inputs_tf(panel)

    j = int(ptf["delta_mj"].shape[1])
    x = bm._inside_choice_onehot(ptf["y_mit"], tf.constant(j, tf.int32))

    h = bm.habit_stock_pre_choice(x_mntj=x, decay=ptf["decay"])
    assert tuple(h.shape) == (1, 1, 5, 1)

    expected = bc.expected_habit_stock_pre_choice_np(
        y_mit=np.asarray(panel["y_mit"], dtype=np.int32),
        J=j,
        decay=float(panel["decay"]),
    )

    np.testing.assert_allclose(h.numpy(), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h.numpy()[0, 0, 0, 0], 0.0, rtol=0, atol=0)


def test_rolling_lookback_sum_excludes_current_and_respects_L() -> None:
    t = 6
    x_vals = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)
    x = tf.convert_to_tensor(x_vals.reshape(1, 1, t, 1), tf.float64)
    L = tf.constant(2, tf.int32)

    c = bm.rolling_lookback_sum(x_mntj=x, lookback=L).numpy().reshape(t)

    expected = np.zeros((t,), dtype=np.float64)
    expected[0] = 0.0
    expected[1] = x_vals[0]
    expected[2] = x_vals[1] + x_vals[0]
    expected[3] = x_vals[2] + x_vals[1]
    expected[4] = x_vals[3] + x_vals[2]
    expected[5] = x_vals[4] + x_vals[3]

    np.testing.assert_allclose(c, expected, rtol=0, atol=1e-12)


def test_peer_exposure_from_onehot_matches_manual_neighbor_counts() -> None:
    panel = bc.toy_peer_case()
    ptf = bc.model_state_inputs_tf(panel)

    j = int(ptf["delta_mj"].shape[1])
    x = bm._inside_choice_onehot(ptf["y_mit"], tf.constant(j, tf.int32))

    p = bm.peer_exposure_from_onehot(
        x_mntj=x,
        peer_adj_m=ptf["peer_adj_m"],
        lookback=ptf["lookback"],
    )

    expected = bc.expected_peer_exposure_np(
        y_mit=np.asarray(panel["y_mit"], dtype=np.int32),
        neighbors_m=panel["neighbors_m"],
        J=j,
        lookback=int(panel["lookback"]),
    )

    assert tuple(p.shape) == expected.shape
    np.testing.assert_allclose(p.numpy(), expected, rtol=0, atol=1e-12)


def test_peer_exposure_handles_empty_network() -> None:
    dims = {"M": 1, "N": 3, "J": 2, "T": 4, "K": 0}
    hyper = {"lookback": 2, "decay": 0.9, "season_period": 365, "eps": 1e-12}

    panel = bc.panel_np(
        dims=dims,
        hyper=hyper,
        y_pattern="alternating",
        neighbor_pattern="empty",
        weekend_pattern="0101",
    )
    ptf = bc.model_state_inputs_tf(panel)

    j = int(ptf["delta_mj"].shape[1])
    x = bm._inside_choice_onehot(ptf["y_mit"], tf.constant(j, tf.int32))

    p = bm.peer_exposure_from_onehot(
        x_mntj=x,
        peer_adj_m=ptf["peer_adj_m"],
        lookback=ptf["lookback"],
    )

    np.testing.assert_allclose(p.numpy(), 0.0, rtol=0, atol=0)


def test_build_deterministic_states_is_consistent_with_components() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )

    ptf = bc.model_state_inputs_tf(panel)
    j = int(ptf["delta_mj"].shape[1])

    x1 = bm._inside_choice_onehot(ptf["y_mit"], tf.constant(j, tf.int32))
    h1 = bm.habit_stock_pre_choice(x_mntj=x1, decay=ptf["decay"])
    p1 = bm.peer_exposure_from_onehot(
        x_mntj=x1,
        peer_adj_m=ptf["peer_adj_m"],
        lookback=ptf["lookback"],
    )

    h2, p2 = bm.build_deterministic_states(
        y_mit=ptf["y_mit"],
        n_products=tf.constant(j, tf.int32),
        peer_adj_m=ptf["peer_adj_m"],
        lookback=ptf["lookback"],
        decay=ptf["decay"],
    )

    np.testing.assert_allclose(h2.numpy(), h1.numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(p2.numpy(), p1.numpy(), rtol=0, atol=1e-12)


def test_market_fourier_seasonality_matches_reference() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())

    m, t = dims["M"], dims["T"]
    a_mk = tf.convert_to_tensor(np.full((m, dims["K"]), -0.02, np.float64), tf.float64)
    b_mk = tf.convert_to_tensor(np.full((m, dims["K"]), 0.03, np.float64), tf.float64)

    sin = tf.convert_to_tensor(
        np.asarray(panel["season_sin_kt"], np.float64),
        tf.float64,
    )
    cos = tf.convert_to_tensor(
        np.asarray(panel["season_cos_kt"], np.float64),
        tf.float64,
    )

    s = bm.market_fourier_seasonality(
        a_mk=a_mk,
        b_mk=b_mk,
        season_sin_kt=sin,
        season_cos_kt=cos,
    )
    assert tuple(s.shape) == (m, t)

    expected = bc.expected_market_seasonality_np(
        a_mk=a_mk.numpy(),
        b_mk=b_mk.numpy(),
        season_sin_kt=sin.numpy(),
        season_cos_kt=cos.numpy(),
    )
    np.testing.assert_allclose(s.numpy(), expected, rtol=0, atol=1e-12)


def test_market_fourier_seasonality_K0_returns_zeros() -> None:
    dims = bc.tiny_dims_k0()
    panel = bc.panel_np(dims=dims, hyper=bc.tiny_hyperparams())

    m, t = dims["M"], dims["T"]

    a_mk = tf.zeros((m, 0), tf.float64)
    b_mk = tf.zeros((m, 0), tf.float64)

    sin = tf.convert_to_tensor(
        np.asarray(panel["season_sin_kt"], np.float64),
        tf.float64,
    )
    cos = tf.convert_to_tensor(
        np.asarray(panel["season_cos_kt"], np.float64),
        tf.float64,
    )

    s = bm.market_fourier_seasonality(
        a_mk=a_mk,
        b_mk=b_mk,
        season_sin_kt=sin,
        season_cos_kt=cos,
    )
    assert tuple(s.shape) == (m, t)
    np.testing.assert_allclose(s.numpy(), 0.0, rtol=0, atol=0)


def test_predict_choice_probs_shape_and_simplex() -> None:
    dims = bc.tiny_dims()
    panel = bc.panel_np(
        dims=dims,
        hyper=bc.tiny_hyperparams(),
        neighbor_pattern="ring",
    )
    ptf = bc.posterior_inputs_tf(panel)

    m, n, t = (int(x) for x in ptf["y_mit"].shape)
    j = int(ptf["delta_mj"].shape[1])
    k = int(ptf["season_sin_kt"].shape[0])

    theta = _theta_zeros(m=m, j=j, k=k)

    p = bm.predict_choice_probs_from_theta(
        theta=theta,
        delta_mj=ptf["delta_mj"],
        is_weekend_t=ptf["is_weekend_t"],
        season_sin_kt=ptf["season_sin_kt"],
        season_cos_kt=ptf["season_cos_kt"],
        h_mntj=ptf["h_mntj"],
        p_mntj=ptf["p_mntj"],
    )

    assert tuple(p.shape) == (m, n, t, j + 1)

    p_np = p.numpy()
    assert np.isfinite(p_np).all()
    assert np.all((p_np >= 0.0) & (p_np <= 1.0))

    sums = p_np.sum(axis=3)
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-12)


def test_uniform_logits_gives_uniform_probs_and_known_loglik() -> None:
    dims = {"M": 2, "N": 3, "J": 4, "T": 5, "K": 1}

    panel = bc.panel_np(
        dims=dims,
        hyper={"lookback": 2, "decay": 0.85, "season_period": 365, "eps": 1e-12},
        y_pattern="alternating",
        neighbor_pattern="asymmetric",
        weekend_pattern="0101",
    )

    panel["delta_mj"] = np.zeros((dims["M"], dims["J"]), dtype=np.float64)
    ptf = bc.posterior_inputs_tf(panel)

    m, n, t = (int(x) for x in ptf["y_mit"].shape)
    j = int(ptf["delta_mj"].shape[1])
    k = int(ptf["season_sin_kt"].shape[0])

    theta = _theta_zeros(m=m, j=j, k=k)

    p = bm.predict_choice_probs_from_theta(
        theta=theta,
        delta_mj=ptf["delta_mj"],
        is_weekend_t=ptf["is_weekend_t"],
        season_sin_kt=ptf["season_sin_kt"],
        season_cos_kt=ptf["season_cos_kt"],
        h_mntj=ptf["h_mntj"],
        p_mntj=ptf["p_mntj"],
    )

    uniform = 1.0 / float(j + 1)
    np.testing.assert_allclose(p.numpy(), uniform, rtol=0, atol=1e-12)

    ll_mnt = bm.loglik_mnt_from_theta(
        theta=theta,
        y_mit=ptf["y_mit"],
        delta_mj=ptf["delta_mj"],
        is_weekend_t=ptf["is_weekend_t"],
        season_sin_kt=ptf["season_sin_kt"],
        season_cos_kt=ptf["season_cos_kt"],
        h_mntj=ptf["h_mntj"],
        p_mntj=ptf["p_mntj"],
    )

    expected_ll = -math.log(float(j + 1))
    np.testing.assert_allclose(ll_mnt.numpy(), expected_ll, rtol=0, atol=1e-12)
