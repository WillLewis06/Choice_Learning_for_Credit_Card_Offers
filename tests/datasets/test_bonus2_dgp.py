# tests/bonus2/test_bonus2_dgp.py
"""
Unit tests for `datasets.bonus2_dgp`.

These tests focus on:
- exact algebraic identities (time features, Fourier helper),
- shape/dtype/range invariants,
- determinism under fixed RNG seeds,
- rolling peer-window bookkeeping,
- end-to-end DGP schema and validation behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.bonus2_dgp import (
    _update_recent_choice_counts_inplace,
    advance_peer_window,
    fourier_seasonality,
    generate_sparse_network,
    generate_time_features,
    make_rng,
    peer_exposure_from_recent_counts,
    sample_core_product_params,
    sample_mnl,
    sample_product_intercepts,
    sample_time_market_params,
    sample_time_product_params,
    simulate_bonus2_dgp,
    simulate_one_market,
)


def _params_true_default() -> dict[str, float]:
    return {
        "habit_mean": 0.2,
        "habit_sd": 0.1,
        "peer_mean": 0.1,
        "peer_sd": 0.05,
        "mktprod_sd": 0.3,
        "weekend_prod_sd": 0.2,
        "season_mkt_sd": 0.15,
    }


# -----------------------------------------------------------------------------
# Time features
# -----------------------------------------------------------------------------


def test_generate_time_features_weekend_matches_dow_rule() -> None:
    T = 20
    P = 365
    K = 2

    weekend_t, season_angle_t, sin_kt, cos_kt, dow_t = generate_time_features(T, P, K)

    assert weekend_t.shape == (T,)
    assert season_angle_t.shape == (T,)
    assert sin_kt.shape == (K, T)
    assert cos_kt.shape == (K, T)
    assert dow_t.shape == (T,)

    assert weekend_t.dtype == np.int64
    assert dow_t.dtype == np.int64
    assert season_angle_t.dtype == np.float64
    assert sin_kt.dtype == np.float64
    assert cos_kt.dtype == np.float64

    t = np.arange(T, dtype=np.int64)
    np.testing.assert_array_equal(dow_t, (t % 7).astype(np.int64))
    np.testing.assert_array_equal(weekend_t, (dow_t >= 5).astype(np.int64))
    assert set(np.unique(weekend_t)).issubset({0, 1})

    tau = (t % P).astype(np.float64)
    expected_angle = (2.0 * np.pi / float(P) * tau).astype(np.float64)
    np.testing.assert_allclose(season_angle_t, expected_angle, rtol=0, atol=1e-15)


def test_generate_time_features_K0_returns_empty_basis() -> None:
    T = 17
    P = 365
    K = 0

    weekend_t, season_angle_t, sin_kt, cos_kt, dow_t = generate_time_features(T, P, K)

    assert weekend_t.shape == (T,)
    assert season_angle_t.shape == (T,)
    assert dow_t.shape == (T,)

    assert sin_kt.shape == (0, T)
    assert cos_kt.shape == (0, T)

    assert sin_kt.dtype == np.float64
    assert cos_kt.dtype == np.float64


def test_generate_time_features_basis_matches_trig_definition() -> None:
    T = 12
    P = 30
    K = 3

    _, season_angle_t, sin_kt, cos_kt, _ = generate_time_features(T, P, K)

    for k in range(1, K + 1):
        np.testing.assert_allclose(
            sin_kt[k - 1], np.sin(k * season_angle_t), rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            cos_kt[k - 1], np.cos(k * season_angle_t), rtol=0, atol=1e-15
        )


# -----------------------------------------------------------------------------
# Fourier seasonality helper
# -----------------------------------------------------------------------------


def test_fourier_seasonality_matches_matrix_formula() -> None:
    R = 4
    K = 2
    T = 9
    rng = np.random.default_rng(0)

    a = rng.normal(size=(R, K)).astype(np.float64)
    b = rng.normal(size=(R, K)).astype(np.float64)
    sin = rng.normal(size=(K, T)).astype(np.float64)
    cos = rng.normal(size=(K, T)).astype(np.float64)

    out = fourier_seasonality(a, b, sin, cos)
    expected = (a @ sin + b @ cos).astype(np.float64)

    assert out.shape == (R, T)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_fourier_seasonality_K0_returns_zeros() -> None:
    R = 3
    K = 0
    T = 8

    a = np.zeros((R, K), dtype=np.float64)
    b = np.zeros((R, K), dtype=np.float64)
    sin = np.zeros((K, T), dtype=np.float64)
    cos = np.zeros((K, T), dtype=np.float64)

    out = fourier_seasonality(a, b, sin, cos)
    assert out.shape == (R, T)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.zeros((R, T), dtype=np.float64))


# -----------------------------------------------------------------------------
# Parameter draws
# -----------------------------------------------------------------------------


def test_sample_core_product_params_shapes_and_seed_reproducible() -> None:
    J = 7
    params = _params_true_default()

    rng1 = make_rng(123)
    rng2 = make_rng(123)

    h1, p1 = sample_core_product_params(J=J, rng=rng1, dgp_hyperparams=params)
    h2, p2 = sample_core_product_params(J=J, rng=rng2, dgp_hyperparams=params)

    assert h1.shape == (J,)
    assert p1.shape == (J,)
    assert h1.dtype == np.float64
    assert p1.dtype == np.float64

    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(p1, p2)


def test_sample_product_intercepts_shapes_and_dtype() -> None:
    J = 5
    params = _params_true_default()
    rng = make_rng(0)

    b = sample_product_intercepts(J=J, rng=rng, dgp_hyperparams=params)
    assert b.shape == (J,)
    assert b.dtype == np.float64
    assert np.isfinite(b).all()


def test_sample_time_product_params_shapes_and_dtype() -> None:
    J = 6
    params = _params_true_default()
    rng = make_rng(0)

    bw = sample_time_product_params(J=J, rng=rng, dgp_hyperparams=params)
    assert bw.shape == (J, 2)
    assert bw.dtype == np.float64
    assert np.isfinite(bw).all()


def test_sample_time_market_params_shapes_K0_and_Kpos() -> None:
    M = 3
    params = _params_true_default()

    rng0 = make_rng(0)
    a0, b0 = sample_time_market_params(M=M, K=0, rng=rng0, dgp_hyperparams=params)
    assert a0.shape == (M, 0)
    assert b0.shape == (M, 0)
    assert a0.dtype == np.float64
    assert b0.dtype == np.float64

    rng1 = make_rng(1)
    a1, b1 = sample_time_market_params(M=M, K=2, rng=rng1, dgp_hyperparams=params)
    assert a1.shape == (M, 2)
    assert b1.shape == (M, 2)
    assert a1.dtype == np.float64
    assert b1.dtype == np.float64
    assert np.isfinite(a1).all()
    assert np.isfinite(b1).all()


# -----------------------------------------------------------------------------
# Network and peer window mechanics
# -----------------------------------------------------------------------------


def test_generate_sparse_network_contract_and_constraints() -> None:
    N = 10
    avg_friends = 3.0
    friends_sd = 0.5

    rng1 = make_rng(123)
    rng2 = make_rng(123)

    net1 = generate_sparse_network(N, avg_friends, friends_sd, rng1)
    net2 = generate_sparse_network(N, avg_friends, friends_sd, rng2)

    assert isinstance(net1, list)
    assert len(net1) == N

    for i in range(N):
        ni = net1[i]
        assert isinstance(ni, np.ndarray)
        assert ni.dtype == np.int64

        assert ((ni >= 0) & (ni < N)).all()
        assert not np.any(ni == i)

        if ni.size:
            assert np.unique(ni).size == ni.size

        np.testing.assert_array_equal(net1[i], net2[i])


def test_generate_sparse_network_all_empty_when_avg_friends_zero() -> None:
    N = 7
    rng = make_rng(0)
    net = generate_sparse_network(N, avg_friends=0.0, friends_sd=0.0, rng=rng)

    assert len(net) == N
    for i in range(N):
        assert net[i].dtype == np.int64
        assert net[i].shape == (0,)


def test_update_recent_choice_counts_ignores_outside_and_uses_y_minus_1() -> None:
    N = 5
    J = 4
    counts = np.zeros((N, J), dtype=np.int32)

    y = np.asarray([0, 1, 0, 4, 2], dtype=np.int64)
    _update_recent_choice_counts_inplace(counts, y, sign=+1)

    expected = np.zeros((N, J), dtype=np.int32)
    expected[1, 0] += 1
    expected[3, 3] += 1
    expected[4, 1] += 1

    np.testing.assert_array_equal(counts, expected)

    _update_recent_choice_counts_inplace(counts, y, sign=-1)
    np.testing.assert_array_equal(counts, np.zeros((N, J), dtype=np.int32))


def test_advance_peer_window_updates_buffer_and_counts_consistently() -> None:
    L = 3
    N = 4
    J = 3

    peer_buf = np.zeros((L, N), dtype=np.int64)
    counts = np.zeros((N, J), dtype=np.int32)
    buf_pos = 0

    y0 = np.asarray([0, 1, 2, 0], dtype=np.int64)
    y1 = np.asarray([3, 0, 2, 1], dtype=np.int64)
    y2 = np.asarray([0, 0, 0, 3], dtype=np.int64)
    y3 = np.asarray([1, 2, 3, 0], dtype=np.int64)

    buf_pos = advance_peer_window(peer_buf, buf_pos, counts, y0)
    exp = np.zeros((N, J), dtype=np.int32)
    exp[1, 0] += 1
    exp[2, 1] += 1
    np.testing.assert_array_equal(counts, exp)

    buf_pos = advance_peer_window(peer_buf, buf_pos, counts, y1)
    exp[0, 2] += 1
    exp[2, 1] += 1
    exp[3, 0] += 1
    np.testing.assert_array_equal(counts, exp)

    buf_pos = advance_peer_window(peer_buf, buf_pos, counts, y2)
    exp[3, 2] += 1
    np.testing.assert_array_equal(counts, exp)

    buf_pos = advance_peer_window(peer_buf, buf_pos, counts, y3)
    exp[1, 0] -= 1
    exp[2, 1] -= 1
    exp[0, 0] += 1
    exp[1, 1] += 1
    exp[2, 2] += 1
    np.testing.assert_array_equal(counts, exp)
    assert buf_pos == 1


def test_peer_exposure_from_recent_counts_sums_neighbors_rows() -> None:
    N = 4
    J = 3

    C = np.zeros((N, J), dtype=np.int32)
    C[0] = [1, 2, 3]
    C[1] = [0, 1, 0]
    C[2] = [5, 0, 0]
    C[3] = [0, 0, 4]

    neighbors = [
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([0, 3], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
    ]

    P = peer_exposure_from_recent_counts(C, neighbors)
    assert P.shape == (N, J)
    assert P.dtype == np.float64

    np.testing.assert_allclose(P[0], C[1] + C[2], rtol=0, atol=0)
    np.testing.assert_allclose(P[1], np.zeros(J), rtol=0, atol=0)
    np.testing.assert_allclose(P[2], C[0] + C[3], rtol=0, atol=0)
    np.testing.assert_allclose(P[3], C[2], rtol=0, atol=0)


# -----------------------------------------------------------------------------
# MNL sampler
# -----------------------------------------------------------------------------


def test_sample_mnl_output_range_and_dtype_and_shape() -> None:
    rng = make_rng(0)

    v0 = np.zeros((5, 0), dtype=np.float64)
    y0 = sample_mnl(v0, rng)
    assert y0.shape == (5,)
    assert y0.dtype == np.int64
    np.testing.assert_array_equal(y0, np.zeros(5, dtype=np.int64))

    v = np.zeros((7, 3), dtype=np.float64)
    y = sample_mnl(v, make_rng(1))
    assert y.shape == (7,)
    assert y.dtype == np.int64
    assert ((y >= 0) & (y <= 3)).all()


def test_sample_mnl_numerical_stability_large_utilities() -> None:
    rng = make_rng(123)
    N = 50
    J = 4

    v = np.full((N, J), 1000.0, dtype=np.float64)
    y = sample_mnl(v, rng)

    assert y.shape == (N,)
    assert y.dtype == np.int64
    assert ((y >= 0) & (y <= J)).all()
    assert np.isfinite(y.astype(np.float64)).all()


# -----------------------------------------------------------------------------
# One-market simulator
# -----------------------------------------------------------------------------


def test_simulate_one_market_deterministic_given_seed_and_inputs() -> None:
    N = 15
    T = 25
    J = 3
    K = 2
    P = 365
    lookback = 3
    decay = 0.9

    weekend_t, _, sin_kt, cos_kt, _ = generate_time_features(T, P, K)

    delta_j = np.asarray([0.1, -0.2, 0.05], dtype=np.float64)
    beta_intercept_j = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    beta_weekend_jw = np.zeros((J, 2), dtype=np.float64)
    beta_habit_j = np.asarray([0.2, 0.1, 0.05], dtype=np.float64)
    beta_peer_j = np.asarray([0.05, 0.03, 0.02], dtype=np.float64)

    a_m = np.asarray([[0.1, -0.05]], dtype=np.float64)
    b_m = np.asarray([[0.02, 0.03]], dtype=np.float64)
    S_m = fourier_seasonality(a_m, b_m, sin_kt, cos_kt)[0]

    neighbors = generate_sparse_network(
        N, avg_friends=2.0, friends_sd=0.0, rng=make_rng(999)
    )

    y1 = simulate_one_market(
        delta_mj=delta_j,
        beta_intercept_j=beta_intercept_j,
        beta_weekend_jw=beta_weekend_jw,
        weekend_t=weekend_t,
        S_m=S_m,
        beta_habit_j=beta_habit_j,
        beta_peer_j=beta_peer_j,
        decay=decay,
        neighbors=neighbors,
        rng=make_rng(123),
        lookback=lookback,
    )

    y2 = simulate_one_market(
        delta_mj=delta_j,
        beta_intercept_j=beta_intercept_j,
        beta_weekend_jw=beta_weekend_jw,
        weekend_t=weekend_t,
        S_m=S_m,
        beta_habit_j=beta_habit_j,
        beta_peer_j=beta_peer_j,
        decay=decay,
        neighbors=neighbors,
        rng=make_rng(123),
        lookback=lookback,
    )

    assert y1.shape == (N, T)
    assert y1.dtype == np.int64
    assert ((y1 >= 0) & (y1 <= J)).all()
    np.testing.assert_array_equal(y1, y2)


def test_simulate_one_market_runs_with_empty_neighbors_and_lookback_one() -> None:
    N = 10
    T = 12
    J = 2
    K = 0
    P = 365
    lookback = 1
    decay = 0.8

    weekend_t, _, sin_kt, cos_kt, _ = generate_time_features(T, P, K)
    assert sin_kt.shape == (0, T)
    assert cos_kt.shape == (0, T)

    delta_j = np.asarray([0.1, 0.2], dtype=np.float64)
    beta_intercept_j = np.zeros((J,), dtype=np.float64)
    beta_weekend_jw = np.zeros((J, 2), dtype=np.float64)
    beta_habit_j = np.zeros((J,), dtype=np.float64)
    beta_peer_j = np.ones((J,), dtype=np.float64)
    S_m = np.zeros((T,), dtype=np.float64)

    neighbors = [np.zeros(0, dtype=np.int64) for _ in range(N)]

    y = simulate_one_market(
        delta_mj=delta_j,
        beta_intercept_j=beta_intercept_j,
        beta_weekend_jw=beta_weekend_jw,
        weekend_t=weekend_t,
        S_m=S_m,
        beta_habit_j=beta_habit_j,
        beta_peer_j=beta_peer_j,
        decay=decay,
        neighbors=neighbors,
        rng=make_rng(0),
        lookback=lookback,
    )

    assert y.shape == (N, T)
    assert y.dtype == np.int64
    assert ((y >= 0) & (y <= J)).all()


# -----------------------------------------------------------------------------
# End-to-end DGP simulator
# -----------------------------------------------------------------------------


def test_simulate_bonus2_dgp_outputs_panel_and_theta_true_with_correct_schema_and_is_deterministic() -> (
    None
):
    M = 2
    J = 3
    N = 25
    T = 30
    K = 2
    P = 365
    lookback = 3
    decay = 0.9
    seed = 123

    delta_mj = np.asarray([[0.1, -0.1, 0.0], [0.2, 0.0, -0.2]], dtype=np.float64)
    params_true = _params_true_default()

    out1 = simulate_bonus2_dgp(
        delta_mj=delta_mj,
        N=N,
        T=T,
        avg_friends=2.0,
        params_true=params_true,
        decay=decay,
        seed=seed,
        season_period=P,
        friends_sd=0.25,
        K=K,
        lookback=lookback,
    )
    out2 = simulate_bonus2_dgp(
        delta_mj=delta_mj,
        N=N,
        T=T,
        avg_friends=2.0,
        params_true=params_true,
        decay=decay,
        seed=seed,
        season_period=P,
        friends_sd=0.25,
        K=K,
        lookback=lookback,
    )

    assert set(out1.keys()) == {"panel", "theta_true"}
    panel = out1["panel"]
    theta = out1["theta_true"]

    assert set(panel.keys()) == {
        "y_mit",
        "delta_mj",
        "is_weekend_t",
        "neighbors_m",
        "lookback",
        "season_sin_kt",
        "season_cos_kt",
        "decay",
    }

    y_mit = np.asarray(panel["y_mit"])
    assert y_mit.shape == (M, N, T)
    assert y_mit.dtype == np.int64
    assert ((y_mit >= 0) & (y_mit <= J)).all()

    d_mj = np.asarray(panel["delta_mj"])
    assert d_mj.shape == (M, J)
    assert d_mj.dtype == np.float64
    np.testing.assert_allclose(d_mj, delta_mj, rtol=0, atol=0)

    is_weekend_t = np.asarray(panel["is_weekend_t"])
    assert is_weekend_t.shape == (T,)
    assert is_weekend_t.dtype == np.int64
    assert set(np.unique(is_weekend_t)).issubset({0, 1})

    sin_kt = np.asarray(panel["season_sin_kt"])
    cos_kt = np.asarray(panel["season_cos_kt"])
    assert sin_kt.shape == (K, T)
    assert cos_kt.shape == (K, T)
    assert sin_kt.dtype == np.float64
    assert cos_kt.dtype == np.float64

    assert int(panel["lookback"]) == lookback
    assert float(panel["decay"]) == decay

    neighbors_m = panel["neighbors_m"]
    assert isinstance(neighbors_m, list)
    assert len(neighbors_m) == M
    for m in range(M):
        assert len(neighbors_m[m]) == N
        for i in range(N):
            ni = neighbors_m[m][i]
            assert isinstance(ni, np.ndarray)
            assert ni.dtype == np.int64
            assert ((ni >= 0) & (ni < N)).all()
            assert not np.any(ni == i)
            if ni.size:
                assert np.unique(ni).size == ni.size

    assert set(theta.keys()) == {
        "beta_intercept_j",
        "beta_habit_j",
        "beta_peer_j",
        "beta_weekend_jw",
        "a_m",
        "b_m",
    }
    assert np.asarray(theta["beta_intercept_j"]).shape == (J,)
    assert np.asarray(theta["beta_habit_j"]).shape == (J,)
    assert np.asarray(theta["beta_peer_j"]).shape == (J,)
    assert np.asarray(theta["beta_weekend_jw"]).shape == (J, 2)
    assert np.asarray(theta["a_m"]).shape == (M, K)
    assert np.asarray(theta["b_m"]).shape == (M, K)

    np.testing.assert_array_equal(out1["panel"]["y_mit"], out2["panel"]["y_mit"])
    for k in theta.keys():
        np.testing.assert_array_equal(out1["theta_true"][k], out2["theta_true"][k])


def test_simulate_bonus2_dgp_handles_K0() -> None:
    M = 2
    J = 3
    N = 20
    T = 15
    K = 0
    P = 365
    lookback = 2

    delta_mj = np.zeros((M, J), dtype=np.float64)
    out = simulate_bonus2_dgp(
        delta_mj=delta_mj,
        N=N,
        T=T,
        avg_friends=1.0,
        params_true=_params_true_default(),
        decay=0.85,
        seed=0,
        season_period=P,
        friends_sd=0.0,
        K=K,
        lookback=lookback,
    )

    panel = out["panel"]
    theta = out["theta_true"]

    assert np.asarray(panel["season_sin_kt"]).shape == (0, T)
    assert np.asarray(panel["season_cos_kt"]).shape == (0, T)
    assert np.asarray(theta["a_m"]).shape == (M, 0)
    assert np.asarray(theta["b_m"]).shape == (M, 0)


def test_simulate_bonus2_dgp_rejects_missing_required_hyperparams() -> None:
    delta_mj = np.zeros((1, 2), dtype=np.float64)
    params = _params_true_default()
    params.pop("habit_sd")

    with pytest.raises(
        ValueError, match=r"params_true.*missing|required.*keys|missing keys"
    ):
        simulate_bonus2_dgp(
            delta_mj=delta_mj,
            N=5,
            T=5,
            avg_friends=1.0,
            params_true=params,
            decay=0.9,
            seed=0,
            season_period=365,
            friends_sd=0.0,
            K=0,
            lookback=1,
        )
