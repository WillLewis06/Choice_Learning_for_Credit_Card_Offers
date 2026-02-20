# tests/datasets/test_zhang_with_lu_dgp.py
from __future__ import annotations

import numpy as np
import pytest

import datasets.zhang_with_lu_dgp as dgp


def test_assign_groups_contiguous_blocks_and_range() -> None:
    # Case 1: not divisible
    J, G = 10, 3
    group_id = dgp._assign_groups(J, G)
    assert group_id.shape == (J,)
    assert group_id.dtype == np.int64
    assert group_id.min() == 0
    assert group_id.max() == G - 1

    # sizes differ by at most 1
    counts = np.bincount(group_id, minlength=G)
    assert counts.sum() == J
    assert counts.max() - counts.min() <= 1

    # contiguity: indices per group form one contiguous block
    for g in range(G):
        idx = np.where(group_id == g)[0]
        assert idx.size > 0
        assert idx[-1] - idx[0] + 1 == idx.size

    # Case 2: divisible
    J, G = 12, 3
    group_id = dgp._assign_groups(J, G)
    counts = np.bincount(group_id, minlength=G)
    assert np.all(counts == J // G)
    for g in range(G):
        idx = np.where(group_id == g)[0]
        assert idx[-1] - idx[0] + 1 == idx.size


def test_compute_delta_true_reduces_to_quadratic_when_g_true_zero() -> None:
    xj = np.asarray([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float64)
    a_true = 1.2
    b_true = -0.7
    g_true = np.zeros_like(xj)

    delta = dgp.compute_delta_true(xj, a_true, b_true, g_true)
    expected = a_true * xj + b_true * (xj**2)

    assert delta.shape == xj.shape
    assert np.isfinite(delta).all()
    np.testing.assert_allclose(delta, expected, rtol=0.0, atol=1e-12)


def test_compute_delta_true_uses_first_feature_column_when_xj_2d() -> None:
    x_scalar = np.asarray([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float64)
    xj_2d = np.stack([x_scalar, 999.0 * np.ones_like(x_scalar)], axis=1)  # (J,2)

    a_true = 0.4
    b_true = -0.2
    g_true = np.asarray([0.0, 1.5, -0.3, 0.7, 2.0], dtype=np.float64)

    delta_1d = dgp.compute_delta_true(x_scalar, a_true, b_true, g_true)
    delta_2d = dgp.compute_delta_true(xj_2d, a_true, b_true, g_true)

    assert delta_2d.shape == x_scalar.shape
    np.testing.assert_allclose(delta_2d, delta_1d, rtol=0.0, atol=1e-12)


def test_compute_njt_true_matches_group_effect_indexing() -> None:
    # J=4 products, G=2 groups
    group_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    T, G = 3, 2
    z_true = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.int64,
    )
    u_true_group = np.asarray(
        [
            [0.2, -0.5],
            [1.0, 2.0],
            [-3.0, 0.25],
        ],
        dtype=np.float64,
    )

    njt = dgp.compute_njt_true(group_id, z_true, u_true_group)
    assert njt.shape == (T, group_id.shape[0])

    group_effect = z_true * u_true_group  # (T,G)
    expected = group_effect[:, group_id]  # (T,J)
    np.testing.assert_allclose(njt, expected, rtol=0.0, atol=0.0)

    # If z_true is all zeros then njt is all zeros
    njt0 = dgp.compute_njt_true(group_id, np.zeros_like(z_true), u_true_group)
    np.testing.assert_allclose(njt0, 0.0, rtol=0.0, atol=0.0)


def test_probs_with_outside_sums_to_one_and_is_finite_under_extremes() -> None:
    # Moderate utilities
    u_j = np.asarray([0.1, -0.2, 0.0], dtype=np.float64)
    p_inside, p0 = dgp.probs_with_outside(u_j)
    assert p_inside.shape == u_j.shape
    assert np.isfinite(p_inside).all()
    assert np.isfinite(p0)
    assert (p_inside >= 0.0).all()
    assert p0 >= 0.0
    np.testing.assert_allclose(p0 + float(p_inside.sum()), 1.0, rtol=0.0, atol=1e-12)

    # Extreme utilities (numerical stability)
    u_j = np.asarray([1000.0, -1000.0, 0.0], dtype=np.float64)
    p_inside, p0 = dgp.probs_with_outside(u_j)
    assert np.isfinite(p_inside).all()
    assert np.isfinite(p0)
    assert (p_inside >= 0.0).all()
    assert p0 >= 0.0
    np.testing.assert_allclose(p0 + float(p_inside.sum()), 1.0, rtol=0.0, atol=1e-12)


def test_sample_counts_single_sums_to_N_and_probs_consistent() -> None:
    rng = np.random.default_rng(123)
    u_j = np.asarray([0.4, -0.1, 0.2, 0.0], dtype=np.float64)
    N = 200

    qj, q0, p_inside, p0 = dgp.sample_counts_single(rng, u_j, N)

    assert qj.shape == u_j.shape
    assert isinstance(q0, int)
    assert int(q0) >= 0
    assert (qj >= 0).all()
    assert int(q0) + int(qj.sum()) == N

    p_inside_ref, p0_ref = dgp.probs_with_outside(u_j)
    np.testing.assert_allclose(p_inside, p_inside_ref, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(p0, p0_ref, rtol=0.0, atol=1e-15)


def test_sample_counts_markets_row_sums_to_N() -> None:
    rng = np.random.default_rng(456)
    T, J = 5, 4
    u_tj = rng.normal(0.0, 1.0, size=(T, J))
    N = 300

    qjt, q0t = dgp.sample_counts_markets(rng, u_tj, N)
    assert qjt.shape == (T, J)
    assert q0t.shape == (T,)
    assert (qjt >= 0).all()
    assert (q0t >= 0).all()

    totals = q0t + qjt.sum(axis=1)
    np.testing.assert_array_equal(totals, np.full((T,), N, dtype=totals.dtype))


def test_generate_choice_learn_market_shocks_dgp_keys_shapes_and_sums() -> None:
    out = dgp.generate_choice_learn_market_shocks_dgp(
        seed=7,
        num_markets=4,
        num_products=6,
        num_groups=2,
        N_base=500,
        N_shock=400,
        num_features=1,
        x_sd=1.0,
        coef_sd=1.0,
        p_g_active=0.3,
        g_sd=1.0,
        sd_E=0.5,
        p_active=0.4,
        sd_u=0.6,
    )

    required_keys = {
        "xj",
        "group_id",
        "qj_base",
        "q0_base",
        "p_base",
        "p0_base",
        "qjt_shock",
        "q0t_shock",
        "a_true",
        "b_true",
        "g_true",
        "delta_true",
        "E_bar_true",
        "njt_true",
    }
    assert set(out.keys()) == required_keys

    xj = out["xj"]
    group_id = out["group_id"]
    qj_base = out["qj_base"]
    q0_base = out["q0_base"]
    p_base = out["p_base"]
    p0_base = out["p0_base"]
    qjt_shock = out["qjt_shock"]
    q0t_shock = out["q0t_shock"]
    g_true = out["g_true"]
    delta_true = out["delta_true"]
    E_bar_true = out["E_bar_true"]
    njt_true = out["njt_true"]

    T = 4
    J = 6
    G = 2
    num_features = 1

    assert xj.shape == (J, num_features)
    assert group_id.shape == (J,)
    assert qj_base.shape == (J,)
    assert isinstance(q0_base, int)
    assert p_base.shape == (J,)
    assert isinstance(p0_base, float)
    assert qjt_shock.shape == (T, J)
    assert q0t_shock.shape == (T,)
    assert g_true.shape == (J,)
    assert delta_true.shape == (J,)
    assert E_bar_true.shape == (T,)
    assert njt_true.shape == (T, J)

    # Range checks
    assert group_id.min() == 0
    assert group_id.max() == G - 1

    # Count sums
    assert q0_base + int(qj_base.sum()) == 500
    np.testing.assert_array_equal(q0t_shock + qjt_shock.sum(axis=1), np.full((T,), 400))

    # Probabilities sum
    assert np.isfinite(p_base).all()
    assert np.isfinite(p0_base)
    np.testing.assert_allclose(
        float(p0_base) + float(p_base.sum()), 1.0, rtol=0.0, atol=1e-12
    )

    # Finiteness of truth arrays
    assert np.isfinite(xj).all()
    assert np.isfinite(g_true).all()
    assert np.isfinite(delta_true).all()
    assert np.isfinite(E_bar_true).all()
    assert np.isfinite(njt_true).all()


def test_generate_choice_learn_market_shocks_dgp_rejects_invalid_num_features() -> None:
    with pytest.raises(ValueError):
        _ = dgp.generate_choice_learn_market_shocks_dgp(
            seed=0,
            num_markets=2,
            num_products=5,
            num_groups=2,
            N_base=100,
            N_shock=80,
            num_features=0,
            x_sd=1.0,
            coef_sd=1.0,
            p_g_active=0.2,
            g_sd=1.0,
            sd_E=0.5,
            p_active=0.2,
            sd_u=0.5,
        )


def test_generate_choice_learn_market_shocks_dgp_is_deterministic_given_seed() -> None:
    kwargs = dict(
        seed=11,
        num_markets=3,
        num_products=7,
        num_groups=3,
        N_base=200,
        N_shock=150,
        num_features=2,
        x_sd=1.0,
        coef_sd=1.0,
        p_g_active=0.25,
        g_sd=1.0,
        sd_E=0.5,
        p_active=0.3,
        sd_u=0.7,
    )
    out1 = dgp.generate_choice_learn_market_shocks_dgp(**kwargs)
    out2 = dgp.generate_choice_learn_market_shocks_dgp(**kwargs)

    # Scalars
    assert out1["a_true"] == out2["a_true"]
    assert out1["b_true"] == out2["b_true"]
    assert out1["q0_base"] == out2["q0_base"]
    assert out1["p0_base"] == out2["p0_base"]

    # Integers
    np.testing.assert_array_equal(out1["group_id"], out2["group_id"])
    np.testing.assert_array_equal(out1["qj_base"], out2["qj_base"])
    np.testing.assert_array_equal(out1["qjt_shock"], out2["qjt_shock"])
    np.testing.assert_array_equal(out1["q0t_shock"], out2["q0t_shock"])

    # Floats (exact determinism is expected; allclose is robust)
    np.testing.assert_allclose(out1["xj"], out2["xj"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out1["g_true"], out2["g_true"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        out1["delta_true"], out2["delta_true"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(out1["p_base"], out2["p_base"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        out1["E_bar_true"], out2["E_bar_true"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(out1["njt_true"], out2["njt_true"], rtol=0.0, atol=0.0)
