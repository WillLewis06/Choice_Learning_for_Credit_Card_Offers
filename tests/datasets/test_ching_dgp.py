# tests/datasets/test_ching_dgp.py
from __future__ import annotations

import numpy as np
import pytest

import datasets.ching_dgp as dgp


def test_sample_theta_true_shapes_ranges_and_types() -> None:
    rng = np.random.default_rng(123)
    M, N = 2, 3
    theta = dgp.sample_theta_true(rng=rng, M=M, N=N)

    assert set(theta.keys()) == {"beta", "alpha", "v", "fc", "lambda_c"}
    for k, v in theta.items():
        assert v.shape == (M, N)
        assert v.dtype == np.float64
        assert np.isfinite(v).all()

    assert np.all((theta["beta"] > 0.0) & (theta["beta"] < 1.0))
    assert np.all((theta["lambda_c"] > 0.0) & (theta["lambda_c"] < 1.0))
    assert np.all(theta["alpha"] > 0.0)
    assert np.all(theta["v"] > 0.0)
    assert np.all(theta["fc"] > 0.0)


def test_logsumexp2_matches_reference_and_is_stable() -> None:
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(4, 3))
    x1 = rng.normal(size=(4, 3))

    out = dgp.logsumexp2(x0, x1)
    ref = np.log(np.exp(x0) + np.exp(x1))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    # Extreme stability
    x0e = np.array([1000.0])
    x1e = np.array([-1000.0])
    oute = dgp.logsumexp2(x0e, x1e)
    assert np.isfinite(oute).all()
    np.testing.assert_allclose(oute, 1000.0, rtol=0.0, atol=1e-9)


def test_compute_u_m_matches_formula_for_product_index() -> None:
    delta_true = np.asarray([0.2, 0.5, -0.1], dtype=np.float64)  # (J,)
    E_bar_true = np.asarray([1.0, 2.0], dtype=np.float64)  # (M,)
    njt_true = np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float64)  # (M,J)
    j_star = 1

    u_m = dgp.compute_u_m(delta_true, E_bar_true, njt_true, j_star)
    expected = delta_true[j_star] + E_bar_true + njt_true[:, j_star]

    assert u_m.shape == (E_bar_true.shape[0],)
    np.testing.assert_allclose(u_m, expected, rtol=0.0, atol=0.0)


def test_next_inventory_clips_to_bounds() -> None:
    I_max = 2

    # I=0, consume=1 cannot go negative
    out = dgp.next_inventory(
        I=np.asarray([0], dtype=np.int64),
        a=np.asarray([0], dtype=bool),
        c=np.asarray([1], dtype=bool),
        I_max=I_max,
    )
    np.testing.assert_array_equal(out, np.asarray([0], dtype=np.int64))

    # I=I_max, buy=1 cannot exceed cap
    out = dgp.next_inventory(
        I=np.asarray([I_max], dtype=np.int64),
        a=np.asarray([1], dtype=bool),
        c=np.asarray([0], dtype=bool),
        I_max=I_max,
    )
    np.testing.assert_array_equal(out, np.asarray([I_max], dtype=np.int64))

    # General bounds
    I = np.asarray([0, 1, 2], dtype=np.int64)
    a = np.asarray([1, 0, 1], dtype=bool)
    c = np.asarray([0, 1, 1], dtype=bool)
    out = dgp.next_inventory(I=I, a=a, c=c, I_max=I_max)
    assert out.min() >= 0
    assert out.max() <= I_max
    assert out.shape == I.shape


def test_simulate_price_states_absorbing_remains_constant_and_states_valid() -> None:
    rng = np.random.default_rng(7)
    S = 2
    P_absorb = np.eye(S, dtype=np.float64)
    T = 10
    start_state = 1

    s_t = dgp.simulate_price_states(
        rng=rng, P_price=P_absorb, T=T, start_state=start_state
    )
    assert s_t.shape == (T,)
    assert s_t.dtype == np.int64
    assert np.all(s_t == start_state)
    assert s_t.min() >= 0 and s_t.max() <= S - 1


def test_simulate_consumption_shape_and_extremes() -> None:
    rng = np.random.default_rng(9)
    N, T = 4, 7

    lam0 = np.zeros((N,), dtype=np.float64)
    c0 = dgp.simulate_consumption(rng=rng, N=N, T=T, lambda_c_n=lam0)
    assert c0.shape == (N, T)
    assert c0.dtype == bool
    assert np.all(c0 == False)

    lam1 = np.ones((N,), dtype=np.float64)
    c1 = dgp.simulate_consumption(rng=rng, N=N, T=T, lambda_c_n=lam1)
    assert np.all(c1 == True)


def test_solve_ccp_buy_shape_range_and_absorbing_price_monotonicity() -> None:
    # With beta=0, CCP reduces to static logit in (u1-u0), so cheaper state must imply >= buy prob.
    u_m = 1.0
    beta = 0.0
    alpha = 1.0
    v = 2.0
    fc = 0.1
    lambda_c = 0.3
    I_max = 2
    price_vals = np.asarray([1.0, 0.8], dtype=np.float64)  # state 1 cheaper
    P_absorb = np.eye(2, dtype=np.float64)
    waste_cost = 0.2

    ccp = dgp.solve_ccp_buy(
        u_m=u_m,
        beta=beta,
        alpha=alpha,
        v=v,
        fc=fc,
        lambda_c=lambda_c,
        I_max=I_max,
        P_price=P_absorb,
        price_vals=price_vals,
        waste_cost=waste_cost,
        tol=1e-10,
        max_iter=50,
    )

    assert ccp.shape == (2, I_max + 1)
    assert np.isfinite(ccp).all()
    assert np.all((ccp > 0.0) & (ccp < 1.0))

    # Cheaper state => higher buy prob for each inventory
    assert np.all(ccp[1, :] >= ccp[0, :] - 1e-12)


def _base_inputs():
    M = 3
    J = 4
    delta_true = np.linspace(-0.2, 0.3, J, dtype=np.float64)
    E_bar_true = np.linspace(0.1, 0.5, M, dtype=np.float64)
    njt_true = np.zeros((M, J), dtype=np.float64)
    njt_true[:, 1] = np.asarray([0.0, 0.2, -0.1], dtype=np.float64)

    product_index = 1
    N = 5
    T = 6
    I_max = 2
    price_vals = np.asarray([1.0, 0.8], dtype=np.float64)
    P_price = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    waste_cost = 0.1
    tol = 1e-8
    max_iter = 80

    return dict(
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        product_index=product_index,
        N=N,
        T=T,
        I_max=I_max,
        P_price=P_price,
        price_vals=price_vals,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
    )


def test_generate_dgp_shapes_types_and_internal_consistency() -> None:
    args = _base_inputs()
    seed = 123

    a_imt, p_state_mt, u_m_true, theta_true = dgp.generate_dgp(seed=seed, **args)

    M = args["E_bar_true"].shape[0]
    N = args["N"]
    T = args["T"]
    S = args["P_price"].shape[0]

    assert a_imt.shape == (M, N, T)
    assert a_imt.dtype == np.int64
    uniq_a = np.unique(a_imt)
    assert set(uniq_a.tolist()).issubset({0, 1})

    assert p_state_mt.shape == (M, T)
    assert p_state_mt.dtype == np.int64
    assert p_state_mt.min() >= 0
    assert p_state_mt.max() <= S - 1

    assert u_m_true.shape == (M,)
    assert u_m_true.dtype == np.float64
    assert np.isfinite(u_m_true).all()

    assert set(theta_true.keys()) == {"beta", "alpha", "v", "fc", "lambda_c"}
    for k, v in theta_true.items():
        assert v.shape == (M, N)
        assert v.dtype == np.float64
        assert np.isfinite(v).all()

    # Consistency: u_m_true matches compute_u_m formula
    u_ref = dgp.compute_u_m(
        args["delta_true"], args["E_bar_true"], args["njt_true"], args["product_index"]
    )
    np.testing.assert_allclose(u_m_true, u_ref, rtol=0.0, atol=0.0)


def test_generate_dgp_is_deterministic_given_seed() -> None:
    args = _base_inputs()
    seed = 999

    out1 = dgp.generate_dgp(seed=seed, **args)
    out2 = dgp.generate_dgp(seed=seed, **args)

    a1, s1, u1, th1 = out1
    a2, s2, u2, th2 = out2

    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_allclose(u1, u2, rtol=0.0, atol=0.0)
    for k in th1.keys():
        np.testing.assert_allclose(th1[k], th2[k], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "bad_case",
    ["product_index_oob", "P_price_rows_not_stochastic", "price_vals_wrong_length"],
)
def test_generate_dgp_rejects_invalid_inputs(bad_case: str) -> None:
    args = _base_inputs()

    if bad_case == "product_index_oob":
        args["product_index"] = int(args["delta_true"].shape[0])  # out of bounds
    elif bad_case == "P_price_rows_not_stochastic":
        P = args["P_price"].copy()
        P[0, :] = np.asarray([0.2, 0.2], dtype=np.float64)  # row sum != 1
        args["P_price"] = P
    elif bad_case == "price_vals_wrong_length":
        args["price_vals"] = np.asarray([1.0, 0.8, 0.6], dtype=np.float64)  # len != S
    else:
        raise ValueError("Unknown bad_case")

    with pytest.raises(ValueError):
        _ = dgp.generate_dgp(seed=1, **args)
