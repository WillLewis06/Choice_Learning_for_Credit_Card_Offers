# tests/ching/test_ching_dgp.py
from __future__ import annotations

import numpy as np
import pytest

import datasets.ching_dgp as dgp


def test_sample_theta_true_shapes_ranges_and_types() -> None:
    rng = np.random.default_rng(123)
    M, N, J = 2, 3, 4

    theta = dgp.sample_theta_true(rng=rng, M=M, N=N, J=J)

    assert set(theta.keys()) == {"beta", "alpha", "v", "fc", "lambda"}

    # Global beta (0-d ndarray)
    beta = theta["beta"]
    assert isinstance(beta, np.ndarray)
    assert beta.shape == ()
    assert beta.dtype == np.float64
    assert np.isfinite(beta)

    # Per-product blocks
    for k in ["alpha", "v", "fc"]:
        arr = theta[k]
        assert arr.shape == (J,)
        assert arr.dtype == np.float64
        assert np.isfinite(arr).all()

    # Market-consumer block
    lam = theta["lambda"]
    assert lam.shape == (M, N)
    assert lam.dtype == np.float64
    assert np.isfinite(lam).all()

    # Constraints
    assert (float(beta) > 0.0) and (float(beta) < 1.0)
    assert np.all((lam > 0.0) & (lam < 1.0))
    assert np.all(theta["alpha"] > 0.0)
    assert np.all(theta["v"] > 0.0)
    assert np.all(theta["fc"] > 0.0)


def test_compute_u_mj_matches_formula() -> None:
    delta_true = np.asarray([0.2, 0.5, -0.1], dtype=np.float64)  # (J,)
    E_bar_true = np.asarray([1.0, 2.0], dtype=np.float64)  # (M,)
    njt_true = np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float64)  # (M,J)

    u_mj = dgp.compute_u_mj(delta_true, E_bar_true, njt_true)
    expected = delta_true[None, :] + E_bar_true[:, None] + njt_true

    assert u_mj.shape == (E_bar_true.shape[0], delta_true.shape[0])
    np.testing.assert_allclose(u_mj, expected, rtol=0.0, atol=0.0)


def test_next_inventory_clips_to_bounds() -> None:
    I_max = 2

    # I=0, consume=1 cannot go negative
    out = dgp.next_inventory(
        I=np.asarray([0], dtype=np.int64),
        a=np.asarray([0], dtype=np.int64),
        c=np.asarray([1], dtype=np.int64),
        I_max=I_max,
    )
    np.testing.assert_array_equal(out, np.asarray([0], dtype=np.int64))

    # I=I_max, buy=1 cannot exceed cap
    out = dgp.next_inventory(
        I=np.asarray([I_max], dtype=np.int64),
        a=np.asarray([1], dtype=np.int64),
        c=np.asarray([0], dtype=np.int64),
        I_max=I_max,
    )
    np.testing.assert_array_equal(out, np.asarray([I_max], dtype=np.int64))

    # General bounds and shape
    I = np.asarray([0, 1, 2], dtype=np.int64)
    a = np.asarray([1, 0, 1], dtype=np.int64)
    c = np.asarray([0, 1, 1], dtype=np.int64)
    out = dgp.next_inventory(I=I, a=a, c=c, I_max=I_max)

    assert out.shape == I.shape
    assert out.min() >= 0
    assert out.max() <= I_max


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
    c0 = dgp.simulate_consumption(rng=rng, N=N, T=T, lambda_n=lam0)
    assert c0.shape == (N, T)
    assert c0.dtype == bool
    assert np.all(c0 == False)

    lam1 = np.ones((N,), dtype=np.float64)
    c1 = dgp.simulate_consumption(rng=rng, N=N, T=T, lambda_n=lam1)
    assert c1.shape == (N, T)
    assert np.all(c1 == True)


def test_solve_ccp_buy_shape_range_and_absorbing_price_monotonicity() -> None:
    # With beta=0, CCP reduces to static logit in (u1-u0), so cheaper state must imply >= buy prob.
    u_eff = 1.0
    beta = 0.0
    alpha_j = 1.0
    v_j = 2.0
    fc_j = 0.1
    lambda_n = 0.3
    I_max = 2
    price_vals = np.asarray([1.0, 0.8], dtype=np.float64)  # state 1 cheaper
    P_absorb = np.eye(2, dtype=np.float64)
    waste_cost = 0.2

    ccp = dgp.solve_ccp_buy(
        u_eff=u_eff,
        beta=beta,
        alpha_j=alpha_j,
        v_j=v_j,
        fc_j=fc_j,
        lambda_n=lambda_n,
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


def _base_inputs() -> dict[str, object]:
    M = 3
    J = 4
    S = 2

    delta_true = np.linspace(-0.2, 0.3, J, dtype=np.float64)
    E_bar_true = np.linspace(0.1, 0.5, M, dtype=np.float64)
    njt_true = np.zeros((M, J), dtype=np.float64)
    njt_true[:, 1] = np.asarray([0.0, 0.2, -0.1], dtype=np.float64)

    N = 5
    T = 6
    I_max = 2

    # Base Markov matrix and price grid, then tile across (M,J).
    P_base = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)  # (S,S)
    price_base = np.asarray([1.0, 0.8], dtype=np.float64)  # (S,)

    P_price_mj = np.tile(P_base[None, None, :, :], (M, J, 1, 1))  # (M,J,S,S)
    price_vals_mj = np.tile(price_base[None, None, :], (M, J, 1))  # (M,J,S)

    waste_cost = 0.1
    tol = 1e-8
    max_iter = 80

    return dict(
        delta_true=delta_true,
        E_bar_true=E_bar_true,
        njt_true=njt_true,
        N=N,
        T=T,
        I_max=I_max,
        P_price_mj=P_price_mj,
        price_vals_mj=price_vals_mj,
        waste_cost=waste_cost,
        tol=tol,
        max_iter=max_iter,
    )


def test_generate_dgp_shapes_types_and_internal_consistency() -> None:
    args = _base_inputs()
    seed = 123

    a_mnjt, p_state_mjt, u_mj_true, theta_true = dgp.generate_dgp(seed=seed, **args)

    M = args["E_bar_true"].shape[0]
    N = int(args["N"])
    T = int(args["T"])
    J = args["delta_true"].shape[0]
    S = args["P_price_mj"].shape[2]

    assert a_mnjt.shape == (M, N, J, T)
    assert a_mnjt.dtype == np.int64
    uniq_a = np.unique(a_mnjt)
    assert set(uniq_a.tolist()).issubset({0, 1})

    assert p_state_mjt.shape == (M, J, T)
    assert p_state_mjt.dtype == np.int64
    assert p_state_mjt.min() >= 0
    assert p_state_mjt.max() <= S - 1

    assert u_mj_true.shape == (M, J)
    assert u_mj_true.dtype == np.float64
    assert np.isfinite(u_mj_true).all()

    assert set(theta_true.keys()) == {"beta", "alpha", "v", "fc", "lambda"}

    beta = theta_true["beta"]
    assert isinstance(beta, np.ndarray)
    assert beta.shape == ()
    assert beta.dtype == np.float64
    assert np.isfinite(beta)

    for k in ["alpha", "v", "fc"]:
        arr = theta_true[k]
        assert arr.shape == (J,)
        assert arr.dtype == np.float64
        assert np.isfinite(arr).all()

    lam = theta_true["lambda"]
    assert lam.shape == (M, N)
    assert lam.dtype == np.float64
    assert np.isfinite(lam).all()

    # Consistency: u_mj_true matches compute_u_mj formula
    u_ref = dgp.compute_u_mj(args["delta_true"], args["E_bar_true"], args["njt_true"])
    np.testing.assert_allclose(u_mj_true, u_ref, rtol=0.0, atol=0.0)


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

    assert th1.keys() == th2.keys()
    for k in th1.keys():
        np.testing.assert_allclose(th1[k], th2[k], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "bad_case",
    [
        "P_price_mj_rows_not_stochastic",
        "price_vals_mj_wrong_shape",
        "P_price_mj_wrong_shape",
    ],
)
def test_generate_dgp_rejects_invalid_inputs(bad_case: str) -> None:
    args = _base_inputs()

    if bad_case == "P_price_mj_rows_not_stochastic":
        P = args["P_price_mj"].copy()
        # Break one row sum badly enough to exceed validator tolerance.
        P[0, 0, 0, :] = np.asarray([0.2, 0.2], dtype=np.float64)
        args["P_price_mj"] = P

    elif bad_case == "price_vals_mj_wrong_shape":
        pv = args["price_vals_mj"]
        # Add an extra price state so shape becomes (M,J,S+1).
        args["price_vals_mj"] = np.concatenate([pv, pv[..., :1]], axis=2)

    elif bad_case == "P_price_mj_wrong_shape":
        # Drop one dimension (ndim != 4).
        args["P_price_mj"] = args["P_price_mj"][0]

    else:
        raise ValueError("Unknown bad_case")

    with pytest.raises(ValueError):
        _ = dgp.generate_dgp(seed=1, **args)
