import numpy as np
import pytest

from datasets.lu_dgp import (
    BasicLuChoiceModel,
    _generate_market_shares,
    generate_market,
    generate_market_conditions,
)


# -----------------------------------------------------------------------------
# Local deterministic helpers (NumPy-only)
# -----------------------------------------------------------------------------
def assert_finite_np(x: np.ndarray, name: str = "array") -> None:
    """
    Assert that a NumPy array contains only finite values.

    This is defined locally to keep this test module independent of pytest's
    conftest mechanism.
    """
    x = np.asarray(x)
    ok = np.isfinite(x)
    if not np.all(ok):
        idx = np.argwhere(~ok)
        preview = idx[:5].tolist()
        raise AssertionError(f"{name} contains non-finite values at indices {preview}.")


def _assert_prob_simplex(sjt: np.ndarray, s0t: np.ndarray, atol: float = 1e-12) -> None:
    """
    sjt: (T,J), s0t: (T,)
    Checks: finite, bounds, and s0t + sum_j sjt == 1 marketwise.
    """
    assert_finite_np(sjt, name="sjt")
    assert_finite_np(s0t, name="s0t")

    assert sjt.ndim == 2
    assert s0t.ndim == 1

    assert np.all(sjt >= -atol)
    assert np.all(sjt <= 1.0 + atol)
    assert np.all(s0t >= -atol)
    assert np.all(s0t <= 1.0 + atol)

    err = np.max(np.abs(s0t + sjt.sum(axis=1) - 1.0))
    assert (
        err <= atol
    ), f"Share identity violated (max abs err={err:.3e}, atol={atol:.3e})."


def _deterministic_panel(T: int, J: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic (pjt, wjt, Ejt) with:
      - wjt in [1, 2]
      - moderate magnitudes for stability
    """
    pjt = np.linspace(-1.0, 1.0, T * J, dtype=float).reshape(T, J)

    w_grid = np.linspace(1.0, 2.0, J, dtype=float)[None, :]
    wjt = np.repeat(w_grid, repeats=T, axis=0)

    Ejt = np.linspace(-0.5, 0.5, T * J, dtype=float).reshape(T, J)
    return pjt, wjt, Ejt


# -----------------------------------------------------------------------------
# generate_market_conditions
# -----------------------------------------------------------------------------
def test_generate_market_conditions_invalid_dgp_type_raises():
    for bad in [0, 5, -1, 999]:
        with pytest.raises(ValueError):
            generate_market_conditions(T=3, J=4, dgp_type=bad, seed=123)


def test_generate_market_conditions_contract_all_types():
    """
    Single contract test for dgp_type in {1,2,3,4}:
      - shapes
      - finiteness
      - wjt support
    """
    T, J = 7, 11
    for dgp_type in [1, 2, 3, 4]:
        wjt, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=123
        )

        assert wjt.shape == (T, J)
        assert Ejt.shape == (T, J)
        assert ujt.shape == (T, J)
        assert alpha.shape == (T, J)

        assert_finite_np(wjt, name="wjt")
        assert_finite_np(Ejt, name="Ejt")
        assert_finite_np(ujt, name="ujt")
        assert_finite_np(alpha, name="alpha")

        assert np.min(wjt) >= 1.0
        assert np.max(wjt) <= 2.0


def test_generate_market_conditions_njt_structure_sparse_dgp12():
    """
    For DGP 1/2:
      - E_bar_t is fixed at -1, so njt = Ejt + 1
      - n_active = int(0.4*J)
      - for j < n_active: njt[t,j] = +1 if j even else -1
      - for j >= n_active: njt[t,j] = 0
    """
    T, J = 10, 15
    n_active = int(0.4 * J)
    expected_active = np.array([1.0 if (j % 2 == 0) else -1.0 for j in range(n_active)])

    for dgp_type in [1, 2]:
        wjt, Ejt, ujt, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=123
        )
        njt = Ejt + 1.0

        assert np.all(njt[:, :n_active] == expected_active[None, :])
        assert np.all(njt[:, n_active:] == 0.0)

        assert_finite_np(wjt, name="wjt")
        assert_finite_np(ujt, name="ujt")
        assert_finite_np(alpha, name="alpha")


def test_generate_market_conditions_alpha_rules_all_types():
    """
    Alpha rules (using njt = Ejt + 1 because E_bar_t = -1):
      - dgp 1,3: alpha == 0
      - dgp 2: alpha = +0.3 if njt==+1, -0.3 if njt==-1, else 0
      - dgp 4: alpha = +0.3 if njt>=1/3, -0.3 if njt<=-1/3, else 0
    """
    T, J = 6, 15
    thr = 1.0 / 3.0

    for dgp_type in [1, 2, 3, 4]:
        _, Ejt, _, alpha = generate_market_conditions(
            T=T, J=J, dgp_type=dgp_type, seed=123
        )
        njt = Ejt + 1.0

        if dgp_type in [1, 3]:
            assert np.all(alpha == 0.0)

        elif dgp_type == 2:
            expected = np.zeros((T, J), dtype=float)
            expected[njt == 1.0] = 0.3
            expected[njt == -1.0] = -0.3
            assert np.array_equal(alpha, expected)

        else:  # dgp_type == 4
            expected = np.zeros((T, J), dtype=float)
            expected[njt >= thr] = 0.3
            expected[njt <= -thr] = -0.3
            assert np.array_equal(alpha, expected)


# -----------------------------------------------------------------------------
# BasicLuChoiceModel.utilities
# -----------------------------------------------------------------------------
def test_utilities_validation_raises():
    model = BasicLuChoiceModel(N=10, beta_p=-1.0, beta_w=0.5, sigma=1.0, seed=123)

    pjt = np.zeros((3, 4), dtype=float)
    wjt = np.zeros((3, 4), dtype=float)
    Ejt = np.zeros((3, 4), dtype=float)

    with pytest.raises(ValueError):
        model.utilities(pjt=pjt[0], wjt=wjt, Ejt=Ejt)
    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt[..., None], Ejt=Ejt)
    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt[..., None])

    wjt_bad = np.zeros((3, 5), dtype=float)
    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt_bad, Ejt=Ejt)


def test_utilities_output_shape_and_finite():
    T, J, N_sim = 4, 6, 17
    model = BasicLuChoiceModel(N=N_sim, beta_p=-1.0, beta_w=0.5, sigma=1.0, seed=123)

    pjt, wjt, Ejt = _deterministic_panel(T, J)
    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    assert uijt.shape == (T, N_sim, J)
    assert_finite_np(uijt, name="uijt")


def test_utilities_sigma_zero_constant_across_consumers():
    T, J, N_sim = 4, 6, 25
    model = BasicLuChoiceModel(N=N_sim, beta_p=-1.0, beta_w=0.5, sigma=0.0, seed=123)

    pjt, wjt, Ejt = _deterministic_panel(T, J)
    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    assert np.allclose(uijt, uijt[:, :1, :], atol=0.0, rtol=0.0)


def test_utilities_reduces_when_beta_w_zero_and_Ejt_zero():
    T, J, N_sim = 3, 5, 12
    model = BasicLuChoiceModel(N=N_sim, beta_p=-2.0, beta_w=0.0, sigma=1.3, seed=123)

    pjt, wjt, _ = _deterministic_panel(T, J)
    Ejt = np.zeros((T, J), dtype=float)

    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    beta_p_i = np.asarray(model.beta_p_i, dtype=float)
    if beta_p_i.ndim == 0:
        beta_p_i = np.full((N_sim,), float(beta_p_i), dtype=float)

    expected = beta_p_i[None, :, None] * pjt[:, None, :]
    assert np.allclose(uijt, expected)


# -----------------------------------------------------------------------------
# _generate_market_shares
# -----------------------------------------------------------------------------
def test_generate_market_shares_rejects_non_3d_uijt():
    with pytest.raises(ValueError):
        _generate_market_shares(np.zeros((3, 4), dtype=float))


def test_generate_market_shares_contract_and_extremes():
    T, J, N_sim = 4, 7, 50

    uijt = np.linspace(-1.0, 1.0, T * N_sim * J, dtype=float).reshape(T, N_sim, J)
    sjt, s0t = _generate_market_shares(uijt)

    assert sjt.shape == (T, J)
    assert s0t.shape == (T,)
    _assert_prob_simplex(sjt, s0t)

    for level in [-50.0, 50.0]:
        u_ext = np.full((T, N_sim, J), level, dtype=float)
        sjt_e, s0t_e = _generate_market_shares(u_ext)
        _assert_prob_simplex(sjt_e, s0t_e)

        if level < 0:
            assert np.all(s0t_e > 1.0 - 1e-10)
            assert np.all(sjt_e < 1e-10)
        else:
            assert np.all(s0t_e < 1e-10)
            assert np.all(np.abs(sjt_e.sum(axis=1) - 1.0) < 1e-10)


def test_generate_market_shares_monotone_under_additive_shift():
    T, J, N_sim = 4, 6, 40
    uijt = np.linspace(-0.5, 0.5, T * N_sim * J, dtype=float).reshape(T, N_sim, J)

    sjt0, s0t0 = _generate_market_shares(uijt)
    sjt1, s0t1 = _generate_market_shares(uijt + 1.0)

    assert np.all(s0t1 < s0t0)
    assert np.all(sjt1.sum(axis=1) > sjt0.sum(axis=1))

    _assert_prob_simplex(sjt0, s0t0)
    _assert_prob_simplex(sjt1, s0t1)


# -----------------------------------------------------------------------------
# generate_market
# -----------------------------------------------------------------------------
def test_generate_market_rejects_non_3d_uijt():
    with pytest.raises(ValueError):
        generate_market(uijt=np.zeros((3, 4), dtype=float), N=100, seed=123)


def test_generate_market_count_identity_and_shapes():
    T, J, N_sim = 5, 8, 200
    N = 2000

    uijt = np.linspace(-1.0, 1.0, T * N_sim * J, dtype=float).reshape(T, N_sim, J)
    sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=123)

    assert sjt.shape == (T, J)
    assert s0t.shape == (T,)
    assert qjt.shape == (T, J)
    assert q0t.shape == (T,)

    assert np.issubdtype(qjt.dtype, np.integer)
    assert np.issubdtype(q0t.dtype, np.integer)
    assert np.all(qjt >= 0)
    assert np.all(q0t >= 0)

    totals = q0t + qjt.sum(axis=1)
    assert np.all(
        totals == N
    ), f"Count identity violated: min={totals.min()}, max={totals.max()}, N={N}"

    _assert_prob_simplex(sjt, s0t)
