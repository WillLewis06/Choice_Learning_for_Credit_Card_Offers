import numpy as np
import pytest

from datasets.dgp import (
    generate_market_conditions,
    BasicLuChoiceModel,
    _generate_market_shares,
    generate_market,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _assert_all_finite(*arrays) -> None:
    for a in arrays:
        assert np.all(np.isfinite(a)), "Found non-finite values."


def _assert_in_unit_interval(x: np.ndarray, *, atol: float = 0.0) -> None:
    assert np.all(x >= -atol), f"Values below 0 (min={x.min()})."
    assert np.all(x <= 1.0 + atol), f"Values above 1 (max={x.max()})."


def _assert_prob_simplex(
    sjt: np.ndarray, s0t: np.ndarray, *, atol: float = 1e-12
) -> None:
    _assert_all_finite(sjt, s0t)
    _assert_in_unit_interval(sjt, atol=atol)
    _assert_in_unit_interval(s0t, atol=atol)

    err = np.max(np.abs(s0t + sjt.sum(axis=1) - 1.0))
    assert (
        err <= atol
    ), f"Share identity violated (max abs err={err:.3e}, atol={atol:.3e})."


def _mean_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _make_problem(*, T: int = 5, J: int = 8, N_sim: int = 1000) -> tuple[int, int, int]:
    return int(T), int(J), int(N_sim)


# -----------------------------------------------------------------------------
# generate_market_conditions
# -----------------------------------------------------------------------------
def test_generate_market_conditions_invalid_dgp_type_raises():
    with pytest.raises(ValueError):
        generate_market_conditions(T=3, J=4, dgp_type=0, seed=123)
    with pytest.raises(ValueError):
        generate_market_conditions(T=3, J=4, dgp_type=5, seed=123)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_generate_market_conditions_shapes_and_finite(dgp_type):
    T, J, _ = _make_problem(T=7, J=11, N_sim=10)
    wjt, Ejt, ujt, alpha = generate_market_conditions(
        T=T, J=J, dgp_type=dgp_type, seed=123
    )

    assert wjt.shape == (T, J)
    assert Ejt.shape == (T, J)
    assert ujt.shape == (T, J)
    assert alpha.shape == (T, J)

    _assert_all_finite(wjt, Ejt, ujt, alpha)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_generate_market_conditions_wjt_support(dgp_type):
    T, J, _ = _make_problem(T=7, J=11, N_sim=10)
    wjt, Ejt, ujt, alpha = generate_market_conditions(
        T=T, J=J, dgp_type=dgp_type, seed=123
    )

    assert np.min(wjt) >= 1.0
    assert np.max(wjt) <= 2.0
    _assert_all_finite(Ejt, ujt, alpha)


@pytest.mark.parametrize("dgp_type", [1, 2])
def test_generate_market_conditions_njt_structure_sparse_dgp12(dgp_type):
    """
    For DGP 1/2:
      - E_bar_t is fixed at -1, so njt = Ejt - E_bar_t = Ejt + 1
      - n_active = int(0.4*J)
      - for j < n_active: njt[t,j] = +1 if j even else -1
      - for j >= n_active: njt[t,j] = 0
    """
    T, J, _ = _make_problem(T=10, J=15, N_sim=10)
    wjt, Ejt, ujt, alpha = generate_market_conditions(
        T=T, J=J, dgp_type=dgp_type, seed=123
    )

    njt = Ejt + 1.0
    n_active = int(0.4 * J)

    for t in range(T):
        active = njt[t, :n_active]
        inactive = njt[t, n_active:]

        expected_active = np.array(
            [1.0 if (j % 2 == 0) else -1.0 for j in range(n_active)],
            dtype=float,
        )

        assert np.array_equal(active, expected_active)
        assert np.array_equal(inactive, np.zeros(J - n_active, dtype=float))

    _assert_all_finite(wjt, ujt, alpha)


@pytest.mark.parametrize("dgp_type", [1, 3])
def test_generate_market_conditions_alpha_rules_dgp1_and_dgp3(dgp_type):
    T, J, _ = _make_problem(T=6, J=12, N_sim=10)
    _, Ejt, _, alpha = generate_market_conditions(T=T, J=J, dgp_type=dgp_type, seed=123)
    assert np.array_equal(alpha, np.zeros((T, J), dtype=float))

    # Sanity: njt exists (via Ejt + 1), but alpha is still zero
    njt = Ejt + 1.0
    _assert_all_finite(njt)


def test_generate_market_conditions_alpha_rules_dgp2():
    T, J, _ = _make_problem(T=6, J=15, N_sim=10)
    _, Ejt, _, alpha = generate_market_conditions(T=T, J=J, dgp_type=2, seed=123)
    njt = Ejt + 1.0  # since E_bar_t = -1

    expected = np.zeros((T, J), dtype=float)
    expected[njt == 1.0] = 0.3
    expected[njt == -1.0] = -0.3

    assert np.array_equal(alpha, expected)


def test_generate_market_conditions_alpha_rules_dgp4():
    T, J, _ = _make_problem(T=6, J=12, N_sim=10)
    _, Ejt, _, alpha = generate_market_conditions(T=T, J=J, dgp_type=4, seed=123)
    njt = Ejt + 1.0  # since E_bar_t = -1

    thr = 1.0 / 3.0
    expected = np.zeros((T, J), dtype=float)
    expected[njt >= thr] = 0.3
    expected[njt <= -thr] = -0.3

    assert np.array_equal(alpha, expected)


# -----------------------------------------------------------------------------
# BasicLuChoiceModel.utilities
# -----------------------------------------------------------------------------
def test_utilities_rejects_bad_rank_inputs():
    model = BasicLuChoiceModel(N=10, beta_p=-1.0, beta_w=0.5, sigma=1.0, seed=123)

    pjt = np.zeros((3, 4), dtype=float)
    wjt = np.zeros((3, 4), dtype=float)
    Ejt = np.zeros((3, 4), dtype=float)

    with pytest.raises(ValueError):
        model.utilities(pjt=pjt[0], wjt=wjt, Ejt=Ejt)  # 1D pjt
    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt[..., None], Ejt=Ejt)  # 3D wjt
    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt[..., None])  # 3D Ejt


def test_utilities_rejects_mismatched_shapes():
    model = BasicLuChoiceModel(N=10, beta_p=-1.0, beta_w=0.5, sigma=1.0, seed=123)

    pjt = np.zeros((3, 4), dtype=float)
    wjt = np.zeros((3, 5), dtype=float)
    Ejt = np.zeros((3, 4), dtype=float)

    with pytest.raises(ValueError):
        model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)


def test_utilities_output_shape():
    T, J, N_sim = _make_problem(T=4, J=6, N_sim=17)
    model = BasicLuChoiceModel(N=N_sim, beta_p=-1.0, beta_w=0.5, sigma=1.0, seed=123)

    pjt = np.zeros((T, J), dtype=float)
    wjt = np.ones((T, J), dtype=float)
    Ejt = np.zeros((T, J), dtype=float)

    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)
    assert uijt.shape == (T, N_sim, J)
    _assert_all_finite(uijt)


def test_utilities_sigma_zero_constant_across_consumers():
    T, J, N_sim = _make_problem(T=4, J=6, N_sim=25)
    model = BasicLuChoiceModel(N=N_sim, beta_p=-1.0, beta_w=0.5, sigma=0.0, seed=123)

    pjt = np.random.normal(size=(T, J))
    wjt = np.random.uniform(1.0, 2.0, size=(T, J))
    Ejt = np.random.normal(size=(T, J))

    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    base = uijt[:, 0, :]
    for i in range(1, N_sim):
        assert np.allclose(uijt[:, i, :], base, atol=0.0, rtol=0.0)


def test_utilities_reduces_when_beta_w_zero_and_Ejt_zero():
    T, J, N_sim = _make_problem(T=3, J=5, N_sim=12)
    model = BasicLuChoiceModel(N=N_sim, beta_p=-2.0, beta_w=0.0, sigma=1.3, seed=123)

    pjt = np.random.normal(size=(T, J))
    wjt = np.random.uniform(1.0, 2.0, size=(T, J))
    Ejt = np.zeros((T, J), dtype=float)

    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    expected = np.zeros_like(uijt)
    for t in range(T):
        expected[t] = model.beta_p_i[:, None] * pjt[t][None, :]

    assert np.allclose(uijt, expected)


# -----------------------------------------------------------------------------
# _generate_market_shares
# -----------------------------------------------------------------------------
def test_generate_market_shares_rejects_non_3d_uijt():
    with pytest.raises(ValueError):
        _generate_market_shares(np.zeros((3, 4), dtype=float))


def test_generate_market_shares_simplex_and_bounds():
    T, J, N_sim = _make_problem(T=4, J=7, N_sim=50)
    uijt = np.random.normal(size=(T, N_sim, J))

    sjt, s0t = _generate_market_shares(uijt)
    assert sjt.shape == (T, J)
    assert s0t.shape == (T,)
    _assert_prob_simplex(sjt, s0t, atol=1e-12)


def test_generate_market_shares_extreme_negative_utilities_outside_near_one():
    T, J, N_sim = _make_problem(T=3, J=5, N_sim=20)
    uijt = -50.0 * np.ones((T, N_sim, J), dtype=float)

    sjt, s0t = _generate_market_shares(uijt)
    _assert_prob_simplex(sjt, s0t, atol=1e-12)

    assert np.all(s0t > 1.0 - 1e-10)
    assert np.all(sjt < 1e-10)


def test_generate_market_shares_extreme_positive_utilities_outside_near_zero():
    T, J, N_sim = _make_problem(T=3, J=5, N_sim=20)
    uijt = 50.0 * np.ones((T, N_sim, J), dtype=float)

    sjt, s0t = _generate_market_shares(uijt)
    _assert_prob_simplex(sjt, s0t, atol=1e-12)

    assert np.all(s0t < 1e-10)
    assert np.all(np.abs(sjt.sum(axis=1) - 1.0) < 1e-10)


def test_generate_market_shares_monotone_under_additive_shift():
    T, J, N_sim = _make_problem(T=4, J=6, N_sim=100)
    uijt = np.random.normal(size=(T, N_sim, J))

    sjt0, s0t0 = _generate_market_shares(uijt)
    sjt1, s0t1 = _generate_market_shares(uijt + 1.0)

    assert np.all(s0t1 < s0t0)
    assert np.all(sjt1.sum(axis=1) > sjt0.sum(axis=1))

    _assert_prob_simplex(sjt0, s0t0, atol=1e-12)
    _assert_prob_simplex(sjt1, s0t1, atol=1e-12)


# -----------------------------------------------------------------------------
# generate_market
# -----------------------------------------------------------------------------
def test_generate_market_rejects_non_3d_uijt():
    with pytest.raises(ValueError):
        generate_market(uijt=np.zeros((3, 4), dtype=float), N=100, seed=123)


def test_generate_market_count_identity_and_shapes():
    T, J, N_sim = _make_problem(T=5, J=8, N_sim=500)
    N = 2000

    uijt = np.random.normal(size=(T, N_sim, J))

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

    _assert_prob_simplex(sjt, s0t, atol=1e-12)


def test_generate_market_q_over_N_approximates_sjt_large_N():
    """
    For large N, multinomial counts qjt should concentrate around expected shares sjt.

    This is intentionally loose and averages across all T*J cells to reduce variance.
    """
    T, J, N_sim = _make_problem(T=6, J=10, N_sim=2000)
    N = 20000

    wjt, Ejt, ujt, alpha = generate_market_conditions(T=T, J=J, dgp_type=4, seed=123)
    pjt = alpha + 0.3 * wjt + ujt

    model = BasicLuChoiceModel(N=N_sim, beta_p=-1.0, beta_w=0.5, sigma=1.5, seed=123)
    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=123)

    mae_inside = _mean_abs_error(qjt / float(N), sjt)
    mae_outside = _mean_abs_error(q0t / float(N), s0t)

    assert mae_inside < 0.02, f"MAE(qjt/N, sjt) too large: {mae_inside:.4f}"
    assert mae_outside < 0.02, f"MAE(q0t/N, s0t) too large: {mae_outside:.4f}"
