import numpy as np
import pytest

from datasets.dgp import (
    generate_market_conditions,
    BasicLuChoiceModel,
    generate_market,
)


def run_pipeline(T, J, N, dgp_type, seed, beta_p=-1.0, beta_w=0.5, sigma=1.5):
    """
    Mirror simulation_run.py (Lu Section 4 DGP):

      1) generate_market_conditions(T,J,dgp_type,seed)
      2) pjt = alpha + 0.3*wjt + ujt
      3) BasicLuChoiceModel(..., seed) draws heterogeneous beta_p_i
      4) uijt = model.utilities(...)
      5) (sjt, s0t, qjt, q0t) = generate_market(uijt, N=N, seed=seed)
    """
    wjt, E_bar_t, njt, Ejt, ujt, alpha = generate_market_conditions(
        T=T, J=J, dgp_type=dgp_type, seed=seed
    )
    pjt = alpha + 0.3 * wjt + ujt

    model = BasicLuChoiceModel(
        N=N, beta_p=beta_p, beta_w=beta_w, sigma=sigma, seed=seed
    )
    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=seed)

    return {
        "wjt": wjt,
        "E_bar_t": E_bar_t,
        "njt": njt,
        "Ejt": Ejt,
        "ujt": ujt,
        "alpha": alpha,
        "pjt": pjt,
        "uijt": uijt,
        "sjt": sjt,
        "s0t": s0t,
        "qjt": qjt,
        "q0t": q0t,
        "N": int(N),
    }


# ---------------------------------------------------------------------
# 1) Basic shape + validity
# ---------------------------------------------------------------------
@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_shapes(dgp_type):
    T, J, N = 25, 15, 200
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    assert out["wjt"].shape == (T, J)
    assert out["E_bar_t"].shape == (T,)
    assert out["njt"].shape == (T, J)
    assert out["Ejt"].shape == (T, J)
    assert out["ujt"].shape == (T, J)
    assert out["alpha"].shape == (T, J)
    assert out["pjt"].shape == (T, J)
    assert out["uijt"].shape == (T, N, J)
    assert out["sjt"].shape == (T, J)
    assert out["s0t"].shape == (T,)
    assert out["qjt"].shape == (T, J)
    assert out["q0t"].shape == (T,)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_share_validity_and_identity(dgp_type):
    T, J, N = 25, 15, 200
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    sjt = out["sjt"]
    s0t = out["s0t"]

    assert np.all(np.isfinite(sjt))
    assert np.all(np.isfinite(s0t))

    # Shares must be in [0,1]
    assert np.all(sjt >= 0.0)
    assert np.all(sjt <= 1.0)
    assert np.all(s0t >= 0.0)
    assert np.all(s0t <= 1.0)

    # Share identity: s0t + sum_j sjt == 1
    share_err = np.max(np.abs(s0t + sjt.sum(axis=1) - 1.0))
    assert share_err < 1e-12, f"Share identity violated: max error={share_err:.3e}"


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_count_validity_and_identity(dgp_type):
    T, J, N = 25, 15, 200
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    qjt = out["qjt"]
    q0t = out["q0t"]

    assert np.issubdtype(qjt.dtype, np.integer)
    assert np.issubdtype(q0t.dtype, np.integer)

    assert np.all(qjt >= 0)
    assert np.all(q0t >= 0)

    # Multinomial accounting per market: q0t + sum_j qjt == N
    totals = q0t + qjt.sum(axis=1)
    assert np.all(
        totals == N
    ), f"Count identity violated: min={totals.min()}, max={totals.max()}, N={N}"


# ---------------------------------------------------------------------
# 2) Sparse vs dense njt structure (matches dgp.py exactly)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("dgp_type", [1, 2])
def test_sparse_njt_exact_pattern(dgp_type):
    """
    For DGP 1/2, njt is deterministic:
      - n_active = int(0.4*J)
      - for j < n_active: njt[t,j] = +1 if j even else -1
      - for j >= n_active: njt[t,j] = 0
    """
    T, J, N = 10, 15, 50
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)
    njt = out["njt"]

    n_active = int(0.4 * J)
    for t in range(T):
        expected_active = np.array(
            [1.0 if j % 2 == 0 else -1.0 for j in range(n_active)]
        )
        assert np.array_equal(njt[t, :n_active], expected_active)
        assert np.array_equal(njt[t, n_active:], np.zeros(J - n_active))


@pytest.mark.parametrize("dgp_type", [3, 4])
def test_dense_njt_not_sparse(dgp_type):
    """
    For DGP 3/4, njt is drawn from Normal(0, 1/3), so it should not contain many exact zeros.
    We test that the fraction of (near-)zeros is small.
    """
    T, J, N = 25, 25, 50
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)
    njt = out["njt"]

    near_zero = np.mean(np.abs(njt) < 1e-12)
    assert (
        near_zero < 1e-3
    ), f"Unexpected near-zero mass for dense njt: frac={near_zero:.3e}"


# ---------------------------------------------------------------------
# 3) Endogeneity on/off signal in price equation
# ---------------------------------------------------------------------
def _corr_flat(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return np.corrcoef(a, b)[0, 1]


def test_endogeneity_signal_relative_strength():
    """
    In dgp.py:
      - DGP 1/3: alpha == 0 => pjt = 0.3*wjt + ujt (no designed link to Ejt)
      - DGP 2/4: alpha depends on njt, and Ejt = -1 + njt => designed correlation between pjt and Ejt

    Use a relative comparison (endogenous corr > exogenous corr + margin) to reduce flakiness.
    """
    T, J, N = 50, 30, 100
    seed = 123

    out1 = run_pipeline(T=T, J=J, N=N, dgp_type=1, seed=seed)
    out2 = run_pipeline(T=T, J=J, N=N, dgp_type=2, seed=seed)
    out3 = run_pipeline(T=T, J=J, N=N, dgp_type=3, seed=seed)
    out4 = run_pipeline(T=T, J=J, N=N, dgp_type=4, seed=seed)

    c1 = _corr_flat(out1["pjt"], out1["Ejt"])
    c2 = _corr_flat(out2["pjt"], out2["Ejt"])
    c3 = _corr_flat(out3["pjt"], out3["Ejt"])
    c4 = _corr_flat(out4["pjt"], out4["Ejt"])

    assert (
        c2 > c1 + 0.05
    ), f"DGP2 endogeneity not stronger than DGP1: corr2={c2:.3f}, corr1={c1:.3f}"
    assert (
        c4 > c3 + 0.05
    ), f"DGP4 endogeneity not stronger than DGP3: corr4={c4:.3f}, corr3={c3:.3f}"


# ---------------------------------------------------------------------
# 5) Utility/share monotonicity sanity checks
# ---------------------------------------------------------------------
def test_beta_p_more_negative_reduces_inside_shares():
    """
    Holding (wjt, Ejt, alpha, ujt) fixed via seed,
    making beta_p more negative should reduce mean inside shares.
    """
    T, J, N = 25, 15, 400
    seed = 123
    dgp_type = 3

    out_lo = run_pipeline(
        T=T, J=J, N=N, dgp_type=dgp_type, seed=seed, beta_p=-0.5, sigma=1.5
    )
    out_hi = run_pipeline(
        T=T, J=J, N=N, dgp_type=dgp_type, seed=seed, beta_p=-2.0, sigma=1.5
    )

    mean_inside_lo = out_lo["sjt"].sum(axis=1).mean()
    mean_inside_hi = out_hi["sjt"].sum(axis=1).mean()

    assert mean_inside_hi < mean_inside_lo, (
        f"Expected more negative beta_p to reduce inside shares. "
        f"Got mean_inside(beta_p=-2.0)={mean_inside_hi:.6f}, "
        f"mean_inside(beta_p=-0.5)={mean_inside_lo:.6f}"
    )


def test_sigma_increases_heterogeneity_in_shares():
    """
    With sigma=0, all consumers share the same beta_p, so within-market shares are less dispersed.
    With sigma>0, heterogeneity in beta_p increases dispersion in predicted shares.
    """
    T, J, N = 25, 15, 800
    seed = 123
    dgp_type = 3

    out0 = run_pipeline(
        T=T, J=J, N=N, dgp_type=dgp_type, seed=seed, beta_p=-1.0, sigma=0.0
    )
    out1 = run_pipeline(
        T=T, J=J, N=N, dgp_type=dgp_type, seed=seed, beta_p=-1.0, sigma=1.5
    )

    disp0 = out0["sjt"].std(axis=1).mean()
    disp1 = out1["sjt"].std(axis=1).mean()

    assert (
        disp1 > disp0
    ), f"Expected higher dispersion with sigma>0. Got disp1={disp1:.6f}, disp0={disp0:.6f}"


# ---------------------------------------------------------------------
# 6) Direct unit tests for generate_market (replaces generate_market_shares tests)
# ---------------------------------------------------------------------
def test_generate_market_rejects_bad_shape():
    with pytest.raises(ValueError):
        generate_market(np.zeros((2, 3)), N=100, seed=0)  # not (T,N,J)


def test_generate_market_extreme_negative_utilities_outside_dominates():
    # Very negative inside utilities => inside probs ~ 0 => s0 ~ 1
    T, N_sim, J = 3, 50, 10
    N = 200
    uijt = -1000.0 * np.ones((T, N_sim, J))
    sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=0)

    assert np.all(sjt >= 0.0)
    assert np.all(s0t <= 1.0)
    assert np.all(
        s0t > 1.0 - 1e-10
    ), f"Expected outside share ~1, got min(s0t)={s0t.min():.12f}"

    # Underflow makes inside probs exactly 0 here, so multinomial draw is deterministic.
    assert np.all(qjt == 0)
    assert np.all(q0t == N)


def test_generate_market_extreme_positive_utilities_inside_dominates():
    # Very positive inside utilities => outside prob ~ 0 => s0 ~ 0
    T, N_sim, J = 3, 50, 10
    N = 200
    uijt = 1000.0 * np.ones((T, N_sim, J))
    sjt, s0t, qjt, q0t = generate_market(uijt=uijt, N=N, seed=0)

    assert np.all(s0t >= 0.0)
    assert np.all(
        s0t < 1e-10
    ), f"Expected outside share ~0, got max(s0t)={s0t.max():.12f}"

    # Underflow makes outside prob exactly 0 here, so multinomial draw is deterministic.
    assert np.all(q0t == 0)
    assert np.all(qjt.sum(axis=1) == N)


def test_generate_market_additive_shift_increases_inside_share():
    # Adding a constant to all inside utilities should increase inside shares (reduce outside).
    T, N_sim, J = 3, 200, 10
    N = 500
    rng = np.random.default_rng(0)
    base = rng.normal(size=(T, N_sim, J))

    sjt0, s0t0, _, _ = generate_market(uijt=base, N=N, seed=0)
    sjt1, s0t1, _, _ = generate_market(uijt=base + 1.0, N=N, seed=0)

    inside0 = sjt0.sum(axis=1)
    inside1 = sjt1.sum(axis=1)

    assert np.all(
        inside1 > inside0
    ), "Expected inside share to increase after positive utility shift."
    assert np.all(
        s0t1 < s0t0
    ), "Expected outside share to decrease after positive utility shift."
