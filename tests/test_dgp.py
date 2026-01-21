import numpy as np
import pytest

from datasets.dgp import (
    generate_market_conditions,
    BasicLuChoiceModel,
    generate_market_data,
)


def run_pipeline(T, J, N, dgp_type, seed, beta_p=-1.0, beta_w=0.5, sigma=1.5):
    """
    Mirror simulation_run.py exactly:
      1) generate_market_conditions(T,J,dgp_type,seed)
      2) pjt = alpha + 0.3*wjt + ujt
      3) BasicLuChoiceModel(..., seed)
      4) uijt = model.utilities(...)
      5) qjt = generate_market_data(uijt, seed)
    """
    wjt, E_bar_t, njt, Ejt, ujt, alpha = generate_market_conditions(
        T=T, J=J, dgp_type=dgp_type, seed=seed
    )
    pjt = alpha + 0.3 * wjt + ujt

    model = BasicLuChoiceModel(
        N=N, beta_p=beta_p, beta_w=beta_w, sigma=sigma, seed=seed
    )
    uijt = model.utilities(pjt=pjt, wjt=wjt, Ejt=Ejt)

    qjt = generate_market_data(uijt=uijt, seed=seed)

    return {
        "wjt": wjt,
        "E_bar_t": E_bar_t,
        "njt": njt,
        "Ejt": Ejt,
        "ujt": ujt,
        "alpha": alpha,
        "pjt": pjt,
        "uijt": uijt,
        "qjt": qjt,
    }


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_shapes(dgp_type):
    T, J, N = 25, 15, 1000
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    assert out["wjt"].shape == (T, J)
    assert out["E_bar_t"].shape == (T,)
    assert out["njt"].shape == (T, J)
    assert out["Ejt"].shape == (T, J)
    assert out["ujt"].shape == (T, J)
    assert out["alpha"].shape == (T, J)
    assert out["pjt"].shape == (T, J)
    assert out["uijt"].shape == (T, N, J)
    assert out["qjt"].shape == (T, J)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_sales_bounds(dgp_type):
    T, J, N = 25, 15, 1000
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    total_sales = out["qjt"].sum(axis=1)
    assert np.all(total_sales >= 0)
    assert np.all(
        total_sales <= N
    ), "Inside-good sales exceed market size; outside option handling likely wrong."


@pytest.mark.parametrize("dgp_type", [1, 2])
def test_sparse_njt_has_many_zeros(dgp_type):
    """
    For DGP 1/2, njt is exactly sparse by construction:
      - first 40% products are +/-1
      - remaining 60% are 0
    """
    T, J, N = 25, 15, 1000
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    njt = out["njt"]
    zero_share = np.mean(njt == 0.0)

    # With J=15, n_active=int(0.4*J)=6, so 9/15=0.6 zeros per market exactly.
    assert (
        zero_share > 0.5
    ), f"Expected many zeros under sparse DGP. Got zero_share={zero_share:.3f}."


@pytest.mark.parametrize(
    "dgp_type, expected",
    [
        (1, "exogenous"),
        (2, "endogenous"),
        (3, "exogenous"),
        (4, "endogenous"),
    ],
)
def test_price_endogeneity_signal(dgp_type, expected):
    """
    DGP 1/3: alpha == 0 -> pjt should be (almost) uncorrelated with Ejt.
    DGP 2/4: alpha depends on njt -> pjt should be positively correlated with Ejt.
    """
    T, J, N = 25, 15, 1000
    out = run_pipeline(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    pjt = out["pjt"]
    Ejt = out["Ejt"]
    corr = np.corrcoef(pjt.ravel(), Ejt.ravel())[0, 1]

    if expected == "exogenous":
        assert abs(corr) < 0.1, f"Unexpected endogeneity signal: corr={corr:.3f}"
    else:
        assert corr > 0.1, f"Endogeneity signal too weak: corr={corr:.3f}"


def test_reproducibility_same_seed():
    """
    With the current implementation, all randomness is driven by `seed` passed into:
      - generate_market_conditions
      - BasicLuChoiceModel
      - generate_market_data
    So running twice with the same seed should give identical outputs.
    """
    T, J, N = 25, 15, 1000

    out1 = run_pipeline(T=T, J=J, N=N, dgp_type=2, seed=123)
    out2 = run_pipeline(T=T, J=J, N=N, dgp_type=2, seed=123)

    for k in ("wjt", "njt", "Ejt", "ujt", "alpha", "pjt", "uijt", "qjt"):
        assert np.array_equal(out1[k], out2[k]), f"Mismatch for key={k} under same seed"
