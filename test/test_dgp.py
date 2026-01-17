import numpy as np
import pytest

from datasets.dgp import generate_dgp


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_shapes(dgp_type):
    T, J, N = 25, 15, 1000
    pjt, wjt, xi, qjt = generate_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    assert pjt.shape == (T, J)
    assert wjt.shape == (T, J)
    assert xi.shape == (T, J)
    assert qjt.shape == (T, J)


@pytest.mark.parametrize("dgp_type", [1, 2, 3, 4])
def test_sales_bounds(dgp_type):
    T, J, N = 25, 15, 1000
    _, _, _, qjt = generate_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    total_sales = qjt.sum(axis=1)
    assert np.all(total_sales >= 0)
    assert np.all(
        total_sales <= N
    ), "Inside-good sales exceed market size; outside option handling likely wrong."


@pytest.mark.parametrize("dgp_type", [1, 2])
def test_sparse_eta_has_many_zeros(dgp_type):
    """
    For DGP 1/2, eta_{jt} is exactly sparse by construction.

    We recover eta by removing the market mean from xi:
        eta_hat = xi - mean_j(xi)
    Because eta is exactly in {-1,0,1} with 60% zeros, we should see >50% exact zeros.
    """
    T, J, N = 25, 15, 1000
    _, _, xi, _ = generate_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    eta_hat = xi - xi.mean(axis=1, keepdims=True)
    zero_share = np.mean(eta_hat == 0.0)

    assert (
        zero_share > 0.5
    ), f"Expected many exact zeros under sparse DGP. Got zero_share={zero_share:.3f}."


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
    DGP 1/3: price is exogenous wrt xi -> corr(p, xi) should be small.
    DGP 2/4: price depends on eta (and xi) -> corr(p, xi) should be positive.
    """
    T, J, N = 25, 15, 1000
    pjt, _, xi, _ = generate_dgp(T=T, J=J, N=N, dgp_type=dgp_type, seed=123)

    corr = np.corrcoef(pjt.ravel(), xi.ravel())[0, 1]

    if expected == "exogenous":
        assert abs(corr) < 0.1, f"Unexpected endogeneity signal: corr={corr:.3f}"
    else:
        assert corr > 0.1, f"Endogeneity signal too weak: corr={corr:.3f}"
