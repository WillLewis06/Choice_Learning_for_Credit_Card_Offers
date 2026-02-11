"""
Pytests for BLP-style market inversion (single market).

Contract under test
-------------------
Given:
  - observed inside shares s_obs (length J, strictly positive),
  - outside share s0 in (0, 1),
  - prices p (length J),
  - heterogeneity scale sigma >= 0,
  - fixed simulation draws v_draws (length N),

`invert_market` should return a mean-utility vector `delta_hat` such that, when the
same simulator and the same draws are used,
  simulate_shares(delta_hat, p, sigma, v_draws) ≈ s_obs.

Determinism note:
- `v_draws` must be held fixed between "target share generation" and "evaluation"
  to make inversion a well-defined fixed point test (otherwise Monte Carlo noise
  changes between calls).
"""

from __future__ import annotations

import numpy as np
import pytest

from lu_conftest import assert_finite_np, fixed_draws, make_feasible_shares
from lu.blp.inversion import check_market_inputs, invert_market, simulate_shares

# -----------------------------------------------------------------------------
# Named tolerances (explain once; use everywhere)
# -----------------------------------------------------------------------------
# sigma = 0 reduces to a simple logit mapping; numerically this is near-exact.
ATOL_SIGMA0 = 1e-10

# sigma > 0 introduces simulation error; inversion solves a simulated fixed point.
ATOL_SIGMA_POS = 1e-6

# "tiny" share vectors are near the boundary of the simplex and can be more sensitive.
ATOL_TINY = 5e-6
ATOL_NONTINY = 2e-6


# -----------------------------------------------------------------------------
# Local helpers to reduce duplication and clarify intent
# -----------------------------------------------------------------------------
def assert_valid_inside_shares(inside_shares: np.ndarray, *, J: int, name: str) -> None:
    """
    Assert that a length-J inside-share vector is a valid probability vector for
    the inside goods under an outside-option model (outside share is residual).

    This helper is intentionally strict because invalid shares should be rejected
    early by `check_market_inputs` or indicate a bug in `simulate_shares`.
    """
    assert isinstance(inside_shares, np.ndarray)
    assert inside_shares.ndim == 1
    assert inside_shares.shape == (J,)

    assert_finite_np(inside_shares, name=name)
    assert np.all(inside_shares > 0.0)
    assert np.all(inside_shares < 1.0)

    outside_share = 1.0 - float(inside_shares.sum())
    assert outside_share > 0.0
    assert np.isclose(
        float(inside_shares.sum()) + outside_share, 1.0, atol=1e-10, rtol=0.0
    )


def invert_then_simulate(
    *,
    s_obs: np.ndarray,
    s0: float,
    prices: np.ndarray,
    sigma: float,
    v_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Convenience wrapper used by multiple tests:
      1) invert to get delta_hat
      2) simulate shares at delta_hat using the same inputs

    Returns (delta_hat, s_hat, iters).
    """
    # Act: invert
    delta_hat, iters = invert_market(
        s_obs=s_obs,
        s0=s0,
        p=prices,
        sigma=sigma,
        v_draws=v_draws,
    )
    assert iters > 0
    assert_finite_np(delta_hat, name="delta_hat")

    # Act: re-simulate at the implied fixed point
    s_hat = simulate_shares(delta_hat, prices, sigma, v_draws)
    return delta_hat, s_hat, iters


# -----------------------------------------------------------------------------
# check_market_inputs
# -----------------------------------------------------------------------------
def test_check_market_inputs_accepts_valid_market():
    """
    Smoke test: a well-formed market passes validation.

    This protects downstream inversion/simulation code from:
      - infeasible shares (non-positive inside share, invalid outside share),
      - shares not summing to one,
      - shape mismatches between shares and prices.
    """
    # Arrange
    inside_shares_target = np.array([0.2, 0.1, 0.15], dtype=float)
    outside_share_target = 0.55
    prices = np.array([1.0, 1.5, 2.0], dtype=float)

    # Act / Assert (no exception)
    check_market_inputs(inside_shares_target, outside_share_target, prices)


@pytest.mark.parametrize(
    "s_obs,s0,prices",
    [
        # non-positive inside shares
        (
            np.array([0.2, 0.0, 0.1], dtype=float),
            0.7,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        (
            np.array([-0.2, 0.3, 0.1], dtype=float),
            0.8,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        # non-positive or >= 1 outside share
        (
            np.array([0.2, 0.1, 0.15], dtype=float),
            0.0,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        (
            np.array([0.2, 0.1, 0.15], dtype=float),
            1.0,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        # shares do not sum to 1
        (
            np.array([0.2, 0.1, 0.15], dtype=float),
            0.60,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        # shape mismatch: J differs between shares and prices
        (
            np.array([0.2, 0.1], dtype=float),
            0.70,
            np.array([1.0, 1.5, 2.0], dtype=float),
        ),
        (
            np.array([0.2, 0.1, 0.15], dtype=float),
            0.55,
            np.array([1.0, 2.0], dtype=float),
        ),
    ],
    ids=[
        "inside_share_has_zero",
        "inside_share_has_negative",
        "outside_share_is_zero",
        "outside_share_is_one",
        "shares_do_not_sum_to_one",
        "shape_mismatch_shares_shorter",
        "shape_mismatch_prices_shorter",
    ],
)
def test_check_market_inputs_rejects_invalid_inputs(
    s_obs: np.ndarray, s0: float, prices: np.ndarray
):
    """
    Validation rejects infeasible shares and mismatched shapes.

    Parametrized so failures report the specific invalid case via pytest ids.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        check_market_inputs(s_obs, s0, prices)


# -----------------------------------------------------------------------------
# simulate_shares
# -----------------------------------------------------------------------------
def test_simulate_shares_returns_valid_probabilities():
    """
    The share simulator returns a valid length-J inside-share vector.

    The simulator returns inside shares only; the outside share is defined as:
      s0 = 1 - sum_j s_hat[j]
    and should be strictly positive.
    """
    # Arrange
    delta = np.array([0.5, -0.2, 0.1], dtype=float)
    prices = np.array([1.0, 1.5, 2.0], dtype=float)
    sigma = 0.8
    v_draws = fixed_draws(n=500, seed=123)

    # Act
    inside_shares_sim = simulate_shares(delta, prices, sigma, v_draws)

    # Assert
    assert_valid_inside_shares(
        inside_shares_sim, J=delta.shape[0], name="inside_shares_sim"
    )


# -----------------------------------------------------------------------------
# invert_market
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sigma,delta_true,prices,n_draws,atol",
    [
        (
            0.0,
            np.array([0.6, -0.4, 0.2], dtype=float),
            np.array([1.0, 1.5, 2.0], dtype=float),
            800,
            ATOL_SIGMA0,
        ),
        (
            1.0,
            np.array([0.8, -0.3, 0.2], dtype=float),
            np.array([1.0, 1.3, 1.8], dtype=float),
            2000,
            ATOL_SIGMA_POS,
        ),
    ],
    ids=["sigma0_closed_form_like", "sigma_pos_simulated"],
)
def test_invert_market_roundtrip_for_sigma_zero_and_nonzero(
    sigma: float, delta_true: np.ndarray, prices: np.ndarray, n_draws: int, atol: float
):
    """
    Round-trip test (model-consistent target shares).

    Arrange:
      - generate target shares using the same simulator and fixed draws
    Act:
      - invert the market
      - re-simulate using the implied delta_hat and the same draws
    Assert:
      - simulated shares match the target shares within tolerance

    Determinism requires using the same v_draws in both generation and evaluation.
    """
    # Arrange
    v_draws = fixed_draws(n=n_draws, seed=123)
    inside_shares_target = simulate_shares(delta_true, prices, sigma, v_draws)
    outside_share_target = 1.0 - float(inside_shares_target.sum())

    # Act
    _, inside_shares_sim, _ = invert_then_simulate(
        s_obs=inside_shares_target,
        s0=outside_share_target,
        prices=prices,
        sigma=sigma,
        v_draws=v_draws,
    )

    # Assert
    assert np.allclose(inside_shares_sim, inside_shares_target, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "sigma", [0.0, 0.5, 1.5], ids=["sigma0", "sigma0p5", "sigma1p5"]
)
@pytest.mark.parametrize("tiny", [False, True], ids=["regular", "tiny"])
def test_invert_market_fixed_point_residual_is_small(sigma: float, tiny: bool):
    """
    Fixed-point test (arbitrary feasible target shares).

    Here the target shares are not generated by the model; we only require that:
      - (s_obs, s0) are feasible and pass `check_market_inputs`
      - inversion produces delta_hat whose simulated shares match s_obs

    The "tiny" case stresses numerical stability near the simplex boundary.
    """
    # Arrange
    J = 5
    prices = np.linspace(1.0, 2.0, J, dtype=float)
    v_draws = fixed_draws(n=3000, seed=123)

    inside_shares_target, outside_share_target = make_feasible_shares(
        J, tiny=tiny, seed=11
    )
    check_market_inputs(inside_shares_target, outside_share_target, prices)

    # Act
    _, inside_shares_sim, _ = invert_then_simulate(
        s_obs=inside_shares_target,
        s0=outside_share_target,
        prices=prices,
        sigma=sigma,
        v_draws=v_draws,
    )

    # Assert
    residual = float(np.max(np.abs(inside_shares_sim - inside_shares_target)))
    if sigma == 0.0:
        tol = ATOL_SIGMA0
    else:
        tol = ATOL_TINY if tiny else ATOL_NONTINY
    assert residual <= tol


def test_invert_market_sigma0_matches_closed_form_delta():
    """
    Analytic anchor: with sigma = 0, the model reduces to simple logit with outside option.

    For any feasible (s_obs, s0), the implied mean utilities satisfy:
      delta_j = log(s_obs[j]) - log(s0)

    This test ensures the inversion implementation is consistent with the closed form.
    """
    # Arrange
    J = 6
    prices = np.linspace(0.8, 2.2, J, dtype=float)
    sigma = 0.0

    # Note: prices and v_draws are part of the API even though sigma=0 does not need
    # simulation draws; the path may ignore them but the interface stays uniform.
    v_draws = fixed_draws(n=1000, seed=123)

    inside_shares_target, outside_share_target = make_feasible_shares(
        J, tiny=False, seed=7
    )
    check_market_inputs(inside_shares_target, outside_share_target, prices)

    # Act
    delta_hat, iters = invert_market(
        s_obs=inside_shares_target,
        s0=outside_share_target,
        p=prices,
        sigma=sigma,
        v_draws=v_draws,
    )

    # Assert
    assert iters > 0
    assert_finite_np(delta_hat, name="delta_hat")

    delta_closed_form = np.log(inside_shares_target) - np.log(outside_share_target)
    assert np.allclose(delta_hat, delta_closed_form, atol=ATOL_SIGMA0, rtol=0.0)
