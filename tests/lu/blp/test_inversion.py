"""
Pytests for BLP-style market inversion (single market and market panel).

Contract under test
-------------------
Given:
  - observed inside shares s_obs (length J, strictly positive),
  - prices p (length J),
  - heterogeneity scale sigma >= 0,
  - fixed simulation draws v_draws (length R),
  - an explicit starting value delta_init (length J),
  - explicit numerical controls (damping, tol, share_tol, max_iter),

`invert_market` returns (delta_hat, iterations) such that, when the same simulator
and the same draws are used:
  simulate_shares(delta_hat, p, sigma, v_draws) ≈ s_obs.

Notes:
- The outside share is implicit: s0 = 1 - sum_j s_obs[j]. It is used only to
  construct a standard logit warm-start delta_init via `logit_delta_init`.
- Determinism requires holding v_draws fixed between "target share generation"
  and "evaluation" (otherwise Monte Carlo noise changes between calls).
"""

from __future__ import annotations

import numpy as np
import pytest

from lu_conftest import assert_finite_np, fixed_draws, make_feasible_shares
from lu.blp.inversion import (
    invert_all_markets,
    invert_market,
    logit_delta_init,
    simulate_shares,
)

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
# Local helpers
# -----------------------------------------------------------------------------
def assert_valid_inside_shares(inside_shares: np.ndarray, J: int, name: str) -> None:
    """
    Assert that a length-J inside-share vector is a valid probability vector for
    the inside goods under an outside-option model (outside share is residual).
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


def choose_controls(sigma: float, tiny: bool) -> tuple[float, float, float, int]:
    """
    Choose explicit contraction controls that are stable and consistent with the
    assertions in this test module.
    """
    damping = 0.7
    max_iter = 5000

    if sigma == 0.0:
        tol = 1e-12
        share_tol = 1e-12
        return damping, tol, share_tol, max_iter

    # For sigma > 0 we stop in share-space at a level safely below the assertion
    # tolerances, but not so strict that the contraction wastes iterations.
    tol = 1e-10
    share_tol = 1e-9 if tiny else 5e-10
    return damping, tol, share_tol, max_iter


def invert_then_simulate(
    s_obs: np.ndarray,
    s0: float,
    prices: np.ndarray,
    sigma: float,
    v_draws: np.ndarray,
    damping: float,
    tol: float,
    share_tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Convenience wrapper used by multiple tests:
      1) construct delta_init via logit_delta_init(s_obs, s0)
      2) invert to get delta_hat
      3) simulate shares at delta_hat using the same inputs

    Returns (delta_hat, s_hat, iters).
    """
    delta_init = logit_delta_init(s_obs, s0)

    delta_hat, iters = invert_market(
        s_obs=s_obs,
        p=prices,
        sigma=sigma,
        v_draws=v_draws,
        delta_init=delta_init,
        damping=damping,
        tol=tol,
        share_tol=share_tol,
        max_iter=max_iter,
    )
    assert iters > 0
    assert_finite_np(delta_hat, name="delta_hat")

    s_hat = simulate_shares(delta_hat, prices, sigma, v_draws)
    return delta_hat, s_hat, iters


# -----------------------------------------------------------------------------
# logit_delta_init
# -----------------------------------------------------------------------------
def test_logit_delta_init_matches_closed_form():
    """
    logit_delta_init returns log(s_obs) - log(s0) and rejects non-positive shares.
    """
    s_obs = np.array([0.2, 0.1, 0.15], dtype=float)
    s0 = 0.55

    delta_init = logit_delta_init(s_obs, s0)
    assert_finite_np(delta_init, name="delta_init")

    expected = np.log(s_obs) - np.log(s0)
    assert np.allclose(delta_init, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "s_obs,s0",
    [
        (np.array([0.2, 0.0, 0.1], dtype=float), 0.7),
        (np.array([-0.2, 0.3, 0.1], dtype=float), 0.8),
        (np.array([0.2, 0.1, 0.15], dtype=float), 0.0),
        (np.array([0.2, 0.1, 0.15], dtype=float), -0.1),
    ],
    ids=[
        "s_obs_has_zero",
        "s_obs_has_negative",
        "s0_is_zero",
        "s0_is_negative",
    ],
)
def test_logit_delta_init_rejects_nonpositive_shares(s_obs: np.ndarray, s0: float):
    with pytest.raises(ValueError):
        logit_delta_init(s_obs, s0)


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
    delta = np.array([0.5, -0.2, 0.1], dtype=float)
    prices = np.array([1.0, 1.5, 2.0], dtype=float)
    sigma = 0.8
    v_draws = fixed_draws(n=500, seed=123)

    inside_shares_sim = simulate_shares(delta, prices, sigma, v_draws)
    assert_valid_inside_shares(
        inside_shares_sim, J=delta.shape[0], name="inside_shares_sim"
    )


# -----------------------------------------------------------------------------
# invert_market: input rejection
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "s_obs",
    [
        np.array([0.2, 0.0, 0.1], dtype=float),
        np.array([-0.2, 0.3, 0.1], dtype=float),
    ],
    ids=["has_zero", "has_negative"],
)
def test_invert_market_rejects_nonpositive_observed_shares(s_obs: np.ndarray):
    J = s_obs.shape[0]
    prices = np.linspace(1.0, 2.0, J, dtype=float)
    sigma = 0.5
    v_draws = fixed_draws(n=200, seed=123)

    delta_init = np.zeros(J, dtype=float)
    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    with pytest.raises(ValueError):
        invert_market(
            s_obs=s_obs,
            p=prices,
            sigma=sigma,
            v_draws=v_draws,
            delta_init=delta_init,
            damping=damping,
            tol=tol,
            share_tol=share_tol,
            max_iter=max_iter,
        )


def test_invert_market_rejects_shape_mismatch():
    s_obs = np.array([0.2, 0.1, 0.15], dtype=float)
    prices = np.array([1.0, 1.5], dtype=float)  # wrong length
    sigma = 0.5
    v_draws = fixed_draws(n=200, seed=123)

    delta_init = np.zeros(3, dtype=float)
    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    with pytest.raises(ValueError):
        invert_market(
            s_obs=s_obs,
            p=prices,
            sigma=sigma,
            v_draws=v_draws,
            delta_init=delta_init,
            damping=damping,
            tol=tol,
            share_tol=share_tol,
            max_iter=max_iter,
        )


# -----------------------------------------------------------------------------
# invert_market: fixed point properties
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
      - invert the market (using a logit warm-start)
      - re-simulate using the implied delta_hat and the same draws
    Assert:
      - simulated shares match the target shares within tolerance
    """
    v_draws = fixed_draws(n=n_draws, seed=123)
    inside_shares_target = simulate_shares(delta_true, prices, sigma, v_draws)
    outside_share_target = 1.0 - float(inside_shares_target.sum())

    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    _, inside_shares_sim, _ = invert_then_simulate(
        inside_shares_target,
        outside_share_target,
        prices,
        sigma,
        v_draws,
        damping,
        tol,
        share_tol,
        max_iter,
    )

    assert np.allclose(inside_shares_sim, inside_shares_target, atol=atol, rtol=0.0)


@pytest.mark.parametrize(
    "sigma", [0.0, 0.5, 1.5], ids=["sigma0", "sigma0p5", "sigma1p5"]
)
@pytest.mark.parametrize("tiny", [False, True], ids=["regular", "tiny"])
def test_invert_market_fixed_point_residual_is_small(sigma: float, tiny: bool):
    """
    Fixed-point test (arbitrary feasible target shares).

    Here the target shares are not generated by the model; we only require that
    inversion produces delta_hat whose simulated shares match s_obs.
    """
    J = 5
    prices = np.linspace(1.0, 2.0, J, dtype=float)
    v_draws = fixed_draws(n=3000, seed=123)

    inside_shares_target, outside_share_target = make_feasible_shares(
        J, tiny=tiny, seed=11
    )

    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=tiny)

    _, inside_shares_sim, _ = invert_then_simulate(
        inside_shares_target,
        outside_share_target,
        prices,
        sigma,
        v_draws,
        damping,
        tol,
        share_tol,
        max_iter,
    )

    residual = float(np.max(np.abs(inside_shares_sim - inside_shares_target)))
    if sigma == 0.0:
        assert residual <= ATOL_SIGMA0
    else:
        assert residual <= (ATOL_TINY if tiny else ATOL_NONTINY)


def test_invert_market_sigma0_matches_closed_form_delta():
    """
    Analytic anchor: with sigma = 0, the model reduces to simple logit with outside option.

    For any feasible (s_obs, s0), the implied mean utilities satisfy:
      delta_j = log(s_obs[j]) - log(s0)
    """
    J = 6
    prices = np.linspace(0.8, 2.2, J, dtype=float)
    sigma = 0.0
    v_draws = fixed_draws(n=1000, seed=123)

    inside_shares_target, outside_share_target = make_feasible_shares(
        J, tiny=False, seed=7
    )

    delta_init = logit_delta_init(inside_shares_target, outside_share_target)
    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    delta_hat, iters = invert_market(
        s_obs=inside_shares_target,
        p=prices,
        sigma=sigma,
        v_draws=v_draws,
        delta_init=delta_init,
        damping=damping,
        tol=tol,
        share_tol=share_tol,
        max_iter=max_iter,
    )

    assert iters > 0
    assert_finite_np(delta_hat, name="delta_hat")

    delta_closed_form = np.log(inside_shares_target) - np.log(outside_share_target)
    assert np.allclose(delta_hat, delta_closed_form, atol=ATOL_SIGMA0, rtol=0.0)


# -----------------------------------------------------------------------------
# invert_all_markets
# -----------------------------------------------------------------------------
def test_invert_all_markets_rejects_shape_mismatch():
    sjt = np.ones((3, 4), dtype=float)
    pjt = np.ones((3, 4), dtype=float)
    delta_init = np.ones((3, 5), dtype=float)  # wrong shape

    sigma = 0.5
    v_draws = fixed_draws(n=100, seed=123)
    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    with pytest.raises(ValueError):
        invert_all_markets(
            sjt=sjt,
            pjt=pjt,
            sigma=sigma,
            v_draws=v_draws,
            delta_init=delta_init,
            damping=damping,
            tol=tol,
            share_tol=share_tol,
            max_iter=max_iter,
        )


def test_invert_all_markets_roundtrip_panel():
    """
    Panel round-trip: generate (sjt) from known deltas using a fixed v_draws,
    invert all markets, then verify simulated shares match sjt market-by-market.
    """
    T = 4
    J = 3
    sigma = 0.9

    rng = np.random.default_rng(123)
    delta_true = rng.normal(loc=0.0, scale=0.5, size=(T, J)).astype(float)
    pjt = np.tile(np.array([1.0, 1.3, 1.8], dtype=float), (T, 1))

    v_draws = fixed_draws(n=2500, seed=123)

    sjt = np.empty((T, J), dtype=float)
    delta_init = np.empty((T, J), dtype=float)

    for t in range(T):
        sjt[t] = simulate_shares(delta_true[t], pjt[t], sigma, v_draws)
        s0_t = 1.0 - float(sjt[t].sum())
        delta_init[t] = logit_delta_init(sjt[t], s0_t)

    damping, tol, share_tol, max_iter = choose_controls(sigma=sigma, tiny=False)

    delta_hat = invert_all_markets(
        sjt=sjt,
        pjt=pjt,
        sigma=sigma,
        v_draws=v_draws,
        delta_init=delta_init,
        damping=damping,
        tol=tol,
        share_tol=share_tol,
        max_iter=max_iter,
    )

    assert isinstance(delta_hat, np.ndarray)
    assert delta_hat.shape == (T, J)
    assert_finite_np(delta_hat.reshape(-1), name="delta_hat_panel")

    for t in range(T):
        s_hat_t = simulate_shares(delta_hat[t], pjt[t], sigma, v_draws)
        assert np.allclose(s_hat_t, sjt[t], atol=ATOL_SIGMA_POS, rtol=0.0)
