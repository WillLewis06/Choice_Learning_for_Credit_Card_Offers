import numpy as np
import pytest

from market_shock_estimators.inversion import (
    check_market_inputs,
    simulate_shares,
    invert_market,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fixed_draws(n=500, seed=123):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


# ---------------------------------------------------------------------
# 1) check_market_inputs — valid case
# ---------------------------------------------------------------------
def test_check_market_inputs_accepts_valid_market():
    """
    Valid market inputs should pass without raising.
    """
    s_obs = np.array([0.2, 0.1, 0.15])
    s0 = 0.55
    p = np.array([1.0, 1.5, 2.0])

    check_market_inputs(s_obs, s0, p)


# ---------------------------------------------------------------------
# 2) check_market_inputs — invalid cases (consolidated)
# ---------------------------------------------------------------------
def test_check_market_inputs_rejects_invalid_inputs():
    """
    All infeasible or ill-defined markets should raise immediately.
    """
    p = np.array([1.0, 1.5, 2.0])

    invalid_cases = [
        # zero inside share
        (np.array([0.0, 0.2, 0.3]), 0.5, p),
        # negative inside share
        (np.array([-0.1, 0.2, 0.3]), 0.6, p),
        # zero outside share
        (np.array([0.3, 0.3, 0.4]), 0.0, p),
        # negative outside share
        (np.array([0.3, 0.3, 0.3]), -0.1, p),
        # shares do not sum to one
        (np.array([0.2, 0.2, 0.2]), 0.1, p),
        # shape mismatch
        (np.array([0.2, 0.2]), 0.6, p),
        # NaN in shares
        (np.array([0.2, np.nan, 0.1]), 0.7, p),
        # inf in prices
        (np.array([0.2, 0.1, 0.1]), 0.6, np.array([1.0, np.inf, 2.0])),
    ]

    for s_obs, s0, prices in invalid_cases:
        with pytest.raises(ValueError):
            check_market_inputs(s_obs, s0, prices)


# ---------------------------------------------------------------------
# 3) simulate_shares — probability sanity
# ---------------------------------------------------------------------
def test_simulate_shares_returns_valid_probabilities():
    """
    simulate_shares should return a valid probability vector
    with an implicit outside option.
    """
    delta = np.array([0.5, -0.2, 0.1])
    p = np.array([1.0, 1.5, 2.0])
    sigma = 0.8
    draws = fixed_draws()

    s_hat = simulate_shares(delta, p, sigma, draws)

    assert s_hat.ndim == 1
    assert len(s_hat) == len(delta)
    assert np.all(s_hat > 0.0)
    assert np.all(s_hat < 1.0)

    s0 = 1.0 - s_hat.sum()
    assert s0 > 0.0
    assert np.isclose(s_hat.sum() + s0, 1.0, atol=1e-10)


# ---------------------------------------------------------------------
# 4) invert_market — sigma = 0 (exact round-trip)
# ---------------------------------------------------------------------
def test_invert_market_sigma_zero():
    """
    With sigma = 0, inversion should exactly recover the delta
    that generated the shares.
    """
    delta_true = np.array([0.6, -0.4, 0.2])
    p = np.array([1.0, 1.5, 2.0])

    sigma = 0.0
    draws = fixed_draws()

    # Generate shares from the model itself (guaranteed fixed point)
    s_obs = simulate_shares(delta_true, p, sigma, draws)
    s0 = 1.0 - s_obs.sum()

    delta_hat, iters = invert_market(
        s_obs=s_obs,
        s0=s0,
        p=p,
        sigma=sigma,
        v_draws=draws,
    )

    assert iters > 0

    # Fixed-point check
    s_hat = simulate_shares(delta_hat, p, sigma, draws)
    assert np.allclose(s_hat, s_obs, atol=1e-10)


# ---------------------------------------------------------------------
# 5) invert_market — sigma ≠ 0 (round-trip fixed point)
# ---------------------------------------------------------------------
def test_invert_market_sigma_nonzero_roundtrip():
    """
    For sigma > 0, inversion should recover a delta that reproduces
    the observed shares (fixed-point correctness).
    """
    delta_true = np.array([0.8, -0.3, 0.2])
    p = np.array([1.0, 1.3, 1.8])
    sigma = 1.0
    draws = fixed_draws(n=1000)

    # Generate artificial data
    s_obs = simulate_shares(delta_true, p, sigma, draws)
    s0 = 1.0 - s_obs.sum()

    delta_hat, iters = invert_market(
        s_obs=s_obs,
        s0=s0,
        p=p,
        sigma=sigma,
        v_draws=draws,
    )

    assert iters > 0

    # Fixed-point condition (this is the correctness criterion)
    s_hat = simulate_shares(delta_hat, p, sigma, draws)
    assert np.allclose(s_hat, s_obs, atol=1e-6)
