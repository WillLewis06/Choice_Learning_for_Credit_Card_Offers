"""
Pytests for `lu.blp.blp` (BLP-style demand estimation on a small synthetic panel).

Contracts under test
--------------------
1) Instrument builders:
   - `build_strong_IVs(wjt, ujt)` and `build_weak_IVs(wjt)` return finite arrays with
     shape (n_markets, n_products, n_instruments).

2) Input validation:
   - `BLPEstimator(...)` rejects infeasible shares and inconsistent shapes at
     construction time (via its internal input checks).

3) End-to-end fit robustness:
   - `BLPEstimator.fit()` completes on a tiny synthetic panel and returns finite
     outputs in the expected schema.

4) Limiting case (sigma near 0):
   - When shares are generated from a sigma=0 logit model, constraining the fit to
     a tiny sigma interval yields beta estimates close to explicit 2SLS computed
     from the closed-form delta.

5) Internal objective consistency:
   - Holding the second-step weighting matrix fixed at the estimate, the GMM
     objective evaluated at `sigma_hat` should be no worse than at reasonable
     alternative sigma values within bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from lu_conftest import assert_finite_np, fixed_draws, make_feasible_shares
from lu.blp.blp import BLPEstimator, build_strong_IVs, build_weak_IVs

# -----------------------------------------------------------------------------
# Module constants (centralize repeated literals)
# -----------------------------------------------------------------------------
seed_panel_default = 0
seed_fit_default = 11
seed_draws_default = 123

n_draws_smoke = 10
n_draws_sigma_near_zero = 500
n_draws_objective = 200

sigma_min_default = 1e-3
sigma_max_default = 2.0
grid_step_default = 8

# In the sigma~0 test, inversion uses simulation draws even for tiny sigma, so
# the resulting beta estimates are only approximate.
beta_sigma_near_zero_atol = 1e-2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _logit_shares_from_delta(delta_j: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Given mean utilities `delta_j` (shape (n_products,)), return (inside_shares, outside_share)
    under standard logit with an outside option.

      inside_shares[j] = exp(delta_j) / (1 + sum_k exp(delta_k))
      outside_share     = 1            / (1 + sum_k exp(delta_k))
    """
    exp_delta = np.exp(delta_j - np.max(delta_j))  # numerical stabilization
    denom = 1.0 + float(np.sum(exp_delta))
    inside_shares = exp_delta / denom
    outside_share = 1.0 / denom
    return inside_shares, outside_share


def _make_toy_panel(
    n_markets: int = 3, n_products: int = 4, seed: int = seed_panel_default
):
    """
    Construct a small synthetic panel consistent with the estimator wiring.

    Data-generating sketch (sigma=0 shares):
      - price is endogenous by construction: p = alpha + 0.3*w + u
      - mean utility: delta = beta_p*p + beta_w*w + e

    Notes:
      - This is a unit-test panel, not an empirically realistic IV setup.
      - The "strong IV" builder uses `u` (unobserved in real settings) to create a
        strong instrument for test purposes.

    Returns:
      sjt: (n_markets, n_products) inside shares
      s0t: (n_markets,)            outside shares
      pjt: (n_markets, n_products) prices
      wjt: (n_markets, n_products) observed characteristic
      ujt: (n_markets, n_products) unobserved shock entering price
      ejt: (n_markets, n_products) unobserved shock entering delta
    """
    rng = np.random.default_rng(seed)

    beta_p_true = -1.0
    beta_w_true = 0.5
    alpha = 1.0

    wjt = rng.normal(size=(n_markets, n_products))
    ujt = rng.normal(size=(n_markets, n_products))
    ejt = 0.1 * rng.normal(size=(n_markets, n_products))

    pjt = alpha + 0.3 * wjt + ujt

    sjt = np.zeros((n_markets, n_products), dtype=float)
    s0t = np.zeros((n_markets,), dtype=float)
    for t in range(n_markets):
        delta_t = beta_p_true * pjt[t] + beta_w_true * wjt[t] + ejt[t]
        sj, s0 = _logit_shares_from_delta(delta_t)
        sjt[t] = sj
        s0t[t] = s0

    # Basic feasibility checks (shares strictly positive and sum to one)
    assert np.all(sjt > 0.0)
    assert np.all(s0t > 0.0)
    assert np.allclose(sjt.sum(axis=1) + s0t, 1.0, atol=1e-12, rtol=0.0)

    return sjt, s0t, pjt, wjt, ujt, ejt


def _fit_blp_once(
    sjt: np.ndarray,
    s0t: np.ndarray,
    pjt: np.ndarray,
    wjt: np.ndarray,
    zjt: np.ndarray,
    n_draws: int = n_draws_smoke,
    seed: int = seed_fit_default,
    sigma_init: float = 1.0,
    sigma_min: float = sigma_min_default,
    sigma_max: float = sigma_max_default,
    grid_step: int = grid_step_default,
) -> dict:
    """
    Fit the estimator once and return its results dictionary.

    This keeps the tests focused on contracts rather than repeated construction code.
    """
    est = BLPEstimator(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=zjt,  # estimator uses the paper-style name internally
        n_draws=n_draws,
        seed=seed,
    )
    est.fit(
        sigma_init=sigma_init,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        grid_step=grid_step,
    )
    return est.get_results()


def _assert_valid_results_schema(res: dict, sjt_shape: tuple[int, int]) -> None:
    """
    Assert the result dictionary contains finite estimates with consistent shapes.
    """
    assert isinstance(res, dict)
    assert res.get("success") is True

    sigma_hat = res.get("sigma_hat")
    assert sigma_hat is not None
    assert np.isfinite(sigma_hat)
    assert float(sigma_hat) > 0.0

    beta_p_hat = res.get("beta_p_hat")
    beta_w_hat = res.get("beta_w_hat")
    assert beta_p_hat is not None and np.isfinite(beta_p_hat)
    assert beta_w_hat is not None and np.isfinite(beta_w_hat)

    e_hat = res.get("E_hat")
    assert e_hat is not None
    assert isinstance(e_hat, np.ndarray)
    assert e_hat.shape == sjt_shape
    assert_finite_np(e_hat, name="E_hat")


def _delta_closed_form_sigma0(sjt: np.ndarray, s0t: np.ndarray) -> np.ndarray:
    """
    Closed-form delta for sigma=0 logit:
      delta_jt = log(s_jt) - log(s0_t)
    Returns array shape (n_markets, n_products).
    """
    return np.log(sjt) - np.log(s0t)[:, None]


def _two_stage_least_squares(
    delta: np.ndarray, x: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    2SLS estimator matching the algebra used by the estimator's beta step:

      beta = (X' Pz X)^+ X' Pz y
      Pz   = Z (Z'Z)^+ Z'

    Inputs:
      delta: (n_obs,) or (n_obs,1)
      x:     (n_obs, k)
      z:     (n_obs, l)

    Returns:
      beta: (k, 1)
    """
    y = np.asarray(delta, dtype=float).reshape(-1, 1)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    ztz_inv = np.linalg.pinv(z.T @ z)
    pz = z @ ztz_inv @ z.T

    xpzx = x.T @ pz @ x
    xpzy = x.T @ pz @ y
    return np.linalg.pinv(xpzx) @ xpzy


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_blp_build_ivs_shapes_and_finite():
    """
    Instrument builders return finite arrays with expected panel shapes.

    This is a shape/finite-value contract test, not an economic-validity test.
    """
    # Arrange
    _, _, _, wjt, ujt, _ = _make_toy_panel(n_markets=2, n_products=5, seed=1)

    # Act
    z_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
    z_weak = build_weak_IVs(wjt=wjt)

    # Assert
    assert z_strong.shape[:2] == (2, 5)
    assert z_weak.shape[:2] == (2, 5)

    # Current design: both builders produce 5 instruments.
    assert z_strong.shape[2] == 5
    assert z_weak.shape[2] == 5

    assert_finite_np(z_strong, name="z_strong")
    assert_finite_np(z_weak, name="z_weak")


@pytest.mark.parametrize(
    "overrides",
    [
        {"sjt": "make_sjt_zero"},  # non-positive inside share
        {"s0t": "make_s0t_zero"},  # non-positive outside share
        {"pjt": "drop_last_product"},  # shape mismatch in prices
        {"wjt": "drop_last_product"},  # shape mismatch in characteristics
        {"Zjt": "drop_last_product"},  # shape mismatch in instruments
    ],
    ids=[
        "reject_nonpositive_inside_share",
        "reject_nonpositive_outside_share",
        "reject_price_shape_mismatch",
        "reject_w_shape_mismatch",
        "reject_z_shape_mismatch",
    ],
)
def test_blp_input_validation_raises_valueerror_on_bad_shapes_or_shares(
    overrides: dict,
):
    """
    BLPEstimator rejects invalid inputs at construction time.

    This test asserts the estimator's input checker is responsible for rejecting:
      - infeasible shares
      - inconsistent (n_markets, n_products) shapes across arrays
    """
    # Arrange
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(n_markets=3, n_products=4, seed=2)
    zjt = build_weak_IVs(wjt=wjt)

    kwargs = dict(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=zjt)

    # Apply the requested invalidation.
    if overrides.get("sjt") == "make_sjt_zero":
        sjt_bad = sjt.copy()
        sjt_bad[0, 0] = 0.0
        kwargs["sjt"] = sjt_bad

    if overrides.get("s0t") == "make_s0t_zero":
        s0t_bad = s0t.copy()
        s0t_bad[0] = 0.0
        kwargs["s0t"] = s0t_bad

    if overrides.get("pjt") == "drop_last_product":
        kwargs["pjt"] = pjt[:, :-1]

    if overrides.get("wjt") == "drop_last_product":
        kwargs["wjt"] = wjt[:, :-1]

    if overrides.get("Zjt") == "drop_last_product":
        kwargs["Zjt"] = zjt[:, :-1, :]

    # Act / Assert
    with pytest.raises(ValueError):
        BLPEstimator(**kwargs, n_draws=5, seed=seed_fit_default)


@pytest.mark.parametrize("iv_kind", ["strong", "weak"], ids=["strong_iv", "weak_iv"])
def test_blp_end_to_end_runs_and_returns_finite_outputs(iv_kind: str):
    """
    End-to-end robustness check on a tiny sigma=0 DGP.

    This test does not assert parameter accuracy. It asserts:
      - fit completes successfully
      - returned estimates are finite
      - E_hat has the correct shape
    """
    # Arrange
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(n_markets=3, n_products=4, seed=3)
    zjt = (
        build_strong_IVs(wjt=wjt, ujt=ujt)
        if iv_kind == "strong"
        else build_weak_IVs(wjt=wjt)
    )

    # Act
    res = _fit_blp_once(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        zjt=zjt,
        n_draws=n_draws_smoke,
        seed=seed_fit_default,
    )

    # Assert
    _assert_valid_results_schema(res, sjt_shape=sjt.shape)


def test_blp_sigma_near_zero_matches_explicit_2sls():
    """
    Limiting-case correctness (sigma constrained near 0).

    Arrange:
      - generate shares from a sigma=0 logit DGP
      - compute closed-form delta and explicit 2SLS beta

    Act:
      - fit BLP with sigma constrained to a tiny interval (sigma_min <= sigma <= sigma_max)

    Assert:
      - estimated beta is close to explicit 2SLS (within simulation tolerance)
    """
    # Arrange
    n_markets, n_products = 6, 7
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(
        n_markets=n_markets, n_products=n_products, seed=10
    )
    zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    delta_cf = _delta_closed_form_sigma0(sjt=sjt, s0t=s0t)

    x = np.stack([pjt, wjt], axis=2).reshape(-1, 2)
    z = zjt.reshape(-1, zjt.shape[2])
    beta_2sls = _two_stage_least_squares(delta_cf.reshape(-1), x, z)

    # Act
    res = _fit_blp_once(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        zjt=zjt,
        n_draws=n_draws_sigma_near_zero,
        seed=seed_draws_default,
        sigma_init=1e-6,
        sigma_min=1e-10,
        sigma_max=1e-6,
        grid_step=grid_step_default,
    )

    # Assert
    assert res["success"] is True
    assert res["sigma_hat"] is not None
    assert 1e-10 <= float(res["sigma_hat"]) <= 1e-6

    beta_p_hat = float(res["beta_p_hat"])
    beta_w_hat = float(res["beta_w_hat"])
    assert np.isfinite(beta_p_hat)
    assert np.isfinite(beta_w_hat)

    assert np.isclose(
        beta_p_hat, float(beta_2sls[0, 0]), atol=beta_sigma_near_zero_atol, rtol=0.0
    )
    assert np.isclose(
        beta_w_hat, float(beta_2sls[1, 0]), atol=beta_sigma_near_zero_atol, rtol=0.0
    )


def test_blp_objective_at_sigma_hat_beats_alternatives_under_w2():
    """
    Internal objective consistency test.

    This test intentionally calls private methods to:
      (1) reconstruct the second-step weighting matrix at sigma_hat, and
      (2) compare the fixed-W2 objective at sigma_hat versus alternative sigma values.

    This guards against regressions where:
      - the objective is evaluated inconsistently with the stored weighting matrix, or
      - sigma selection drifts away from a local objective minimum within bounds.
    """
    # Arrange
    n_markets, n_products = 8, 6
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(
        n_markets=n_markets, n_products=n_products, seed=20
    )
    zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    est = BLPEstimator(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=zjt,
        n_draws=n_draws_objective,
        seed=seed_panel_default,
    )

    # Act (fit)
    est.fit(
        sigma_init=1.0,
        sigma_min=sigma_min_default,
        sigma_max=sigma_max_default,
        grid_step=grid_step_default,
    )

    # Assert (basic fit success)
    assert est.success is True
    assert est.sigma_hat is not None
    sigma_hat = float(est.sigma_hat)

    # Rebuild w2_hat at sigma_hat (as in the estimator's two-step GMM flow).
    delta_hat = est._invert_demand(sigma_hat)
    beta_hat = est._estimate_beta(delta_hat)
    e_hat = est._compute_E_hat(delta_hat, beta_hat)
    g_hat, omega_hat = est._moments_and_omega(e_hat)
    w2_hat = np.linalg.pinv(omega_hat)

    q_hat = float(g_hat @ w2_hat @ g_hat)
    assert np.isfinite(q_hat)

    sigma_lo = float(getattr(est, "_sigma_lo"))
    sigma_hi = float(getattr(est, "_sigma_hi"))
    assert 0.0 < sigma_lo <= sigma_hat <= sigma_hi

    # Candidate alternatives: bounds, multiplicative perturbations, and a log grid.
    alt_sigmas = [
        sigma_lo,
        sigma_hi,
        max(sigma_lo, sigma_hat * 0.2),
        max(sigma_lo, sigma_hat * 0.5),
        min(sigma_hi, sigma_hat * 2.0),
        min(sigma_hi, sigma_hat * 5.0),
    ]
    if sigma_hi > sigma_lo:
        alt_sigmas.extend(
            np.logspace(np.log10(sigma_lo), np.log10(sigma_hi), 7).tolist()
        )

    alt_sigmas = sorted({float(s) for s in alt_sigmas if np.isfinite(s) and s > 0.0})
    alt_sigmas = [s for s in alt_sigmas if abs(s - sigma_hat) > 1e-14]

    q_alts: list[float] = []
    for s in alt_sigmas:
        q = float(est._safe_gmm_objective(s, w2_hat))
        if np.isfinite(q) and (q < est.fail_penalty):
            q_alts.append(q)

    assert len(q_alts) >= 2
    assert q_hat <= (min(q_alts) + 1e-8)
