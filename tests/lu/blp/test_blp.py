"""
Pytests for `lu.blp.blp` (BLP-style demand estimation on a small synthetic panel).

Contracts under test
--------------------
1) Instrument builders:
   - `build_strong_IVs(wjt, ujt)` and `build_weak_IVs(wjt)` return finite arrays with
     shape (n_markets, n_products, n_instruments).
   - Instrument columns match the fixed Lu(25) Section 4 benchmark definitions.
   - Public input-contract checks raise on invalid shapes.

2) End-to-end fit robustness (given internally-consistent arrays):
   - `BLPEstimator.fit()` completes on a tiny synthetic panel and returns finite
     outputs in the expected schema.

3) Limiting case (sigma near 0):
   - When shares are generated from a sigma=0 logit model, constraining the fit to
     a tiny sigma interval yields coefficients close to explicit 2SLS computed from
     the closed-form delta using X = [1, pjt, wjt].

4) Internal objective consistency:
   - Holding the second-step weighting matrix fixed at the estimate, the GMM
     objective evaluated at `sigma_hat` should be no worse than at reasonable
     alternative sigma values within bounds.

Notes
-----
- The estimator module assumes input arrays are produced internally and already
  consistent; it does not perform explicit input validation by design.
- Configuration is passed as a fully-specified `config` mapping and validated
  upstream. These tests therefore validate config through
  `validate_blp_config(...)` before constructing the estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from lu_conftest import assert_finite_np
from lu.blp.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from lu.blp.blp_input_validation import validate_blp_config

# -----------------------------------------------------------------------------
# Module constants (centralize repeated literals)
# -----------------------------------------------------------------------------
seed_panel_default = 0
seed_draws_default = 123

n_draws_smoke = 10
n_draws_sigma_near_zero = 500
n_draws_objective = 200

sigma_lower_default = 1e-3
sigma_upper_default = 2.0
sigma_grid_points_default = 8

coef_sigma_near_zero_atol = 1e-2

damping_default = 0.7
tol_default = 1e-10
share_tol_smoke = 1e-4
share_tol_strict = 1e-8
max_iter_default = 5000

nelder_mead_maxiter_default = 200
nelder_mead_xatol_default = 1e-4
nelder_mead_fatol_default = 1e-6

fail_penalty_default = 1e30


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_config(
    n_draws: int,
    seed: int,
    sigma_lower: float,
    sigma_upper: float,
    sigma_grid_points: int,
    share_tol: float,
) -> dict:
    """Return the validated config mapping expected by `BLPEstimator`."""
    raw = {
        "n_draws": int(n_draws),
        "seed": int(seed),
        "sigma_lower": float(sigma_lower),
        "sigma_upper": float(sigma_upper),
        "sigma_grid_points": int(sigma_grid_points),
        "damping": float(damping_default),
        "tol": float(tol_default),
        "share_tol": float(share_tol),
        "max_iter": int(max_iter_default),
        "fail_penalty": float(fail_penalty_default),
        "nelder_mead_maxiter": int(nelder_mead_maxiter_default),
        "nelder_mead_xatol": float(nelder_mead_xatol_default),
        "nelder_mead_fatol": float(nelder_mead_fatol_default),
    }
    return validate_blp_config(raw)


def _logit_shares_from_delta(delta_j: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Given mean utilities `delta_j` (shape (n_products,)), return
    (inside_shares, outside_share) under standard logit with an outside option.

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
        delta_t = alpha + beta_p_true * pjt[t] + beta_w_true * wjt[t] + ejt[t]
        sj, s0 = _logit_shares_from_delta(delta_t)
        sjt[t] = sj
        s0t[t] = s0

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
    config: dict,
) -> dict:
    """Fit once and return the estimator's results dictionary."""
    est = BLPEstimator(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=zjt, config=config)
    est.fit()
    return est.get_results()


def _assert_valid_results_schema(res: dict, sjt_shape: tuple[int, int]) -> None:
    """Assert the result dictionary contains finite estimates with consistent shapes."""
    assert isinstance(res, dict)
    assert res.get("success") is True

    sigma_hat = res.get("sigma_hat")
    assert sigma_hat is not None
    assert np.isfinite(sigma_hat)
    assert float(sigma_hat) > 0.0

    int_hat = res.get("int_hat")
    beta_p_hat = res.get("beta_p_hat")
    beta_w_hat = res.get("beta_w_hat")
    assert int_hat is not None and np.isfinite(int_hat)
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

    This is a shape/finite-value contract test.
    """
    _, _, _, wjt, ujt, _ = _make_toy_panel(n_markets=2, n_products=5, seed=1)

    z_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
    z_weak = build_weak_IVs(wjt=wjt)

    assert z_strong.shape == (2, 5, 5)
    assert z_weak.shape == (2, 5, 5)

    assert_finite_np(z_strong, name="z_strong")
    assert_finite_np(z_weak, name="z_weak")


def test_blp_build_ivs_exact_columns():
    """Instrument builders match the documented fixed column definitions exactly."""
    _, _, _, wjt, ujt, _ = _make_toy_panel(n_markets=3, n_products=4, seed=2)

    z_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
    z_weak = build_weak_IVs(wjt=wjt)

    assert np.array_equal(z_strong[:, :, 0], np.ones_like(wjt))
    assert np.array_equal(z_strong[:, :, 1], wjt)
    assert np.array_equal(z_strong[:, :, 2], wjt**2)
    assert np.array_equal(z_strong[:, :, 3], ujt)
    assert np.array_equal(z_strong[:, :, 4], ujt**2)

    assert np.array_equal(z_weak[:, :, 0], np.ones_like(wjt))
    assert np.array_equal(z_weak[:, :, 1], wjt)
    assert np.array_equal(z_weak[:, :, 2], wjt**2)
    assert np.array_equal(z_weak[:, :, 3], wjt**3)
    assert np.array_equal(z_weak[:, :, 4], wjt**4)


def test_blp_build_strong_ivs_invalid_shapes_raise():
    """Strong IV builder rejects non-2D inputs and mismatched shapes."""
    wjt = np.zeros((3, 4), dtype=float)
    ujt = np.zeros((3, 4), dtype=float)

    with pytest.raises(ValueError):
        build_strong_IVs(wjt=wjt[0], ujt=ujt)

    with pytest.raises(ValueError):
        build_strong_IVs(wjt=wjt, ujt=ujt[..., None])

    with pytest.raises(ValueError):
        build_strong_IVs(wjt=wjt, ujt=np.zeros((3, 5), dtype=float))


def test_blp_build_weak_ivs_invalid_shapes_raise():
    """Weak IV builder rejects non-2D inputs."""
    wjt = np.zeros((3, 4), dtype=float)

    with pytest.raises(ValueError):
        build_weak_IVs(wjt=wjt[0])

    with pytest.raises(ValueError):
        build_weak_IVs(wjt=wjt[..., None])


@pytest.mark.parametrize("iv_kind", ["strong", "weak"], ids=["strong_iv", "weak_iv"])
def test_blp_end_to_end_runs_and_returns_finite_outputs(iv_kind: str):
    """
    End-to-end robustness check on a tiny sigma=0 DGP.

    This test does not assert parameter accuracy. It asserts:
      - fit completes successfully
      - returned estimates are finite
      - E_hat has the correct shape
    """
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(n_markets=3, n_products=4, seed=3)
    zjt = (
        build_strong_IVs(wjt=wjt, ujt=ujt)
        if iv_kind == "strong"
        else build_weak_IVs(wjt=wjt)
    )

    config = _make_config(
        n_draws=n_draws_smoke,
        seed=seed_draws_default,
        sigma_lower=sigma_lower_default,
        sigma_upper=sigma_upper_default,
        sigma_grid_points=sigma_grid_points_default,
        share_tol=share_tol_smoke,
    )

    res = _fit_blp_once(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, zjt=zjt, config=config)

    _assert_valid_results_schema(res, sjt_shape=sjt.shape)


def test_blp_sigma_near_zero_matches_explicit_2sls():
    """
    Limiting-case correctness (sigma constrained near 0).

    Arrange:
      - generate shares from a sigma=0 logit DGP
      - compute closed-form delta and explicit 2SLS beta using X = [1, pjt, wjt]

    Act:
      - fit BLP with sigma constrained to a tiny interval (sigma_lower <= sigma <= sigma_upper)

    Assert:
      - estimated coefficients are close to explicit 2SLS (within simulation tolerance)
    """
    n_markets, n_products = 6, 7
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(
        n_markets=n_markets, n_products=n_products, seed=10
    )
    zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    delta_cf = _delta_closed_form_sigma0(sjt=sjt, s0t=s0t)

    ones = np.ones_like(pjt, dtype=float)
    x = np.stack([ones, pjt, wjt], axis=2).reshape(-1, 3)
    z = zjt.reshape(-1, zjt.shape[2])
    beta_2sls = _two_stage_least_squares(delta_cf.reshape(-1), x, z)

    sigma_lower = 1e-10
    sigma_upper = 1e-6

    config = _make_config(
        n_draws=n_draws_sigma_near_zero,
        seed=seed_draws_default,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        sigma_grid_points=sigma_grid_points_default,
        share_tol=share_tol_strict,
    )

    res = _fit_blp_once(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, zjt=zjt, config=config)

    assert res["success"] is True
    assert res["sigma_hat"] is not None
    assert sigma_lower <= float(res["sigma_hat"]) <= sigma_upper

    int_hat = float(res["int_hat"])
    beta_p_hat = float(res["beta_p_hat"])
    beta_w_hat = float(res["beta_w_hat"])
    assert np.isfinite(int_hat)
    assert np.isfinite(beta_p_hat)
    assert np.isfinite(beta_w_hat)

    assert np.isclose(
        int_hat, float(beta_2sls[0, 0]), atol=coef_sigma_near_zero_atol, rtol=0.0
    )
    assert np.isclose(
        beta_p_hat, float(beta_2sls[1, 0]), atol=coef_sigma_near_zero_atol, rtol=0.0
    )
    assert np.isclose(
        beta_w_hat, float(beta_2sls[2, 0]), atol=coef_sigma_near_zero_atol, rtol=0.0
    )


def test_blp_objective_at_sigma_hat_beats_alternatives_under_w2():
    """
    Internal objective consistency test.

    This test intentionally calls private methods to:
      (1) reconstruct the second-step weighting matrix at sigma_hat, and
      (2) compare the fixed-W2 objective at sigma_hat versus alternative sigma values.

    Key detail:
      - The estimator warm-starts the Berry inversion and mutates `_delta_warm_start`
        on each inversion call. To keep objective comparisons meaningful, this test
        resets the warm start to the deterministic logit start before each objective
        evaluation.
    """
    n_markets, n_products = 8, 6
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(
        n_markets=n_markets, n_products=n_products, seed=20
    )
    zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    config = _make_config(
        n_draws=n_draws_objective,
        seed=seed_draws_default,
        sigma_lower=sigma_lower_default,
        sigma_upper=sigma_upper_default,
        sigma_grid_points=sigma_grid_points_default,
        share_tol=share_tol_strict,
    )

    est = BLPEstimator(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=zjt,
        config=config,
    )
    est.fit()

    assert est.success is True
    assert est.sigma_hat is not None
    sigma_hat = float(est.sigma_hat)

    sigma_lo = float(getattr(est, "_sigma_lo"))
    sigma_hi = float(getattr(est, "_sigma_hi"))
    assert 0.0 < sigma_lo <= sigma_hat <= sigma_hi

    est._delta_warm_start = est._delta_init0
    delta_hat = est._invert_demand(sigma_hat)
    beta_hat = est._estimate_beta(delta_hat)
    e_hat = est._compute_E_hat(delta_hat, beta_hat)
    g_hat, omega_hat = est._moments_and_omega(e_hat)
    w2_hat = np.linalg.pinv(omega_hat)

    q_hat = float(g_hat @ w2_hat @ g_hat)
    assert np.isfinite(q_hat)

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
        est._delta_warm_start = est._delta_init0
        q = float(est._safe_gmm_objective(s, w2_hat))
        if np.isfinite(q) and (q < float(est.config["fail_penalty"])):
            q_alts.append(q)

    assert len(q_alts) >= 2
    assert q_hat <= (min(q_alts) + 1e-6)
