"""
Pytests for market_shock_estimators/blp.py aligned with simulation_run.py wiring.

Simplifications:
- Use shared NumPy helper assert_finite_np from tests/conftest.py.
- No pytest markers/parametrize (use simple loops).
- Remove print-only smoke tests (they test formatting, not estimator behavior).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_finite_np
from market_shock_estimators.blp import BLPEstimator, build_strong_IVs, build_weak_IVs


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _logit_shares_from_delta(delta_tj: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Given delta_tj shape (J,), return (s_j, s0) under standard logit:
      s_j = exp(delta_j) / (1 + sum_k exp(delta_k))
      s0  = 1 / (1 + sum_k exp(delta_k))
    """
    ex = np.exp(delta_tj - np.max(delta_tj))  # stabilize
    denom = 1.0 + float(np.sum(ex))
    sj = ex / denom
    s0 = 1.0 / denom
    return sj, s0


def _make_toy_panel(T: int = 3, J: int = 4, seed: int = 0):
    """
    Construct a tiny Lu-style panel consistent with simulation_run wiring:
      p = alpha + 0.3*w + u
      delta = beta_p*p + beta_w*w + E   (sigma=0 DGP shares)

    Returns:
      sjt (T,J), s0t (T,), pjt (T,J), wjt (T,J), ujt (T,J), Ejt (T,J)
    """
    rng = np.random.default_rng(seed)

    beta_p_true = -1.0
    beta_w_true = 0.5
    alpha = 1.0

    wjt = rng.normal(size=(T, J))
    ujt = rng.normal(size=(T, J))
    Ejt = 0.1 * rng.normal(size=(T, J))

    pjt = alpha + 0.3 * wjt + ujt

    sjt = np.zeros((T, J), dtype=float)
    s0t = np.zeros((T,), dtype=float)
    for t in range(T):
        delta_t = beta_p_true * pjt[t] + beta_w_true * wjt[t] + Ejt[t]
        sj, s0 = _logit_shares_from_delta(delta_t)
        sjt[t] = sj
        s0t[t] = s0

    assert np.all(sjt > 0.0)
    assert np.all(s0t > 0.0)
    assert np.allclose(sjt.sum(axis=1) + s0t, 1.0, atol=1e-12)

    return sjt, s0t, pjt, wjt, ujt, Ejt


def _fit_blp_once(
    sjt: np.ndarray,
    s0t: np.ndarray,
    pjt: np.ndarray,
    wjt: np.ndarray,
    Zjt: np.ndarray,
    *,
    n_draws: int = 10,
    seed: int = 0,
    sigma_init: float = 1.0,
    sigma_min: float = 1e-3,
    sigma_max: float = 2.0,
    grid_step: int = 8,
) -> dict:
    est = BLPEstimator(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=Zjt,
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


def _delta_closed_form_sigma0(sjt: np.ndarray, s0t: np.ndarray) -> np.ndarray:
    """
    For sigma=0 logit:
      delta_jt = log(s_jt) - log(s0_t)
    Returns array shape (T,J).
    """
    return np.log(sjt) - np.log(s0t)[:, None]


def _two_stage_least_squares(
    delta: np.ndarray, X: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """
    2SLS estimator matching BLPEstimator._estimate_beta():

      beta = (X' Pz X)^+ X' Pz y
      Pz = Z (Z'Z)^+ Z'
    """
    y = np.asarray(delta, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)

    ZTZ_inv = np.linalg.pinv(Z.T @ Z)
    Pz = Z @ ZTZ_inv @ Z.T

    XPZX = X.T @ Pz @ X
    XPZy = X.T @ Pz @ y
    return np.linalg.pinv(XPZX) @ XPZy


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_blp_build_ivs_shapes_and_finite():
    _, _, _, wjt, ujt, _ = _make_toy_panel(T=2, J=5, seed=1)

    Z_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
    Z_weak = build_weak_IVs(wjt=wjt)

    assert Z_strong.shape == (2, 5, 5)  # [1, w, w^2, u, u^2]
    assert Z_weak.shape == (2, 5, 5)  # [1, w, w^2, w^3, w^4]

    assert_finite_np(Z_strong, name="Z_strong")
    assert_finite_np(Z_weak, name="Z_weak")


def test_blp_input_validation_raises_on_bad_shapes_or_shares():
    """
    BLPEstimator runs input validation in __init__ (via _check_inputs).
    This test ensures clearly invalid inputs do not silently proceed.
    """
    sjt, s0t, pjt, wjt, _, _ = _make_toy_panel(T=3, J=4, seed=2)
    Zjt = build_weak_IVs(wjt=wjt)

    sjt_bad = sjt.copy()
    sjt_bad[0, 0] = 0.0  # invalid (non-positive inside share)

    s0t_bad = s0t.copy()
    s0t_bad[0] = 0.0  # invalid (non-positive outside share)

    pjt_bad = pjt[:, :3]  # wrong J
    wjt_bad = wjt[:, :3]  # wrong J
    Zjt_bad = Zjt[:, :3, :]  # wrong J

    cases = [
        dict(sjt=sjt_bad),
        dict(s0t=s0t_bad),
        dict(pjt=pjt_bad),
        dict(wjt=wjt_bad),
        dict(Zjt=Zjt_bad),
    ]

    base = dict(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=Zjt)

    for overrides in cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises(
            (ValueError, RuntimeError, FloatingPointError, AssertionError)
        ):
            BLPEstimator(**kwargs, n_draws=5, seed=0).fit(
                sigma_init=1.0,
                sigma_min=1e-3,
                sigma_max=2.0,
                grid_step=5,
            )


def test_blp_end_to_end_runs_and_returns_finite_outputs():
    """
    End-to-end robustness check on a tiny sigma=0 DGP.
    Does not assert parameter accuracy; asserts fit completes and outputs are finite.
    """
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=3, J=4, seed=3)

    for iv_kind in ["strong", "weak"]:
        Zjt = (
            build_strong_IVs(wjt=wjt, ujt=ujt)
            if iv_kind == "strong"
            else build_weak_IVs(wjt=wjt)
        )

        res = _fit_blp_once(
            sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=Zjt, n_draws=10, seed=11
        )

        assert res["success"] is True

        assert res["sigma_hat"] is not None
        assert np.isfinite(res["sigma_hat"])
        assert float(res["sigma_hat"]) > 0.0

        assert res["beta_p_hat"] is not None
        assert res["beta_w_hat"] is not None
        assert np.isfinite(res["beta_p_hat"])
        assert np.isfinite(res["beta_w_hat"])

        E_hat = res["E_hat"]
        assert E_hat is not None
        assert E_hat.shape == sjt.shape
        assert_finite_np(E_hat, name="E_hat")


def test_blp_sigma_near_zero_matches_explicit_2sls():
    """
    Limiting-case correctness (sigma ~ 0):

    - Data are generated from sigma=0 logit shares.
    - delta has closed form delta = log(s) - log(s0).
    - Given delta, beta has a closed-form 2SLS solution.

    Constrain fit to a tiny sigma interval and compare BLPEstimator beta to explicit 2SLS.
    """
    T, J = 6, 7
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=T, J=J, seed=10)
    Zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    delta_cf = _delta_closed_form_sigma0(sjt=sjt, s0t=s0t)

    X = np.stack([pjt, wjt], axis=2).reshape(-1, 2)
    Z = Zjt.reshape(-1, Zjt.shape[2])
    beta_2sls = _two_stage_least_squares(delta_cf.reshape(-1), X, Z)

    res = _fit_blp_once(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=Zjt,
        n_draws=500,
        seed=123,
        sigma_init=1e-6,
        sigma_min=1e-10,
        sigma_max=1e-6,
        grid_step=8,
    )

    assert res["success"] is True
    assert res["sigma_hat"] is not None
    assert 1e-10 <= float(res["sigma_hat"]) <= 1e-6

    beta_p_hat = float(res["beta_p_hat"])
    beta_w_hat = float(res["beta_w_hat"])

    assert np.isfinite(beta_p_hat)
    assert np.isfinite(beta_w_hat)

    # With sigma>0 (even tiny), inversion uses simulation draws; allow loose tolerance.
    assert np.isclose(beta_p_hat, float(beta_2sls[0, 0]), atol=1e-2, rtol=0.0)
    assert np.isclose(beta_w_hat, float(beta_2sls[1, 0]), atol=1e-2, rtol=0.0)


def test_blp_objective_at_sigma_hat_beats_alternatives_under_W2():
    """
    Robust correctness check:

    Fix W2_hat constructed at sigma_hat (second-step weighting).
    Then Q(sigma) = g_bar(sigma)' W2_hat g_bar(sigma) should be minimized (approximately)
    at sigma_hat relative to a set of alternative sigma values inside the fit bounds.
    """
    T, J = 8, 6
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=T, J=J, seed=20)
    Zjt = build_strong_IVs(wjt=wjt, ujt=ujt)

    est = BLPEstimator(
        sjt=sjt,
        s0t=s0t,
        pjt=pjt,
        wjt=wjt,
        Zjt=Zjt,
        n_draws=200,
        seed=0,
    )
    est.fit(sigma_init=1.0, sigma_min=1e-3, sigma_max=2.0, grid_step=8)

    assert est.success is True
    assert est.sigma_hat is not None
    sigma_hat = float(est.sigma_hat)

    # Rebuild W2_hat at sigma_hat (as in fit()).
    delta_hat = est._invert_demand(sigma_hat)
    beta_hat = est._estimate_beta(delta_hat)
    E_hat = est._compute_E_hat(delta_hat, beta_hat)
    g_hat, Omega_hat = est._moments_and_omega(E_hat)
    W2_hat = np.linalg.pinv(Omega_hat)

    Q_hat = float(g_hat @ W2_hat @ g_hat)
    assert np.isfinite(Q_hat)

    sigma_lo = float(getattr(est, "_sigma_lo"))
    sigma_hi = float(getattr(est, "_sigma_hi"))
    assert 0.0 < sigma_lo <= sigma_hat <= sigma_hi

    alts = [
        sigma_lo,
        sigma_hi,
        max(sigma_lo, sigma_hat * 0.2),
        max(sigma_lo, sigma_hat * 0.5),
        min(sigma_hi, sigma_hat * 2.0),
        min(sigma_hi, sigma_hat * 5.0),
    ]
    if sigma_hi > sigma_lo:
        alts.extend(np.logspace(np.log10(sigma_lo), np.log10(sigma_hi), 7).tolist())

    alts = sorted({float(a) for a in alts if np.isfinite(a) and a > 0.0})
    alts = [a for a in alts if abs(a - sigma_hat) > 1e-14]

    Q_alts = []
    for s in alts:
        q = float(est._safe_gmm_objective(s, W2_hat))
        if np.isfinite(q) and (q < est.fail_penalty):
            Q_alts.append(q)

    assert len(Q_alts) >= 2
    assert Q_hat <= (min(Q_alts) + 1e-8)
