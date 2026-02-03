"""
Pytests for market_shock_estimators/blp.py aligned with simulation_run.py wiring.

Key alignment changes vs the older tests:
- BLPEstimator now takes (sjt, s0t, pjt, wjt, Zjt, n_draws, seed) and constructs Xjt and v_draws internally.
- Instruments are built externally via build_strong_IVs / build_weak_IVs (as in simulation_run.py).
- assess_estimator_results() no longer exists; toolbox.assess_estimator exposes print_assessment().
"""

from __future__ import annotations

import numpy as np
import pytest

from market_shock_estimators.blp import BLPEstimator, build_strong_IVs, build_weak_IVs
from toolbox.assess_estimator import print_assessment


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

    # Observables and cost shifter
    wjt = rng.normal(size=(T, J))
    ujt = rng.normal(size=(T, J))

    # Demand shock (xi / E in your codebase)
    Ejt = 0.1 * rng.normal(size=(T, J))

    # Pricing rule (matches simulation_run.py)
    pjt = alpha + 0.3 * wjt + ujt

    # sigma=0 logit-consistent shares
    sjt = np.zeros((T, J), dtype=float)
    s0t = np.zeros((T,), dtype=float)
    for t in range(T):
        delta_t = beta_p_true * pjt[t] + beta_w_true * wjt[t] + Ejt[t]
        sj, s0 = _logit_shares_from_delta(delta_t)
        sjt[t] = sj
        s0t[t] = s0

    # Basic sanity: shares in (0,1) and sum_j sjt + s0t = 1
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
):
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
        sigma_init=1.0,
        sigma_min=1e-3,
        sigma_max=2.0,
        grid_step=8,
    )
    return est.get_results()


def test_blp_build_ivs_shapes():
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=2, J=5, seed=1)

    Z_strong = build_strong_IVs(wjt=wjt, ujt=ujt)
    Z_weak = build_weak_IVs(wjt=wjt)

    assert Z_strong.shape == (2, 5, 5)  # [1, w, w^2, u, u^2]
    assert Z_weak.shape == (2, 5, 5)  # [1, w, w^2, w^3, w^4]
    assert np.all(np.isfinite(Z_strong))
    assert np.all(np.isfinite(Z_weak))


def test_blp_input_validation_raises_on_bad_shapes_or_shares():
    """
    BLPEstimator does not do heavy validation in __init__, so we assert that fit()
    fails on clearly invalid inputs (shape mismatches or zero shares).
    """
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=3, J=4, seed=2)
    Zjt = build_weak_IVs(wjt=wjt)

    # Zero inside share
    sjt_bad = sjt.copy()
    sjt_bad[0, 0] = 0.0

    # Zero outside share
    s0t_bad = s0t.copy()
    s0t_bad[0] = 0.0

    # Shape mismatches
    pjt_bad = pjt[:, :3]  # wrong J
    wjt_bad = wjt[:, :3]
    Zjt_bad = Zjt[:, :3, :]

    cases = [
        ("zero_inside_share", dict(sjt=sjt_bad)),
        ("zero_outside_share", dict(s0t=s0t_bad)),
        ("pjt_wrong_shape", dict(pjt=pjt_bad)),
        ("wjt_wrong_shape", dict(wjt=wjt_bad)),
        ("Zjt_wrong_shape", dict(Zjt=Zjt_bad)),
    ]

    base = dict(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=Zjt)

    for _, overrides in cases:
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


@pytest.mark.parametrize("iv_kind", ["strong", "weak"])
def test_blp_end_to_end_runs_and_returns_finite_outputs(iv_kind: str):
    """
    End-to-end robustness check on a tiny sigma=0 DGP.
    We do not assert parameter accuracy here; we assert:
      - fit completes
      - outputs are finite with correct shapes
    """
    sjt, s0t, pjt, wjt, ujt, _ = _make_toy_panel(T=3, J=4, seed=3)

    if iv_kind == "strong":
        Zjt = build_strong_IVs(wjt=wjt, ujt=ujt)
    else:
        Zjt = build_weak_IVs(wjt=wjt)

    res = _fit_blp_once(
        sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=Zjt, n_draws=10, seed=11
    )

    assert res["success"] is True

    assert res["sigma_hat"] is not None
    assert np.isfinite(res["sigma_hat"])
    assert float(res["sigma_hat"]) > 0.0

    # API returns beta_p_hat / beta_w_hat as scalars
    assert res["beta_p_hat"] is not None
    assert res["beta_w_hat"] is not None
    assert np.isfinite(res["beta_p_hat"])
    assert np.isfinite(res["beta_w_hat"])

    E_hat = res["E_hat"]
    assert E_hat is not None
    assert E_hat.shape == sjt.shape
    assert np.all(np.isfinite(E_hat))


def test_print_assessment_smoke_test(capsys):
    """
    Align with toolbox.assess_estimator.print_assessment().
    This is a smoke test: ensure it runs and prints expected fields.
    """
    sjt, s0t, pjt, wjt, ujt, E_true = _make_toy_panel(T=2, J=3, seed=4)
    Zjt = build_weak_IVs(wjt=wjt)

    res = _fit_blp_once(sjt=sjt, s0t=s0t, pjt=pjt, wjt=wjt, Zjt=Zjt, n_draws=6, seed=7)

    # print_assessment expects results dict with "E_hat" and optionally "sigma_hat"
    print_assessment(results=res, E_true=E_true, sigma_true=None)

    out = capsys.readouterr().out
    assert "rmse=" in out
    assert "mae=" in out
    assert "bias=" in out
    assert "corr=" in out
