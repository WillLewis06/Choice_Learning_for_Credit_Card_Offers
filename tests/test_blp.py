import numpy as np
import pytest

from market_shock_estimators.blp import BLPEstimator


def _make_trivial_panel(T: int = 2, J: int = 3):
    """
    Tiny, well-posed logit-consistent panel (sigma=0) that should converge easily.

    Returns:
      sjt  : (T,J)
      s0t  : (T,)
      pjt  : (T,J)
      Xjt  : (T,J,Kx) with columns [1, w, p]
      Zjt  : (T,J,Kz) (set equal to Xjt for full-rank toy IV)
      v_draws : (R,)
    """

    # Mean utilities delta* (moderate values; vary slightly across markets)
    delta_star = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-0.8, 0.2, 0.9],
        ],
        dtype=float,
    )  # (T,J)

    # Analytic logit shares (sigma=0)
    exp_delta = np.exp(delta_star)
    denom = 1.0 + exp_delta.sum(axis=1, keepdims=True)
    sjt = exp_delta / denom  # (T,J)
    s0t = (1.0 / denom).reshape(T)  # (T,)

    # Non-degenerate prices and characteristics (vary across markets/products)
    pjt = np.array(
        [
            [1.0, 1.5, 2.0],
            [1.1, 1.4, 2.1],
        ],
        dtype=float,
    )
    wjt = np.array(
        [
            [1.2, 1.5, 1.8],
            [1.1, 1.6, 1.9],
        ],
        dtype=float,
    )

    # Xjt: [1, w, p]
    Xjt = np.stack([np.ones_like(wjt), wjt, pjt], axis=2)  # (T,J,3)

    # Zjt: use Xjt as instruments for this toy test (full rank, stable)
    Zjt = Xjt.copy()  # (T,J,3)

    # Fixed simulation draws (only used inside inversion/share simulation)
    v_draws = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)

    return sjt, s0t, pjt, Xjt, Zjt, v_draws


def test_blp_input_validation():
    """
    Packed validation: bad shares/shapes should raise when fit() calls inversion/estimation.
    Keep it minimal and avoid running a "good" fit in this test.
    """
    sjt, s0t, pjt, Xjt, Zjt, v_draws = _make_trivial_panel()

    base = dict(sjt=sjt, s0t=s0t, pjt=pjt, Xjt=Xjt, Zjt=Zjt, v_draws=v_draws)

    # Build invalid variants from the base data
    sjt_bad = sjt.copy()
    sjt_bad[0, 0] = 0.0

    s0t_bad = s0t.copy()
    s0t_bad[0] = 0.0

    cases = [
        ("zero_inside_share", dict(sjt=sjt_bad)),
        ("zero_outside_share", dict(s0t=s0t_bad)),
        ("pjt_wrong_shape", dict(pjt=pjt[:, :2])),
        ("Xjt_wrong_shape", dict(Xjt=Xjt[:, :2, :])),
        ("Zjt_wrong_shape", dict(Zjt=Zjt[:, :2, :])),
    ]

    for name, overrides in cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises((ValueError, RuntimeError)):
            BLPEstimator(**kwargs).fit(sigma_init=0.2)


def test_blp_converges_on_trivial_market():
    """
    End-to-end robustness check: should run and return finite outputs on a tiny,
    logit-consistent dataset (sigma=0 data-generating shares).
    """
    sjt, s0t, pjt, Xjt, Zjt, v_draws = _make_trivial_panel()

    est = BLPEstimator(sjt=sjt, s0t=s0t, pjt=pjt, Xjt=Xjt, Zjt=Zjt, v_draws=v_draws)
    est.fit(sigma_init=0.2)

    assert est.sigma_hat is not None
    assert np.isfinite(est.sigma_hat)
    assert est.sigma_hat >= 0.0

    assert est.beta_hat is not None
    assert np.all(np.isfinite(est.beta_hat))

    E_hat = est.get_E_hat()
    assert E_hat.shape == sjt.shape
    assert np.all(np.isfinite(E_hat))

    # Robustness improvement: if the estimator stores an objective value, it should be finite.
    for attr in ("objective", "objective_", "gmm_objective", "gmm_objective_"):
        if hasattr(est, attr):
            val = getattr(est, attr)
            if val is not None:
                assert np.isfinite(val)
            break


def test_blp_invariant_to_product_permutation():
    """
    Permuting product order (consistently across sjt/pjt/Xjt/Zjt) should not materially
    change sigma_hat or beta_hat. Use loose tolerances since the outer optimizer is
    not guaranteed to land on bit-identical solutions.
    """
    sjt, s0t, pjt, Xjt, Zjt, v_draws = _make_trivial_panel()

    est1 = BLPEstimator(sjt=sjt, s0t=s0t, pjt=pjt, Xjt=Xjt, Zjt=Zjt, v_draws=v_draws)
    est1.fit(sigma_init=0.2)

    perm = np.array([2, 0, 1])  # fixed permutation for J=3

    sjt_p = sjt[:, perm]
    pjt_p = pjt[:, perm]
    Xjt_p = Xjt[:, perm, :]
    Zjt_p = Zjt[:, perm, :]

    est2 = BLPEstimator(
        sjt=sjt_p, s0t=s0t, pjt=pjt_p, Xjt=Xjt_p, Zjt=Zjt_p, v_draws=v_draws
    )
    est2.fit(sigma_init=0.2)

    assert np.isfinite(est2.sigma_hat)
    assert np.all(np.isfinite(est2.beta_hat))

    # Loose tolerances: robustness-oriented, not precision-oriented
    assert np.isclose(est1.sigma_hat, est2.sigma_hat, rtol=1e-4, atol=1e-6)
    assert np.allclose(est1.beta_hat, est2.beta_hat, rtol=1e-4, atol=1e-6)
