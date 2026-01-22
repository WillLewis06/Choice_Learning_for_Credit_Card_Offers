import numpy as np
import pytest

from market_shock_estimators.blp import BLPEstimator
from market_shock_estimators.assess_estimator import assess_estimator_results


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

    for _, overrides in cases:
        kwargs = dict(base)
        kwargs.update(overrides)
        with pytest.raises((ValueError, RuntimeError)):
            BLPEstimator(**kwargs).fit(
                sigma_init=0.2,
                sigma_min=1e-3,
                sigma_max=1.0,
                grid_step=15,
            )


def test_blp_converges_on_trivial_market():
    """
    End-to-end robustness check: should run and return finite outputs on a tiny,
    logit-consistent dataset (sigma=0 data-generating shares).
    """
    sjt, s0t, pjt, Xjt, Zjt, v_draws = _make_trivial_panel()

    est = BLPEstimator(sjt=sjt, s0t=s0t, pjt=pjt, Xjt=Xjt, Zjt=Zjt, v_draws=v_draws)
    est.fit(
        sigma_init=0.2,
        sigma_min=1e-3,
        sigma_max=1.0,
        grid_step=15,
    )

    res = est.get_results()

    assert res["success"] is True
    assert res["sigma_hat"] is not None
    assert np.isfinite(res["sigma_hat"])
    assert res["sigma_hat"] > 0.0

    assert res["beta_hat"] is not None
    assert np.all(np.isfinite(res["beta_hat"]))

    E_hat = res["E_hat"]
    assert E_hat is not None
    assert E_hat.shape == sjt.shape
    assert np.all(np.isfinite(E_hat))


def test_assess_estimator_results_ok_and_failure_paths():
    """
    Unit tests for assess_estimator_results:
      - OK path returns finite metrics with ok=True
      - failure path returns ok=False with a reason
    """
    rng = np.random.default_rng(0)

    # OK case
    E_true = rng.normal(size=(2, 3))
    noise = 0.1 * rng.normal(size=E_true.shape)
    E_hat = E_true + noise

    results_ok = {"success": True, "E_hat": E_hat, "sigma_hat": 1.2}

    ass = assess_estimator_results(
        name="dummy",
        results=results_ok,
        E_true=E_true,
        sigma_true=1.5,
    )

    assert ass["ok"] is True
    assert ass["reason"] is None
    for k in (
        "rmse",
        "mae",
        "bias",
        "corr",
        "std_ratio",
        "rmse_null",
        "rmse_improvement",
    ):
        assert ass[k] is not None
        assert np.isfinite(ass[k])

    assert ass["sigma_abs_err"] is not None
    assert np.isfinite(ass["sigma_abs_err"])
    assert ass["sigma_rel_err"] is not None
    assert np.isfinite(ass["sigma_rel_err"])

    # Failure case: success=False
    results_fail = {"success": False, "E_hat": None, "sigma_hat": None}
    ass2 = assess_estimator_results(
        name="dummy_fail",
        results=results_fail,
        E_true=E_true,
        sigma_true=1.5,
    )

    assert ass2["ok"] is False
    assert isinstance(ass2.get("reason"), str)
    assert ass2["rmse"] is None
    assert ass2["mae"] is None
