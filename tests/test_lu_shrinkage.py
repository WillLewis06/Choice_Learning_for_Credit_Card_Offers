import numpy as np
import pytest

from market_shock_estimators.lu_shrinkage import LuShrinkageEstimator


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def tiny_problem():
    """
    Smallest nontrivial problem that exercises all code paths.
    """
    rng = np.random.default_rng(123)

    T, J, K = 1, 2, 2
    R = 50  # number of RC draws

    x_jt = rng.normal(size=(T, J, K))
    q_jt = np.array([[10, 5]], dtype=float)
    q0_t = np.array([20.0])

    draws = rng.normal(size=R)

    return dict(
        x_jt=x_jt,
        q_jt=q_jt,
        q0_t=q0_t,
        draws=draws,
    )


# ---------------------------------------------------------------------
# 1. Construction and initialization
# ---------------------------------------------------------------------
def test_constructor_and_initialize(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    T, J, K = est.T, est.J, est.K

    assert est.beta.shape == (K,)
    assert np.isscalar(est.r)
    assert est.E_bar_t.shape == (T,)
    assert est.njt.shape == (T, J)

    assert est.gamma_jt.shape == (T, J)
    assert set(np.unique(est.gamma_jt)).issubset({0, 1})

    assert est.phi_t.shape == (T,)
    assert np.all((est.phi_t > 0) & (est.phi_t < 1))


def test_constructor_rejects_bad_T0_T1(tiny_problem):
    with pytest.raises(ValueError):
        LuShrinkageEstimator(**tiny_problem, T0_sq=-1.0)


# ---------------------------------------------------------------------
# 2. Deterministic components
# ---------------------------------------------------------------------
def test_delta_jt_linearity(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    beta = np.ones(est.K)
    E_bar = np.zeros(est.T)
    njt = np.zeros((est.T, est.J))

    delta0 = est.delta_jt(beta, E_bar, njt)

    E_bar2 = E_bar + 1.0
    delta1 = est.delta_jt(beta, E_bar2, njt)

    # All products in the market shift equally
    assert np.allclose(delta1 - delta0, 1.0)


def test_choice_probs_identity(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    delta = np.zeros((est.T, est.J))
    p_jt, p0_t = est.choice_probs(delta, r=0.0)

    assert p_jt.shape == (est.T, est.J)
    assert p0_t.shape == (est.T,)

    total = p0_t + p_jt.sum(axis=1)
    assert np.allclose(total, 1.0, atol=1e-12)


def test_choice_probs_extreme_utilities(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    delta = -1000.0 * np.ones((est.T, est.J))
    p_jt, p0_t = est.choice_probs(delta, r=0.0)

    assert np.all(p0_t > 1.0 - 1e-10)
    assert np.all(p_jt < 1e-10)


# ---------------------------------------------------------------------
# 3. Log posterior
# ---------------------------------------------------------------------
def test_log_posterior_finite(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    lp = est.log_posterior(
        est.beta, est.r, est.E_bar_t, est.njt, est.gamma_jt, est.phi_t
    )

    assert np.isfinite(lp)


def test_log_posterior_monotone_in_njt(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    lp0 = est._full_log_posterior()

    # Increase utility for product 0
    est.njt[0, 0] += 0.5
    lp1 = est._full_log_posterior()

    if est.q_jt[0, 0] > 0:
        assert lp1 > lp0


# ---------------------------------------------------------------------
# 4. TF market-block derivatives
# ---------------------------------------------------------------------
def test_tf_market_block_grad_matches_fd(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0)
    est.initialize()

    t = 0
    theta = np.concatenate([[est.E_bar_t[t]], est.njt[t]])

    grad, _ = est._tf_market_block_grad_hess(t, theta)

    eps = 1e-5
    fd = np.zeros_like(theta)

    for i in range(len(theta)):
        th_p = theta.copy()
        th_m = theta.copy()
        th_p[i] += eps
        th_m[i] -= eps

        lp_p = est._tf_market_block_logp(
            t,
            th_p,
            beta_tf=np.array(est.beta, dtype=float),
            r_tf=np.array(est.r, dtype=float),
            var_tf=np.ones(est.J),
            logvar_tf=np.zeros(est.J),
        ).numpy()

        lp_m = est._tf_market_block_logp(
            t,
            th_m,
            beta_tf=np.array(est.beta, dtype=float),
            r_tf=np.array(est.r, dtype=float),
            var_tf=np.ones(est.J),
            logvar_tf=np.zeros(est.J),
        ).numpy()

        fd[i] = (lp_p - lp_m) / (2 * eps)

    assert np.allclose(grad, fd, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------
# 5. One-step update
# ---------------------------------------------------------------------
def test_update_preserves_state(tiny_problem):
    est = LuShrinkageEstimator(**tiny_problem, seed=0, max_iter=1)
    est.initialize()

    lp0 = est._full_log_posterior()
    lp1 = est.update(lp0)

    assert np.isfinite(lp1)

    assert est.njt.shape == (est.T, est.J)
    assert est.gamma_jt.shape == (est.T, est.J)
    assert np.all((est.gamma_jt == 0) | (est.gamma_jt == 1))
    assert np.all((est.phi_t > 0) & (est.phi_t < 1))


# ---------------------------------------------------------------------
# 6. Minimal end-to-end run
# ---------------------------------------------------------------------
def test_fit_and_results(tiny_problem):
    est = LuShrinkageEstimator(
        **tiny_problem,
        seed=0,
        max_iter=5,
        burn_in=0,
        thin=1,
    )
    est.fit()

    res = est.get_results()

    assert "beta_hat" in res
    assert "sigma_hat" in res
    assert "E_hat" in res
    assert np.isfinite(res["sigma_hat"])
