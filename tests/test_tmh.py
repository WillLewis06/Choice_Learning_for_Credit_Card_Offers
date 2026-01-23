# tests/test_tmh.py
#
# Updated for Lu-style TMH (mode-anchored independence proposal).
# Replaces the old state-dependent TMH tests accordingly. :contentReference[oaicite:0]{index=0}

import numpy as np
import pytest

from market_shock_estimators.tmh import (
    TMHState,
    compute_gradient,
    compute_hessian,
    tmh_step,
    tmh_update_block,
)


# -----------------------------------------------------------------------------
# Robustness (consolidated)
# -----------------------------------------------------------------------------


def test_tmh_robustness_guards():
    """
    Verify TMH robustness under invalid inputs, non-finite posteriors,
    indefinite curvature (with regularization), and fixed-seed determinism.
    """
    rng = np.random.default_rng(123)

    # --- Subcase A: invalid theta shape (scalar / 2D) should fail loudly ---
    def logp_std_normal(theta):
        theta = np.asarray(theta, dtype=float)
        return float(-0.5 * theta @ theta)

    with pytest.raises(Exception):
        _ = tmh_step(
            TMHState(theta=np.array(1.0), logp=logp_std_normal(np.array([0.0]))),
            logp_std_normal,
            rng=rng,
        )

    with pytest.raises(Exception):
        _ = tmh_step(
            TMHState(
                theta=np.zeros((2, 2)), logp=logp_std_normal(np.array([0.0, 0.0]))
            ),
            logp_std_normal,
            rng=rng,
        )

    # --- Subcase B: non-finite log posterior regions should not corrupt state ---
    def logp_nonfinite(theta):
        theta = np.asarray(theta, dtype=float)
        r2 = float(theta @ theta)
        if r2 > 4.0:
            return float("-inf")
        if r2 > 3.0:
            return float("nan")
        return float(-0.5 * r2)

    theta0 = np.array([0.1, -0.2])
    state0 = TMHState(theta=theta0, logp=logp_nonfinite(theta0))
    new_state, _ = tmh_step(
        state0, logp_nonfinite, rng=np.random.default_rng(321), ridge=1e-3
    )

    assert np.all(np.isfinite(new_state.theta))
    assert np.isfinite(new_state.logp)

    # --- Subcase C: indefinite curvature handled via regularization ---
    def logp_indefinite(theta):
        theta = np.asarray(theta, dtype=float)
        t0, t1 = float(theta[0]), float(theta[1])
        return float(-0.5 * t0 * t0 + 0.5 * t1 * t1 - 0.25 * (t1**4))

    theta0 = np.array([0.1, 0.1])
    state0 = TMHState(theta=theta0, logp=logp_indefinite(theta0))
    new_state, _ = tmh_step(
        state0, logp_indefinite, rng=np.random.default_rng(999), ridge=10.0
    )

    assert np.all(np.isfinite(new_state.theta))
    assert np.isfinite(new_state.logp)

    # --- Subcase D: determinism under fixed seed ---
    def logp_small_gauss(theta):
        theta = np.asarray(theta, dtype=float)
        return float(-0.5 * theta @ theta)

    theta0 = np.array([0.3, -0.4, 0.1])
    state0 = TMHState(theta=theta0, logp=logp_small_gauss(theta0))

    rng1 = np.random.default_rng(2024)
    rng2 = np.random.default_rng(2024)

    s1, a1 = tmh_step(state0, logp_small_gauss, rng=rng1, ridge=1e-6)
    s2, a2 = tmh_step(state0, logp_small_gauss, rng=rng2, ridge=1e-6)

    assert a1 == a2
    assert np.allclose(s1.theta, s2.theta)
    assert np.isclose(s1.logp, s2.logp)


# -----------------------------------------------------------------------------
# Core correctness tests
# -----------------------------------------------------------------------------


def test_compute_gradient_quadratic():
    """
    Check numerical gradient against the analytic gradient for
    logp(theta) = -0.5 * theta' A theta.
    """
    d = 4
    A = np.diag(np.arange(1, d + 1, dtype=float))
    theta = np.array([0.2, -0.1, 0.3, 0.05], dtype=float)

    def logp(theta_):
        theta_ = np.asarray(theta_, dtype=float)
        return float(-0.5 * theta_ @ A @ theta_)

    grad_num = compute_gradient(theta, logp, eps=1e-6)
    grad_true = -(A @ theta)

    assert grad_num.shape == (d,)
    assert np.max(np.abs(grad_num - grad_true)) < 5e-5


def test_compute_hessian_quadratic():
    """
    Check numerical Hessian against the analytic Hessian for
    logp(theta) = -0.5 * theta' A theta, whose Hessian is -A.
    """
    d = 3
    A = np.array(
        [[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.2]],
        dtype=float,
    )
    theta = np.array([0.1, -0.2, 0.05], dtype=float)

    def logp(theta_):
        theta_ = np.asarray(theta_, dtype=float)
        return float(-0.5 * theta_ @ A @ theta_)

    H_num = compute_hessian(theta, logp, eps=1e-4)

    assert H_num.shape == (d, d)
    assert np.max(np.abs(H_num - H_num.T)) < 1e-6
    assert np.max(np.abs(H_num - (-A))) < 5e-3


def test_tmh_stationary_gaussian():
    """
    Verify TMH targets N(0, I) on a standard Gaussian log posterior.

    To keep this test stable and fast, use analytic derivatives and set kappa=1,
    so the independence proposal matches the target Laplace approximation.
    """
    d = 3

    def logp(theta):
        theta = np.asarray(theta, dtype=float)
        return float(-0.5 * theta @ theta)

    def grad(theta):
        theta = np.asarray(theta, dtype=float)
        return -theta

    def hess(theta):
        theta = np.asarray(theta, dtype=float)
        return -np.eye(theta.size, dtype=float)

    rng = np.random.default_rng(7)

    state = TMHState(
        theta=np.array([1.0, -1.0, 0.5], dtype=float),
        logp=logp(np.array([1.0, -1.0, 0.5])),
    )

    n_total = 2500
    burn = 500
    samples = []

    for it in range(n_total):
        state, _ = tmh_step(
            state,
            logp,
            grad_log_posterior=grad,
            hess_log_posterior=hess,
            rng=rng,
            kappa=1.0,
            ridge=1e-10,
        )
        if it >= burn:
            samples.append(state.theta.copy())

    X = np.asarray(samples)
    mean = X.mean(axis=0)
    cov = np.cov(X.T, bias=False)

    assert np.max(np.abs(mean)) < 0.15
    assert np.max(np.abs(np.diag(cov) - 1.0)) < 0.20
    offdiag = cov - np.diag(np.diag(cov))
    assert np.max(np.abs(offdiag)) < 0.15


def test_tmh_update_block_isolation():
    """
    Ensure tmh_update_block updates only the specified parameter block and
    leaves non-block components unchanged.
    """
    x = np.array([0.5, -0.2, 0.1, -0.3, 0.4], dtype=float)
    block_idx = np.array([0, 2], dtype=int)

    def full_logp():
        return float(-0.5 * x @ x)

    def get_block():
        return x[block_idx].copy(), full_logp()

    def set_block(theta_block_new):
        x[block_idx] = np.asarray(theta_block_new, dtype=float)

    rng = np.random.default_rng(11)
    x_before = x.copy()

    accepted = tmh_update_block(
        get_block,
        set_block,
        full_logp,
        rng,
        ridge=1e-6,
        kappa=1.0,
    )
    x_after = x.copy()

    non_block = np.array(
        [i for i in range(x.size) if i not in set(block_idx)], dtype=int
    )
    assert np.allclose(x_after[non_block], x_before[non_block])

    if not accepted:
        assert np.allclose(x_after, x_before)
