import numpy as np
from numpy.random import default_rng


def generate_dgp(
    T=25,  # number of markets
    J=15,  # products per market
    N=1000,  # consumers per market
    dgp_type=1,  # 1,2,3,4
    beta_p=-1.0,
    beta_w=0.5,
    sigma=1.5,
    seed=None,
):
    """
    Generate artificial data following Lu–Shimizu (Section 4).

    DGP types:
        1: sparse xi, exogenous price
        2: sparse xi, endogenous price
        3: non-sparse xi, exogenous price
        4: non-sparse xi, endogenous price

    Returns:
        pjt  : (T, J) prices
        wjt  : (T, J) exogenous characteristic
        xi   : (T, J) unobserved demand shocks
        qjt  : (T, J) quantities sold
    """

    rng = default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Exogenous product characteristic
    # ------------------------------------------------------------------
    wjt = rng.uniform(1.0, 2.0, size=(T, J))

    # ------------------------------------------------------------------
    # 2. Unobserved demand shocks: xi = xi_bar + eta
    # ------------------------------------------------------------------
    xi_bar = -1.0 * np.ones(T)
    eta = np.zeros((T, J))

    if dgp_type in (1, 2):  # sparse
        n_active = int(0.4 * J)
        for t in range(T):
            for j in range(n_active):
                eta[t, j] = 1.0 if j % 2 == 0 else -1.0

    elif dgp_type in (3, 4):  # non-sparse
        eta = rng.normal(0.0, 1.0 / 3.0, size=(T, J))

    else:
        raise ValueError("dgp_type must be 1, 2, 3, or 4")

    xi = xi_bar[:, None] + eta

    # ------------------------------------------------------------------
    # 3. Price equation
    # ------------------------------------------------------------------
    cost_shock = rng.normal(0.0, 0.7, size=(T, J))
    alpha = np.zeros((T, J))

    if dgp_type == 2:  # sparse endogenous
        alpha[eta == 1.0] = 0.3
        alpha[eta == -1.0] = -0.3

    elif dgp_type == 4:  # non-sparse endogenous
        alpha[eta >= 1.0 / 3.0] = 0.3
        alpha[eta <= -1.0 / 3.0] = -0.3

    pjt = alpha + 0.3 * wjt + cost_shock

    # ------------------------------------------------------------------
    # 4. Simulate consumers and aggregate choices
    # ------------------------------------------------------------------
    qjt = np.zeros((T, J))

    for t in range(T):
        beta_p_i = rng.normal(beta_p, sigma, size=N)

        for i in range(N):
            utility = beta_p_i[i] * pjt[t] + beta_w * wjt[t] + xi[t]

            exp_u = np.exp(utility)
            denom = 1.0 + exp_u.sum()  # outside option
            probs = exp_u / denom

            choice = rng.choice(J + 1, p=np.append(probs, 1.0 - probs.sum()))
            if choice < J:
                qjt[t, choice] += 1

    return pjt, wjt, xi, qjt
