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
        1: sparse Ejt, exogenous price
        2: sparse Ejt, endogenous price
        3: non-sparse Ejt, exogenous price
        4: non-sparse Ejt, endogenous price

    Returns:
        pjt  : (T, J) prices
        wjt  : (T, J) exogenous characteristic
        Ejt  : (T, J) unobserved demand shocks
        jt  : (T, J) quantities sod
    """

    rng = default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Exogenous product characteristic
    # ------------------------------------------------------------------
    wjt = rng.uniform(1.0, 2.0, size=(T, J))

    # ------------------------------------------------------------------
    # 2. Unobserved demand shocks: Ejt = E_bar_t + njt
    # ------------------------------------------------------------------
    E_bar_t = -1.0 * np.ones(T)
    njt = np.zeros((T, J))

    if dgp_type in (1, 2):  # sparse
        n_active = int(0.4 * J)
        for t in range(T):
            for j in range(n_active):
                njt[t, j] = 1.0 if j % 2 == 0 else -1.0

    elif dgp_type in (3, 4):  # non-sparse
        njt = rng.normal(0.0, 1.0 / 3.0, size=(T, J))

    else:
        raise ValueError("dgp_type must be 1, 2, 3, or 4")

    Ejt = E_bar_t[:, None] + njt

    # ------------------------------------------------------------------
    # 3. Price equation
    # ------------------------------------------------------------------
    cost_shock = rng.normal(0.0, 0.7, size=(T, J))
    alpha = np.zeros((T, J))

    if dgp_type == 2:  # sparse endogenous
        alpha[njt == 1.0] = 0.3
        alpha[njt == -1.0] = -0.3

    elif dgp_type == 4:  # non-sparse endogenous
        alpha[njt >= 1.0 / 3.0] = 0.3
        alpha[njt <= -1.0 / 3.0] = -0.3

    pjt = alpha + 0.3 * wjt + cost_shock

    # ------------------------------------------------------------------
    # 4. Simulate consumers and aggregate choices
   ------------------------------------------------------------------
    qjt = np.zeros((T, J))

    # for each market
    for t in range(T):
        # Sample individual-specific price sensitivities for each consumer in market t
        beta_p_i = rng.normal(beta_p, sigma, size=N)

        # for each consumer
        for i in range(N):
            # Compute utility for each product for consumer i in market t
            utility = beta_p_i[i] * pjt[t] + beta_w * wjt[t] + Ejt[t]

            # Numerator of softmax choice probability (exp(utility) for all products)
            exp_u = np.exp(utility)
            # Denominator includes outside option utility (normalized to zero)
            denom = 1.0 + exp_u.sum()  # 1.0 = outside option
            # Choice probabilities for all products (excluding outside)
            probs = exp_u / denom

            # Simulate the consumer's choice (J+1 options: J products + outside option)
            choice = rng.choice(J + 1, p=np.append(probs, 1.0 - probs.sum()))
            if choice < J:
                # Aggregate: Add to demand count for chosen product
                qjt[t, choice] += 1

    return pjt, wjt, Ejt, qjt
