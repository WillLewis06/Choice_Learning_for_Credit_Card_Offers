import numpy as np
from numpy.random import default_rng


def generate_market_conditions(T: int, J: int, dgp_type: int, seed: int):
    """
    Generate market/product primitives for Lu–Shimizu Section 4.

    Returns:
      wjt   : (T,J) exogenous characteristic
      E_bar_t : (T,) market-level shock baseline
      njt   : (T,J) product deviations
      Ejt   : (T,J) unobserved demand shocks
      ujt   : (T,J) price shock (paper: u_jt)
      alpha : (T,J) endogeneity shifter
    """
    rng = default_rng(seed)

    if dgp_type not in (1, 2, 3, 4):
        raise ValueError("dgp_type must be 1, 2, 3, or 4")

    # Observed characteristic
    wjt = rng.uniform(1.0, 2.0, size=(T, J))

    # Unobserved demand shocks
    E_bar_t = -1.0 * np.ones(T)
    njt = np.zeros((T, J))

    if dgp_type in (1, 2):  # sparse
        n_active = int(0.4 * J)
        for t in range(T):
            for j in range(n_active):
                njt[t, j] = 1.0 if j % 2 == 0 else -1.0
    else:  # non-sparse
        njt = rng.normal(0.0, 1.0 / 3.0, size=(T, J))

    Ejt = E_bar_t[:, None] + njt

    # Price equation components
    ujt = rng.normal(0.0, 0.7, size=(T, J))
    alpha = np.zeros((T, J))

    if dgp_type == 2:  # sparse endogenous
        alpha[njt == 1.0] = 0.3
        alpha[njt == -1.0] = -0.3
    elif dgp_type == 4:  # non-sparse endogenous
        thr = 1.0 / 3.0
        alpha[njt >= thr] = 0.3
        alpha[njt <= -thr] = -0.3

    return wjt, E_bar_t, njt, Ejt, ujt, alpha


class BasicLuChoiceModel:
    """
    Basic Lu–Shimizu simulation-side choice model (Section 4).
    """

    def __init__(self, N: int, beta_p: float, beta_w: float, sigma: float, seed: int):
        rng = default_rng(seed)

        self.N = int(N)
        self.beta_w = float(beta_w)
        self.beta_p_i = rng.normal(beta_p, sigma, size=self.N)

    def utilities(
        self, pjt: np.ndarray, wjt: np.ndarray, Ejt: np.ndarray
    ) -> np.ndarray:
        """
        Returns:
          uijt : (T,N,J) systematic utility
        """
        T, J = pjt.shape
        uijt = np.zeros((T, self.N, J))

        for t in range(T):
            uijt[t] = (
                self.beta_p_i[:, None] * pjt[t][None, :]
                + self.beta_w * wjt[t][None, :]
                + Ejt[t][None, :]
            )
        return uijt


def generate_market_shares(uijt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert systematic utilities into market shares by averaging choice probabilities
    across consumers (no discrete sampling).

    Inputs:
      uijt : (T, N, J) systematic utility

    Returns:
      sjt  : (T, J) inside good shares
      s0t  : (T,) outside good shares
    """
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (T, N, J)")

    # Stable logit probabilities:
    # For each (t,i), compute softmax over J goods with an outside option.
    # Outside utility is normalized to 0, so its exp is 1.
    T, N, J = uijt.shape

    # subtract max over {outside, inside} per (t,i) for numerical stability
    # outside utility is 0, so include it in the max via max(0, max_j uijt)
    m_inside = np.max(uijt, axis=2, keepdims=True)  # (T, N, 1)
    m = np.maximum(0.0, m_inside)  # (T, N, 1)

    exp_u = np.exp(uijt - m)  # (T, N, J)
    exp_out = np.exp(-m[..., 0])  # (T, N)  since outside utility is 0

    denom = exp_out + np.sum(exp_u, axis=2)  # (T, N)

    probs = exp_u / denom[..., None]  # (T, N, J)
    out_prob = exp_out / denom  # (T, N)

    sjt = np.mean(probs, axis=1)  # (T, J)
    s0t = np.mean(out_prob, axis=1)  # (T,)

    return sjt, s0t
