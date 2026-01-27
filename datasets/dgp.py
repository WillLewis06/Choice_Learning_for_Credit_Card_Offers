import numpy as np


def generate_market_conditions(T: int, J: int, dgp_type: int, seed: int):
    """
    Generate market/product primitives for Lu–Shimizu (2025), Section 4.1.

    Notation mapping to the paper:
      wjt    : exogenous characteristic w_jt ~ U(1,2)
      E_bar_t: market intercept xi_bar*_t, fixed at -1
      njt    : eta*_jt (market-product deviations)
      Ejt    : xi*_jt = xi_bar*_t + eta*_jt
      ujt    : cost shock u_jt ~ N(0, 0.7^2)
      alpha  : alpha*_jt (endogeneity shifter in price equation)

    DGP designs (paper):
      DGP1: sparse eta, exogenous price (alpha=0)
      DGP2: sparse eta, endogenous price (alpha depends on eta)
      DGP3: non-sparse eta ~ N(0,(1/3)^2), exogenous price (alpha=0)
      DGP4: non-sparse eta, endogenous price (alpha depends on eta threshold)

    Args:
      T, J     : number of markets and products
      dgp_type : 1..4
      seed     : integer seed (function owns its RNG)

    Returns:
      wjt     : (T,J)
      E_bar_t : (T,)
      njt     : (T,J)
      Ejt     : (T,J)
      ujt     : (T,J)
      alpha   : (T,J)
    """
    if dgp_type not in (1, 2, 3, 4):
        raise ValueError("dgp_type must be 1, 2, 3, or 4")

    rng = np.random.default_rng(int(seed))

    # Exogenous product characteristic: wjt ~ U(1,2)
    wjt = rng.uniform(1.0, 2.0, size=(T, J))

    # Market-level intercept: xi_bar*_t fixed at -1 for all t
    E_bar_t = -1.0 * np.ones(T, dtype=float)

    # Market-product deviations: eta*_jt
    if dgp_type in (1, 2):  # sparse eta (deterministic pattern)
        njt = np.zeros((T, J), dtype=float)
        n_active = int(0.4 * J)
        for t in range(T):
            for j in range(n_active):
                # With 0-based indexing: j=0 -> +1, j=1 -> -1, ...
                njt[t, j] = 1.0 if (j % 2 == 0) else -1.0
    else:  # non-sparse eta ~ N(0, (1/3)^2)
        njt = rng.normal(0.0, 1.0 / 3.0, size=(T, J))

    # Unobserved demand shock: xi*_jt = xi_bar*_t + eta*_jt
    Ejt = E_bar_t[:, None] + njt

    # Cost shock in price equation: ujt ~ N(0, 0.7^2)
    ujt = rng.normal(0.0, 0.7, size=(T, J))

    # Endogeneity shifter alpha*_jt (depends on DGP)
    alpha = np.zeros((T, J), dtype=float)
    if dgp_type == 2:
        # DGP2: alpha*_jt = 0.3 if eta=1, -0.3 if eta=-1, 0 otherwise
        alpha[njt == 1.0] = 0.3
        alpha[njt == -1.0] = -0.3
    elif dgp_type == 4:
        # DGP4: alpha*_jt = 0.3 if eta>=1/3, -0.3 if eta<=-1/3, 0 otherwise
        thr = 1.0 / 3.0
        alpha[njt >= thr] = 0.3
        alpha[njt <= -thr] = -0.3

    return wjt, Ejt, ujt, alpha


class BasicLuChoiceModel:
    """
    Simulation-side RC logit choice model used in Lu–Shimizu Section 4.1:

      u_ijt = beta_{p,i} * p_jt + beta_w * w_jt + xi_jt
      beta_{p,i} ~ N(beta_p, sigma^2)

    eps_ijt is implicit (logit choice probabilities).
    """

    def __init__(
        self,
        N: int,
        beta_p: float,
        beta_w: float,
        sigma: float,
        seed: int,
    ):
        self.N = int(N)
        self.beta_w = float(beta_w)

        rng = np.random.default_rng(int(seed))
        self.beta_p_i = rng.normal(float(beta_p), float(sigma), size=self.N)

    def utilities(
        self, pjt: np.ndarray, wjt: np.ndarray, Ejt: np.ndarray
    ) -> np.ndarray:
        """
        Compute systematic utilities (excluding Gumbel eps).

        Args:
          pjt : (T,J)
          wjt : (T,J)
          Ejt : (T,J)

        Returns:
          uijt : (T, N, J)
        """
        pjt = np.asarray(pjt, dtype=float)
        wjt = np.asarray(wjt, dtype=float)
        Ejt = np.asarray(Ejt, dtype=float)

        if pjt.ndim != 2 or wjt.ndim != 2 or Ejt.ndim != 2:
            raise ValueError("pjt, wjt, Ejt must be 2D arrays of shape (T,J).")
        if pjt.shape != wjt.shape or pjt.shape != Ejt.shape:
            raise ValueError("pjt, wjt, Ejt must have the same shape (T,J).")

        T, J = pjt.shape
        uijt = np.zeros((T, self.N, J), dtype=float)
        for t in range(T):
            uijt[t] = (
                self.beta_p_i[:, None] * pjt[t][None, :]
                + self.beta_w * wjt[t][None, :]
                + Ejt[t][None, :]
            )
        return uijt


def _generate_market_shares(uijt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert systematic utilities into expected shares by averaging logit choice
    probabilities across simulated consumers.

    Args:
      uijt : (T, N, J)

    Returns:
      sjt  : (T, J)
      s0t  : (T,)
    """
    uijt = np.asarray(uijt, dtype=float)
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (T, N, J)")

    # Stabilize softmax with outside option utility normalized to 0.
    m_inside = np.max(uijt, axis=2, keepdims=True)  # (T, N, 1)
    m = np.maximum(0.0, m_inside)  # include outside option in max

    exp_u = np.exp(uijt - m)  # (T, N, J)
    exp_out = np.exp(-m[..., 0])  # (T, N)

    denom = exp_out + np.sum(exp_u, axis=2)  # (T, N)

    probs = exp_u / denom[..., None]  # (T, N, J)
    out_prob = exp_out / denom  # (T, N)

    sjt = np.mean(probs, axis=1)  # (T, J)
    s0t = np.mean(out_prob, axis=1)  # (T,)

    return sjt, s0t


def generate_market(
    uijt: np.ndarray, N: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate expected shares and multinomial counts from simulated utilities.

    Args:
      uijt : (T, N_sim, J)
      N    : number of consumers (multinomial trials)
      seed : integer seed (function owns its RNG)

    Returns:
      sjt : (T,J) expected shares
      s0t : (T,) expected outside share
      qjt : (T,J) multinomial counts for inside goods
      q0t : (T,) multinomial counts for outside good
    """
    uijt = np.asarray(uijt, dtype=float)
    if uijt.ndim != 3:
        raise ValueError("uijt must have shape (T, N_sim, J).")

    rng = np.random.default_rng(int(seed))

    sjt, s0t = _generate_market_shares(uijt)

    T, J = sjt.shape
    qjt = np.zeros((T, J), dtype=int)
    q0t = np.zeros(T, dtype=int)

    for t in range(T):
        probs = np.concatenate([[s0t[t]], sjt[t]])
        probs = probs / probs.sum()
        draw = rng.multinomial(int(N), probs)
        q0t[t] = int(draw[0])
        qjt[t, :] = draw[1:]

    return sjt, s0t, qjt, q0t
