# bonus2_conftest.py
"""
Shared Bonus2 test builders.

This file is intentionally not a pytest conftest module and does not define pytest
fixtures. Tests should import and call these helpers directly.

Key conventions (Bonus2):
  y_mit:         (M, N, T) int32 choices; 0=outside, c=j+1 for inside product j (j=0..J-1)
  delta_mj:      (M, J) float64 phase-1 baseline utilities
  is_weekend_t:  (T,) int32 in {0,1}
  season_sin_kt: (K, T) float64 seasonal basis
  season_cos_kt: (K, T) float64 seasonal basis
  neighbors_m:   neighbors_m[m][i] -> list[int] (within-market, indices in 0..N-1)
  lookback:      scalar int32 >= 1
  decay:         scalar float64 in (0,1)

Config scalars (init_theta):
  beta_intercept, beta_habit, beta_peer,
  beta_weekend_weekday, beta_weekend_weekend,
  a_m, b_m

z-block keys (sigmas / step_size_z):
  z_beta_intercept_j, z_beta_habit_j, z_beta_peer_j, z_beta_weekend_jw, z_a_m, z_b_m
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

# Must be set before importing TensorFlow to suppress most C++ logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402


# =============================================================================
# Logging / RNG
# =============================================================================


def quiet_tf_logger() -> None:
    """Reduce TensorFlow python logging (separate from TF_CPP_MIN_LOG_LEVEL)."""
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        return


quiet_tf_logger()


def seed_everything(seed: int) -> None:
    """Set NumPy and TensorFlow RNG seeds for deterministic test runs."""
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


# =============================================================================
# Canonical small dimensions / hyperparameters
# =============================================================================


def tiny_dims() -> dict[str, int]:
    """
    Small non-degenerate dimensions used across Bonus2 tests.

    Returns:
      dict with keys: M, N, J, T, K
    """
    return {"M": 2, "N": 4, "J": 3, "T": 7, "K": 1}


def tiny_dims_k0() -> dict[str, int]:
    """Variant with K=0 to test the no-seasonality edge case."""
    d = tiny_dims()
    d["K"] = 0
    return d


def tiny_hyperparams() -> dict[str, Any]:
    """
    Small, valid hyperparameters for Bonus2.

    Returns:
      dict with keys: lookback, decay, season_period, eps
    """
    return {"lookback": 2, "decay": 0.85, "season_period": 365, "eps": 1.0e-12}


# =============================================================================
# Panel builders (NumPy)
# =============================================================================


def y_mit_np(dims: dict[str, int], pattern: str = "mixed") -> np.ndarray:
    """
    Build a deterministic (M,N,T) choice tensor with values in {0..J}.

    Encoding:
      0 = outside option
      c = j+1 for inside product j (j=0..J-1), so inside codes are 1..J.

    Patterns:
      - "mixed": uses a modular pattern producing both outside and inside choices
      - "all_outside": all zeros
      - "single_product": always chooses product code 1
      - "alternating": alternates across inside products with occasional outside
    """
    M, N, T, J = int(dims["M"]), int(dims["N"]), int(dims["T"]), int(dims["J"])

    if pattern == "all_outside":
        return np.zeros((M, N, T), dtype=np.int32)

    if pattern == "single_product":
        return np.ones((M, N, T), dtype=np.int32)  # code 1 => product j=0

    y = np.empty((M, N, T), dtype=np.int32)

    if pattern == "alternating":
        for m in range(M):
            for i in range(N):
                for t in range(T):
                    if (t % 5) == 0:
                        y[m, i, t] = 0
                    else:
                        y[m, i, t] = 1 + ((t + i + 2 * m) % J)
        return y

    if pattern != "mixed":
        raise ValueError(f"y_mit_np: unknown pattern '{pattern}'")

    # Mixed: deterministic pattern over {0..J} ensuring variety.
    for m in range(M):
        for i in range(N):
            base = m * 37 + i * 17
            for t in range(T):
                y[m, i, t] = int((base + t) % (J + 1))
    return y


def delta_mj_np(dims: dict[str, int]) -> np.ndarray:
    """Build a non-degenerate (M,J) baseline utility matrix."""
    M, J = int(dims["M"]), int(dims["J"])
    m_term = np.linspace(-0.1, 0.1, M, dtype=np.float64)[:, None]  # (M,1)
    j_term = np.linspace(0.2, 0.6, J, dtype=np.float64)[None, :]  # (1,J)
    return (m_term + j_term).astype(np.float64)


def is_weekend_t_np(dims: dict[str, int], pattern: str = "0101") -> np.ndarray:
    """
    Build (T,) int indicator in {0,1}.

    Patterns:
      - "0101": alternating 0/1
      - "weekend_blocks": two blocks (first half 0, second half 1)
    """
    T = int(dims["T"])
    if T <= 0:
        raise ValueError("is_weekend_t_np: T must be >= 1")

    if pattern == "0101":
        return (np.arange(T, dtype=np.int32) % 2).astype(np.int32)

    if pattern == "weekend_blocks":
        w = np.zeros((T,), dtype=np.int32)
        w[T // 2 :] = 1
        return w

    raise ValueError(f"is_weekend_t_np: unknown pattern '{pattern}'")


def season_features_np(
    dims: dict[str, int],
    season_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build seasonal basis matrices (K,T) using a simple harmonic construction.

    For k=1..K and t=0..T-1:
      angle_t = 2*pi*t/season_period
      sin[k-1,t] = sin(k*angle_t)
      cos[k-1,t] = cos(k*angle_t)

    If K=0, returns arrays of shape (0,T).
    """
    K, T = int(dims["K"]), int(dims["T"])
    P = int(season_period)
    if P < 1:
        raise ValueError("season_features_np: season_period must be >= 1")

    if K == 0:
        return (
            np.zeros((0, T), dtype=np.float64),
            np.zeros((0, T), dtype=np.float64),
        )

    t = np.arange(T, dtype=np.float64)
    angle = (2.0 * np.pi / float(P)) * t  # (T,)
    sin = np.empty((K, T), dtype=np.float64)
    cos = np.empty((K, T), dtype=np.float64)

    for k in range(1, K + 1):
        sin[k - 1, :] = np.sin(float(k) * angle)
        cos[k - 1, :] = np.cos(float(k) * angle)

    return sin.astype(np.float64), cos.astype(np.float64)


def neighbors_m_np(
    dims: dict[str, int], pattern: str = "ring"
) -> list[list[list[int]]]:
    """
    Build neighbors_m with shape [M][N][deg_i].

    Patterns:
      - "empty": all empty lists
      - "ring": each i has neighbor (i+1) mod N
      - "asymmetric": varying degrees with at least one empty list
    """
    M, N = int(dims["M"]), int(dims["N"])
    neighbors_m: list[list[list[int]]] = []

    for m in range(M):
        per_market: list[list[int]] = []
        if pattern == "empty":
            for i in range(N):
                per_market.append([])
            neighbors_m.append(per_market)
            continue

        if pattern == "ring":
            for i in range(N):
                per_market.append([int((i + 1) % N)])
            neighbors_m.append(per_market)
            continue

        if pattern == "asymmetric":
            for i in range(N):
                if i == 0:
                    per_market.append([])  # at least one isolated node
                elif i == 1:
                    per_market.append([0])  # single edge
                else:
                    per_market.append([int((i - 1) % N), int((i + 1) % N)])  # degree 2
            neighbors_m.append(per_market)
            continue

        raise ValueError(f"neighbors_m_np: unknown pattern '{pattern}'")

    # Ensure no self-edges / duplicates (defensive for future edits).
    for m in range(M):
        for i in range(N):
            lst = neighbors_m[m][i]
            if i in lst:
                raise ValueError("neighbors_m_np: self-edge generated")
            if len(set(lst)) != len(lst):
                raise ValueError("neighbors_m_np: duplicate neighbor generated")

    return neighbors_m


def panel_np(
    dims: dict[str, int],
    hyper: dict[str, Any] | None = None,
    y_pattern: str = "mixed",
    neighbor_pattern: str = "ring",
    weekend_pattern: str = "0101",
) -> dict[str, Any]:
    """
    Build a canonical Bonus2 panel dict matching bonus2_input_validation.validate_bonus2_panel.

    Returns dict with keys:
      y_mit, delta_mj, is_weekend_t, season_sin_kt, season_cos_kt,
      neighbors_m, lookback, decay
    """
    h = tiny_hyperparams() if hyper is None else dict(hyper)

    y = y_mit_np(dims=dims, pattern=y_pattern)
    delta = delta_mj_np(dims=dims)
    w = is_weekend_t_np(dims=dims, pattern=weekend_pattern)

    season_sin, season_cos = season_features_np(
        dims=dims,
        season_period=int(h["season_period"]),
    )

    neighbors_m = neighbors_m_np(dims=dims, pattern=neighbor_pattern)

    return {
        "y_mit": y,
        "delta_mj": delta,
        "is_weekend_t": w,
        "season_sin_kt": season_sin,
        "season_cos_kt": season_cos,
        "neighbors_m": neighbors_m,
        "lookback": int(h["lookback"]),
        "decay": float(h["decay"]),
    }


# =============================================================================
# Config builders (init_theta / sigmas / step sizes)
# =============================================================================


def init_theta_scalars(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """
    Build scalar init_theta dict required by Bonus2Estimator.

    Required keys:
      beta_intercept, beta_habit, beta_peer,
      beta_weekend_weekday, beta_weekend_weekend,
      a_m, b_m
    """
    base = {
        "beta_intercept": 0.0,
        "beta_habit": 0.0,
        "beta_peer": 0.0,
        "beta_weekend_weekday": 0.0,
        "beta_weekend_weekend": 0.0,
        "a_m": 0.0,
        "b_m": 0.0,
    }
    if overrides:
        for k, v in overrides.items():
            if k not in base:
                raise ValueError(f"init_theta_scalars: unexpected key '{k}'")
            base[k] = float(v)
    return base


def sigmas_z(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Build positive per-block prior scales keyed by z-block names."""
    base = {
        "z_beta_intercept_j": 1.0,
        "z_beta_habit_j": 1.0,
        "z_beta_peer_j": 1.0,
        "z_beta_weekend_jw": 1.0,
        "z_a_m": 1.0,
        "z_b_m": 1.0,
    }
    if overrides:
        for k, v in overrides.items():
            if k not in base:
                raise ValueError(f"sigmas_z: unexpected key '{k}'")
            base[k] = float(v)
    return base


def step_size_z(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Build positive per-block RW proposal scales keyed by z-block names."""
    base = {
        "z_beta_intercept_j": 0.05,
        "z_beta_habit_j": 0.05,
        "z_beta_peer_j": 0.05,
        "z_beta_weekend_jw": 0.05,
        "z_a_m": 0.05,
        "z_b_m": 0.05,
    }
    if overrides:
        for k, v in overrides.items():
            if k not in base:
                raise ValueError(f"step_size_z: unexpected key '{k}'")
            base[k] = float(v)
    return base


# =============================================================================
# z/theta builders (NumPy)
# =============================================================================


def z_blocks_np(dims: dict[str, int], fill: float = 0.0) -> dict[str, np.ndarray]:
    """
    Build unconstrained z-blocks with correct shapes.

    Shapes:
      z_beta_intercept_j: (J,)
      z_beta_habit_j:     (J,)
      z_beta_peer_j:      (J,)
      z_beta_weekend_jw:  (J,2)
      z_a_m:              (M,K)
      z_b_m:              (M,K)
    """
    M, J, K = int(dims["M"]), int(dims["J"]), int(dims["K"])
    f = float(fill)

    return {
        "z_beta_intercept_j": np.full((J,), f, dtype=np.float64),
        "z_beta_habit_j": np.full((J,), f, dtype=np.float64),
        "z_beta_peer_j": np.full((J,), f, dtype=np.float64),
        "z_beta_weekend_jw": np.full((J, 2), f, dtype=np.float64),
        "z_a_m": np.full((M, K), f, dtype=np.float64),
        "z_b_m": np.full((M, K), f, dtype=np.float64),
    }


def theta_true_np(
    dims: dict[str, int], pattern: str = "nonzero"
) -> dict[str, np.ndarray]:
    """
    Build a structurally-valid theta dict for predictive/recovery tests.

    Keys and shapes:
      beta_intercept_j:  (J,)
      beta_habit_j:      (J,)
      beta_peer_j:       (J,)
      beta_weekend_jw:   (J,2)
      a_m:               (M,K)
      b_m:               (M,K)
    """
    M, J, K = int(dims["M"]), int(dims["J"]), int(dims["K"])

    if pattern == "zeros":
        return {
            "beta_intercept_j": np.zeros((J,), dtype=np.float64),
            "beta_habit_j": np.zeros((J,), dtype=np.float64),
            "beta_peer_j": np.zeros((J,), dtype=np.float64),
            "beta_weekend_jw": np.zeros((J, 2), dtype=np.float64),
            "a_m": np.zeros((M, K), dtype=np.float64),
            "b_m": np.zeros((M, K), dtype=np.float64),
        }

    if pattern != "nonzero":
        raise ValueError(f"theta_true_np: unknown pattern '{pattern}'")

    beta_intercept = np.linspace(-0.1, 0.2, J, dtype=np.float64)
    beta_habit = np.linspace(0.3, 0.6, J, dtype=np.float64)
    beta_peer = np.linspace(0.05, 0.15, J, dtype=np.float64)

    beta_weekend = np.empty((J, 2), dtype=np.float64)
    beta_weekend[:, 0] = np.linspace(-0.05, 0.05, J, dtype=np.float64)  # weekday
    beta_weekend[:, 1] = beta_weekend[:, 0] + 0.10  # weekend lift

    a_m = np.full((M, K), -0.02, dtype=np.float64)
    b_m = np.full((M, K), 0.03, dtype=np.float64)

    return {
        "beta_intercept_j": beta_intercept,
        "beta_habit_j": beta_habit,
        "beta_peer_j": beta_peer,
        "beta_weekend_jw": beta_weekend,
        "a_m": a_m,
        "b_m": b_m,
    }


# =============================================================================
# TF input assembly helpers
# =============================================================================


def posterior_inputs_tf(panel: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a canonical panel dict into PosteriorInputs tensors.

    Returns dict with keys:
      y_mit, delta_mj, is_weekend_t, season_sin_kt, season_cos_kt,
      peer_adj_m, lookback, decay
    """
    y_np = np.asarray(panel["y_mit"], dtype=np.int32)
    delta_np = np.asarray(panel["delta_mj"], dtype=np.float64)
    w_np = np.asarray(panel["is_weekend_t"], dtype=np.int32)
    sin_np = np.asarray(panel["season_sin_kt"], dtype=np.float64)
    cos_np = np.asarray(panel["season_cos_kt"], dtype=np.float64)

    M, N, _T = (int(x) for x in y_np.shape)

    # Import locally so tests can import this file without requiring package wiring
    # until they call this function.
    from bonus2 import bonus2_model as bonus2_model  # noqa: WPS433

    peer_adj_m = bonus2_model.build_peer_adjacency(
        neighbors_m=panel["neighbors_m"],
        n_consumers=N,
    )

    return {
        "y_mit": tf.convert_to_tensor(y_np, dtype=tf.int32),
        "delta_mj": tf.convert_to_tensor(delta_np, dtype=tf.float64),
        "is_weekend_t": tf.convert_to_tensor(w_np, dtype=tf.int32),
        "season_sin_kt": tf.convert_to_tensor(sin_np, dtype=tf.float64),
        "season_cos_kt": tf.convert_to_tensor(cos_np, dtype=tf.float64),
        "peer_adj_m": peer_adj_m,
        "lookback": tf.convert_to_tensor(int(panel["lookback"]), dtype=tf.int32),
        "decay": tf.convert_to_tensor(float(panel["decay"]), dtype=tf.float64),
    }


# =============================================================================
# Reference (NumPy) calculators for expected states
# =============================================================================


def inside_choice_onehot_np(y_mit: np.ndarray, J: int) -> np.ndarray:
    """
    Inside-choice indicator x (M,N,T,J) with x[...,t,j]=1{y[...,t]==j+1}.
    """
    y = np.asarray(y_mit, dtype=np.int32)
    M, N, T = (int(x) for x in y.shape)
    x_mntj = np.zeros((M, N, T, int(J)), dtype=np.float64)
    for m in range(M):
        for i in range(N):
            for t in range(T):
                c = int(y[m, i, t])
                if c > 0:
                    j = c - 1
                    if 0 <= j < J:
                        x_mntj[m, i, t, j] = 1.0
    return x_mntj


def expected_habit_stock_pre_choice_np(
    y_mit: np.ndarray,
    J: int,
    decay: float,
) -> np.ndarray:
    """
    Reference implementation of pre-choice habit stock H_t.

    Recurrence per (m,i,j):
      H_{t+1} = decay * H_t + x_t
    Output is H_t for t=0..T-1 (pre-choice alignment).
    """
    y = np.asarray(y_mit, dtype=np.int32)
    M, N, T = (int(x) for x in y.shape)
    x_mntj = inside_choice_onehot_np(y_mit=y, J=int(J))

    H = np.zeros((M, N, int(J)), dtype=np.float64)  # H_0
    out = np.zeros((M, N, T, int(J)), dtype=np.float64)

    d = float(decay)
    for t in range(T):
        out[:, :, t, :] = H
        H = d * H + x_mntj[:, :, t, :]

    return out


def expected_peer_exposure_np(
    y_mit: np.ndarray,
    neighbors_m: Sequence[Sequence[Sequence[int]]],
    J: int,
    lookback: int,
) -> np.ndarray:
    """
    Reference implementation of peer exposure P (M,N,T,J).

    For each market m, consumer i, product j, time t:
      P[m,i,t,j] = sum_{k in N_m(i)} sum_{ell=1..L} 1{ y[m,k,t-ell] == j+1 }
    """
    y = np.asarray(y_mit, dtype=np.int32)
    M, N, T = (int(x) for x in y.shape)

    x_mntj = inside_choice_onehot_np(y_mit=y, J=int(J))
    L = int(lookback)

    P = np.zeros((M, N, T, int(J)), dtype=np.float64)

    for m in range(M):
        for i in range(N):
            neigh = list(neighbors_m[m][i])
            for t in range(T):
                t0 = max(0, t - L)
                if t0 >= t or len(neigh) == 0:
                    continue
                for k in neigh:
                    kk = int(k)
                    P[m, i, t, :] += x_mntj[m, kk, t0:t, :].sum(axis=0)
    return P


def expected_market_seasonality_np(
    a_mk: np.ndarray,
    b_mk: np.ndarray,
    season_sin_kt: np.ndarray,
    season_cos_kt: np.ndarray,
) -> np.ndarray:
    """
    Reference implementation of market seasonality:
      S_mt = a_mk @ sin_kt + b_mk @ cos_kt
    """
    a = np.asarray(a_mk, dtype=np.float64)
    b = np.asarray(b_mk, dtype=np.float64)
    sin = np.asarray(season_sin_kt, dtype=np.float64)
    cos = np.asarray(season_cos_kt, dtype=np.float64)
    return (a @ sin + b @ cos).astype(np.float64)


# =============================================================================
# Toy cases (small manual check panels)
# =============================================================================


def toy_peer_case() -> dict[str, Any]:
    """
    A tiny panel where peer exposure can be checked by inspection.

    M=1, N=3, J=2, T=5, K=0.
    neighbors: 0 <- {1,2}, 1 <- {}, 2 <- {1}
    """
    dims = {"M": 1, "N": 3, "J": 2, "T": 5, "K": 0}
    hyper = {"lookback": 2, "decay": 0.85, "season_period": 365, "eps": 1.0e-12}

    y = np.asarray(
        [
            [
                [0, 1, 0, 2, 0],  # i=0
                [1, 0, 2, 0, 1],  # i=1
                [2, 2, 0, 0, 0],  # i=2
            ]
        ],
        dtype=np.int32,
    )  # (1,3,5)

    neighbors_m = [
        [
            [1, 2],  # i=0
            [],  # i=1
            [1],  # i=2
        ]
    ]

    season_sin = np.zeros((0, dims["T"]), dtype=np.float64)
    season_cos = np.zeros((0, dims["T"]), dtype=np.float64)

    return {
        "y_mit": y,
        "delta_mj": delta_mj_np(dims),
        "is_weekend_t": is_weekend_t_np(dims, pattern="0101"),
        "season_sin_kt": season_sin,
        "season_cos_kt": season_cos,
        "neighbors_m": neighbors_m,
        "lookback": int(hyper["lookback"]),
        "decay": float(hyper["decay"]),
    }


def toy_habit_case() -> dict[str, Any]:
    """
    A tiny panel where habit stock timing (pre-choice) can be checked by inspection.

    M=1, N=1, J=1, T=5, K=0.
    y: [1,1,0,1,0]
    """
    dims = {"M": 1, "N": 1, "J": 1, "T": 5, "K": 0}
    hyper = {"lookback": 2, "decay": 0.80, "season_period": 365, "eps": 1.0e-12}

    y = np.asarray([[[1, 1, 0, 1, 0]]], dtype=np.int32)  # (1,1,5)
    neighbors_m = [[[]]]  # no peers

    season_sin = np.zeros((0, dims["T"]), dtype=np.float64)
    season_cos = np.zeros((0, dims["T"]), dtype=np.float64)

    return {
        "y_mit": y,
        "delta_mj": delta_mj_np(dims),
        "is_weekend_t": is_weekend_t_np(dims, pattern="0101"),
        "season_sin_kt": season_sin,
        "season_cos_kt": season_cos,
        "neighbors_m": neighbors_m,
        "lookback": int(hyper["lookback"]),
        "decay": float(hyper["decay"]),
    }
