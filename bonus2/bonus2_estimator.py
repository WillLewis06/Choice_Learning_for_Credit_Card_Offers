"""
bonus2/bonus2_estimator.py

Bonus Q2 (habit + peer + DOW + seasonality) estimator, following the Ching-style
RW-MH "orchestration + per-block update functions" architecture.

Key conventions (must match bonus2_model.py and bonus2_posterior.py):
  - Seasonal features are (K,T): sin_k_theta[k,t], cos_k_theta[k,t]
  - Network is passed as peer_adj_m: tuple/list length M of tf.SparseTensor (N,N)
  - Unconstrained sampler blocks are keyed by:

      z_beta_market_mj  (M,J)
      z_beta_habit_j    (J,)
      z_beta_peer_j     (J,)
      z_decay_rate_j    (J,)     -> constrained to (0,1) by sigmoid in model
      z_beta_dow_m      (M,7)
      z_beta_dow_j      (J,7)
      z_a_m             (M,K)
      z_b_m             (M,K)
      z_a_j             (J,K)
      z_b_j             (J,K)

Decay prior hyperparameter:
  - kappa_decay (scalar float64) is stored in inputs and used by the posterior
    for the Beta(kappa_decay, 1) prior on decay_rate_j.

Public API:
  - fit(n_iter, k): k is a dict of RW step sizes keyed by:
      {"beta_market","beta_habit","beta_peer","decay_rate",
       "beta_dow_m","beta_dow_j","a_m","b_m","a_j","b_j"}
  - get_results(): returns theta_hat (LAST draw), n_saved, accept (rates)
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import tensorflow as tf

from bonus2 import bonus2_model as model

from bonus2.bonus2_updates import (
    update_z_a_j,
    update_z_a_m,
    update_z_b_j,
    update_z_b_m,
    update_z_beta_dow_j,
    update_z_beta_dow_m,
    update_z_beta_habit_j,
    update_z_beta_market_mj,
    update_z_beta_peer_j,
    update_z_decay_rate_j,
)

try:
    from bonus2.bonus2_diagnostics import report_iteration_progress
except Exception:  # pragma: no cover

    def report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
        # Minimal fallback (kept TF-safe for @tf.function call sites).
        means = []
        for k in sorted(z.keys()):
            x = tf.cast(z[k], tf.float64)
            means.append(tf.reduce_mean(x))
        tf.print("[Bonus2] it=", it, "| mean(z)=", means)


def _neighbors_to_peer_adj_m(
    neighbors: Any, M: int, N: int
) -> tuple[tf.SparseTensor, ...]:
    """
    Convert neighbor lists into per-market Sparse adjacency matrices.

    Expected input shape: neighbors[m][i] is an iterable of neighbor indices k
    (no cross-market edges). Self-edges are dropped if present.

    Returns:
      peer_adj_m: tuple length M of SparseTensor (N,N) with 1.0 entries.
    """
    # Convert to nested Python lists early (one-time cost, estimator init only).
    nb = neighbors
    if isinstance(nb, tf.RaggedTensor):
        nb = nb.to_list()
    elif isinstance(nb, np.ndarray):
        nb = nb.tolist()

    if not isinstance(nb, (list, tuple)) or len(nb) != M:
        raise ValueError("neighbors must be a nested list/tuple with outer length M")

    peer_adj: list[tf.SparseTensor] = []
    for m in range(M):
        rows = nb[m]
        if not isinstance(rows, (list, tuple)) or len(rows) != N:
            raise ValueError("neighbors[m] must have length N for each market m")

        indices: list[list[int]] = []
        for i in range(N):
            neigh_i = rows[i]
            if neigh_i is None:
                continue
            # Make robust to ragged-like inputs.
            if isinstance(neigh_i, (np.ndarray,)):
                neigh_i = neigh_i.tolist()
            if isinstance(neigh_i, (list, tuple)):
                neigh_iter: Iterable[int] = neigh_i
            else:
                # e.g. a scalar -> treat as one neighbor
                neigh_iter = [int(neigh_i)]

            # De-dup within row; drop self.
            seen = set()
            for k in neigh_iter:
                kk = int(k)
                if kk == i:
                    continue
                if kk < 0 or kk >= N:
                    raise ValueError(
                        f"neighbors[{m}][{i}] contains out-of-range index {kk} "
                        f"(valid: 0..{N-1})"
                    )
                if kk in seen:
                    continue
                seen.add(kk)
                indices.append([i, kk])

        if len(indices) == 0:
            idx = tf.zeros((0, 2), dtype=tf.int64)
            vals = tf.zeros((0,), dtype=tf.float64)
        else:
            idx = tf.convert_to_tensor(indices, dtype=tf.int64)
            vals = tf.ones((idx.shape[0],), dtype=tf.float64)

        A = tf.SparseTensor(indices=idx, values=vals, dense_shape=(N, N))
        A = tf.sparse.reorder(A)
        peer_adj.append(A)

    return tuple(peer_adj)


class Bonus2Estimator:
    """
    Estimate Bonus Q2 parameters with elementwise RW-MH on unconstrained z-blocks.

    Block shapes:
      - Market-product (M,J): z_beta_market_mj
      - Product (J,):        z_beta_habit_j, z_beta_peer_j, z_decay_rate_j
      - Market×DOW (M,7):    z_beta_dow_m
      - Product×DOW (J,7):   z_beta_dow_j
      - Market×K (M,K):      z_a_m, z_b_m
      - Product×K (J,K):     z_a_j, z_b_j
    """

    def __init__(
        self,
        y_mit: Any,
        delta_mj: Any,
        dow_t: Any,
        sin_k_theta: Any,
        cos_k_theta: Any,
        neighbors: Any,
        L: int,
        init_theta: dict[str, float],
        sigmas: dict[str, float],
        seed: int,
        kappa_decay: float,
        eps_decay: float = 0.0,
    ) -> None:
        # ---- infer dimensions ----
        y_np = np.asarray(y_mit)
        if y_np.ndim != 3:
            raise ValueError("y_mit must be 3D (M,N,T)")
        self.M, self.N, self.T = (int(x) for x in y_np.shape)

        delta_np = np.asarray(delta_mj, dtype=np.float64)
        if delta_np.ndim != 2:
            raise ValueError("delta_mj must be 2D (M,J)")
        if int(delta_np.shape[0]) != self.M:
            raise ValueError("delta_mj first axis must match y_mit markets (M)")
        self.J = int(delta_np.shape[1])

        # ---- seasonal features: accept (K,T) or transpose from (T,K) ----
        sin_np = np.asarray(sin_k_theta, dtype=np.float64)
        cos_np = np.asarray(cos_k_theta, dtype=np.float64)
        if sin_np.ndim != 2 or cos_np.ndim != 2:
            raise ValueError("sin_k_theta and cos_k_theta must be 2D")
        if sin_np.shape != cos_np.shape:
            raise ValueError("sin_k_theta and cos_k_theta must have identical shape")

        if int(sin_np.shape[1]) == self.T:
            # (K,T)
            self.K = int(sin_np.shape[0])
            sin_np_kt = sin_np
            cos_np_kt = cos_np
        elif int(sin_np.shape[0]) == self.T:
            # (T,K) -> (K,T)
            self.K = int(sin_np.shape[1])
            sin_np_kt = sin_np.T
            cos_np_kt = cos_np.T
        else:
            raise ValueError(
                "sin_k_theta/cos_k_theta must have T on one axis (shape (K,T) or (T,K))"
            )

        # ---- convert to TF tensors ----
        self.y_mit = tf.convert_to_tensor(y_np, dtype=tf.int32)  # (M,N,T)
        self.delta_mj = tf.convert_to_tensor(delta_np, dtype=tf.float64)  # (M,J)

        dow_np = np.asarray(dow_t)
        if dow_np.ndim != 1 or int(dow_np.shape[0]) != self.T:
            raise ValueError("dow_t must be 1D with length T")
        self.dow_t = tf.convert_to_tensor(dow_np, dtype=tf.int32)  # (T,)

        self.sin_k_theta = tf.convert_to_tensor(sin_np_kt, dtype=tf.float64)  # (K,T)
        self.cos_k_theta = tf.convert_to_tensor(cos_np_kt, dtype=tf.float64)  # (K,T)

        # ---- network: build peer_adj_m ----
        self.peer_adj_m = _neighbors_to_peer_adj_m(neighbors, M=self.M, N=self.N)

        self.L = tf.convert_to_tensor(int(L), dtype=tf.int32)
        self.kappa_decay = tf.convert_to_tensor(float(kappa_decay), dtype=tf.float64)
        if float(kappa_decay) <= 0.0:
            raise ValueError("kappa_decay must be > 0")
        self.eps_decay = tf.convert_to_tensor(float(eps_decay), dtype=tf.float64)

        # ---- prior scales over z (scalar float64 tensors) ----
        # Expected sigma_z keys (z-space):
        #   z_beta_market_mj, z_beta_habit_j, z_beta_peer_j, z_decay_rate_j,
        #   z_beta_dow_m, z_beta_dow_j, z_a_m, z_b_m, z_a_j, z_b_j
        self.sigma_z: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(v), dtype=tf.float64)
            for k, v in sigmas.items()
        }

        # ---- pack constant model inputs for posterior views ----
        self.inputs: dict[str, Any] = {
            "y_mit": self.y_mit,
            "delta_mj": self.delta_mj,
            "dow_t": self.dow_t,
            "sin_k_theta": self.sin_k_theta,
            "cos_k_theta": self.cos_k_theta,
            "peer_adj_m": self.peer_adj_m,
            "L": self.L,
            "kappa_decay": self.kappa_decay,
        }
        if float(eps_decay) != 0.0:
            self.inputs["eps_decay"] = self.eps_decay

        # ---- RNG ----
        self.rng = tf.random.Generator.from_seed(int(seed))

        # ---- initialize z explicitly from init_theta (scalar fills) ----
        beta_market0 = tf.convert_to_tensor(
            float(init_theta["beta_market"]), tf.float64
        )
        beta_habit0 = tf.convert_to_tensor(float(init_theta["beta_habit"]), tf.float64)
        beta_peer0 = tf.convert_to_tensor(float(init_theta["beta_peer"]), tf.float64)

        decay0 = float(init_theta["decay_rate"])
        decay0 = float(np.clip(decay0, 1e-6, 1.0 - 1e-6))
        decay0_t = tf.convert_to_tensor(decay0, tf.float64)

        beta_dow_m0 = tf.convert_to_tensor(float(init_theta["beta_dow_m"]), tf.float64)
        beta_dow_j0 = tf.convert_to_tensor(float(init_theta["beta_dow_j"]), tf.float64)

        a_m0 = tf.convert_to_tensor(float(init_theta["a_m"]), tf.float64)
        b_m0 = tf.convert_to_tensor(float(init_theta["b_m"]), tf.float64)
        a_j0 = tf.convert_to_tensor(float(init_theta["a_j"]), tf.float64)
        b_j0 = tf.convert_to_tensor(float(init_theta["b_j"]), tf.float64)

        z_beta_market0 = tf.fill((self.M, self.J), beta_market0)
        z_beta_habit0 = tf.fill((self.J,), beta_habit0)
        z_beta_peer0 = tf.fill((self.J,), beta_peer0)

        # logit for decay_rate in (0,1)
        z_decay0 = tf.fill((self.J,), tf.math.log(decay0_t) - tf.math.log1p(-decay0_t))

        z_beta_dow_m0 = tf.fill((self.M, 7), beta_dow_m0)
        z_beta_dow_j0 = tf.fill((self.J, 7), beta_dow_j0)

        z_a_m0 = tf.fill((self.M, self.K), a_m0)
        z_b_m0 = tf.fill((self.M, self.K), b_m0)
        z_a_j0 = tf.fill((self.J, self.K), a_j0)
        z_b_j0 = tf.fill((self.J, self.K), b_j0)

        self.z: dict[str, tf.Variable] = {
            "z_beta_market_mj": tf.Variable(
                z_beta_market0, trainable=False, dtype=tf.float64
            ),
            "z_beta_habit_j": tf.Variable(
                z_beta_habit0, trainable=False, dtype=tf.float64
            ),
            "z_beta_peer_j": tf.Variable(
                z_beta_peer0, trainable=False, dtype=tf.float64
            ),
            "z_decay_rate_j": tf.Variable(z_decay0, trainable=False, dtype=tf.float64),
            "z_beta_dow_m": tf.Variable(
                z_beta_dow_m0, trainable=False, dtype=tf.float64
            ),
            "z_beta_dow_j": tf.Variable(
                z_beta_dow_j0, trainable=False, dtype=tf.float64
            ),
            "z_a_m": tf.Variable(z_a_m0, trainable=False, dtype=tf.float64),
            "z_b_m": tf.Variable(z_b_m0, trainable=False, dtype=tf.float64),
            "z_a_j": tf.Variable(z_a_j0, trainable=False, dtype=tf.float64),
            "z_b_j": tf.Variable(z_b_j0, trainable=False, dtype=tf.float64),
        }

        # ---- step sizes (set in fit) ----
        self.k: dict[str, tf.Variable] = {
            "beta_market": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_habit": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_peer": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "decay_rate": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_dow_m": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_dow_j": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "a_m": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "b_m": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "a_j": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "b_j": tf.Variable(0.0, dtype=tf.float64, trainable=False),
        }

        # ---- acceptance counters ----
        self.accept: dict[str, tf.Variable] = {
            "beta_market": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_habit": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_peer": tf.Variable(0, dtype=tf.int32, trainable=False),
            "decay_rate": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_dow_m": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_dow_j": tf.Variable(0, dtype=tf.int32, trainable=False),
            "a_m": tf.Variable(0, dtype=tf.int32, trainable=False),
            "b_m": tf.Variable(0, dtype=tf.int32, trainable=False),
            "a_j": tf.Variable(0, dtype=tf.int32, trainable=False),
            "b_j": tf.Variable(0, dtype=tf.int32, trainable=False),
        }

        # ---- running sums for posterior means (not returned by get_results) ----
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.sums: dict[str, tf.Variable] = {
            "beta_market_mj": tf.Variable(
                tf.zeros((self.M, self.J), tf.float64), trainable=False
            ),
            "beta_habit_j": tf.Variable(
                tf.zeros((self.J,), tf.float64), trainable=False
            ),
            "beta_peer_j": tf.Variable(
                tf.zeros((self.J,), tf.float64), trainable=False
            ),
            "decay_rate_j": tf.Variable(
                tf.zeros((self.J,), tf.float64), trainable=False
            ),
            "beta_dow_m": tf.Variable(
                tf.zeros((self.M, 7), tf.float64), trainable=False
            ),
            "beta_dow_j": tf.Variable(
                tf.zeros((self.J, 7), tf.float64), trainable=False
            ),
            "a_m": tf.Variable(tf.zeros((self.M, self.K), tf.float64), trainable=False),
            "b_m": tf.Variable(tf.zeros((self.M, self.K), tf.float64), trainable=False),
            "a_j": tf.Variable(tf.zeros((self.J, self.K), tf.float64), trainable=False),
            "b_j": tf.Variable(tf.zeros((self.J, self.K), tf.float64), trainable=False),
        }

        # ---- block sizes for acceptance-rate denominators ----
        self._block_sizes: dict[str, int] = {
            "beta_market": self.M * self.J,
            "beta_habit": self.J,
            "beta_peer": self.J,
            "decay_rate": self.J,
            "beta_dow_m": self.M * 7,
            "beta_dow_j": self.J * 7,
            "a_m": self.M * self.K,
            "b_m": self.M * self.K,
            "a_j": self.J * self.K,
            "b_j": self.J * self.K,
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, n_iter: int, k: dict[str, float]) -> None:
        """
        Run MCMC for n_iter iterations (Python loop).

        Args:
          n_iter: number of sweeps
          k: RW step sizes keyed by:
             {"beta_market","beta_habit","beta_peer","decay_rate",
              "beta_dow_m","beta_dow_j","a_m","b_m","a_j","b_j"}
        """
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0")

        self.k["beta_market"].assign(float(k["beta_market"]))
        self.k["beta_habit"].assign(float(k["beta_habit"]))
        self.k["beta_peer"].assign(float(k["beta_peer"]))
        self.k["decay_rate"].assign(float(k["decay_rate"]))
        self.k["beta_dow_m"].assign(float(k["beta_dow_m"]))
        self.k["beta_dow_j"].assign(float(k["beta_dow_j"]))
        self.k["a_m"].assign(float(k["a_m"]))
        self.k["b_m"].assign(float(k["b_m"]))
        self.k["a_j"].assign(float(k["a_j"]))
        self.k["b_j"].assign(float(k["b_j"]))

        self._reset_chain_state()

        for it in range(n_iter):
            self._mcmc_iteration_step(it=tf.constant(it, dtype=tf.int32))

    def get_results(self) -> dict[str, object]:
        """
        Return LAST-draw parameters and acceptance rates as Python/numpy objects.
        """
        theta_last = model.unconstrained_to_theta(self._current_z_dict())

        theta_hat = {
            "beta_market_mj": theta_last["beta_market_mj"].numpy(),
            "beta_habit_j": theta_last["beta_habit_j"].numpy(),
            "beta_peer_j": theta_last["beta_peer_j"].numpy(),
            "decay_rate_j": theta_last["decay_rate_j"].numpy(),
            "beta_dow_m": theta_last["beta_dow_m"].numpy(),
            "beta_dow_j": theta_last["beta_dow_j"].numpy(),
            "a_m": theta_last["a_m"].numpy(),
            "b_m": theta_last["b_m"].numpy(),
            "a_j": theta_last["a_j"].numpy(),
            "b_j": theta_last["b_j"].numpy(),
        }

        n_saved = int(self.saved.numpy())
        counts = {kk: int(vv.numpy()) for kk, vv in self.accept.items()}

        denom_saved = max(1, n_saved)
        rates = {
            kk: counts[kk] / max(1, denom_saved * self._block_sizes[kk])
            for kk in counts.keys()
        }

        return {"theta_hat": theta_hat, "n_saved": n_saved, "accept": rates}

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _current_z_dict(self) -> dict[str, tf.Tensor]:
        return {
            "z_beta_market_mj": self.z["z_beta_market_mj"],
            "z_beta_habit_j": self.z["z_beta_habit_j"],
            "z_beta_peer_j": self.z["z_beta_peer_j"],
            "z_decay_rate_j": self.z["z_decay_rate_j"],
            "z_beta_dow_m": self.z["z_beta_dow_m"],
            "z_beta_dow_j": self.z["z_beta_dow_j"],
            "z_a_m": self.z["z_a_m"],
            "z_b_m": self.z["z_b_m"],
            "z_a_j": self.z["z_a_j"],
            "z_b_j": self.z["z_b_j"],
        }

    def _reset_chain_state(self) -> None:
        for v in self.accept.values():
            v.assign(0)

        self.saved.assign(0)
        for v in self.sums.values():
            v.assign(tf.zeros_like(v))

    # -------------------------------------------------------------------------
    # Only compiled method
    # -------------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it: tf.Tensor) -> None:
        inputs = self.inputs
        sigma_z = self.sigma_z

        # Current z blocks (as Variables).
        z_beta_market_mj = self.z["z_beta_market_mj"]
        z_beta_habit_j = self.z["z_beta_habit_j"]
        z_beta_peer_j = self.z["z_beta_peer_j"]
        z_decay_rate_j = self.z["z_decay_rate_j"]
        z_beta_dow_m = self.z["z_beta_dow_m"]
        z_beta_dow_j = self.z["z_beta_dow_j"]
        z_a_m = self.z["z_a_m"]
        z_b_m = self.z["z_b_m"]
        z_a_j = self.z["z_a_j"]
        z_b_j = self.z["z_b_j"]

        # ---- per-block updates ----

        z_new, accepted = update_z_beta_market_mj(
            rng=self.rng,
            k=self.k["beta_market"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=z_beta_market_mj,
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_decay_rate_j=z_decay_rate_j,
            z_beta_dow_m=z_beta_dow_m,
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_beta_market_mj"].assign(z_new)
        self.accept["beta_market"].assign_add(
            tf.reduce_sum(tf.cast(accepted, tf.int32))
        )

        z_new, accepted = update_z_beta_habit_j(
            rng=self.rng,
            k=self.k["beta_habit"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=z_beta_habit_j,
            z_beta_peer_j=z_beta_peer_j,
            z_decay_rate_j=z_decay_rate_j,
            z_beta_dow_m=z_beta_dow_m,
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_beta_habit_j"].assign(z_new)
        self.accept["beta_habit"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_beta_peer_j(
            rng=self.rng,
            k=self.k["beta_peer"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=z_beta_peer_j,
            z_decay_rate_j=z_decay_rate_j,
            z_beta_dow_m=z_beta_dow_m,
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_beta_peer_j"].assign(z_new)
        self.accept["beta_peer"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_decay_rate_j(
            rng=self.rng,
            k=self.k["decay_rate"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=z_decay_rate_j,
            z_beta_dow_m=z_beta_dow_m,
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_decay_rate_j"].assign(z_new)
        self.accept["decay_rate"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_beta_dow_m(
            rng=self.rng,
            k=self.k["beta_dow_m"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=z_beta_dow_m,
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_beta_dow_m"].assign(z_new)
        self.accept["beta_dow_m"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_beta_dow_j(
            rng=self.rng,
            k=self.k["beta_dow_j"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=self.z["z_beta_dow_m"],
            z_beta_dow_j=z_beta_dow_j,
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_beta_dow_j"].assign(z_new)
        self.accept["beta_dow_j"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_a_m(
            rng=self.rng,
            k=self.k["a_m"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=self.z["z_beta_dow_m"],
            z_beta_dow_j=self.z["z_beta_dow_j"],
            z_a_m=z_a_m,
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_a_m"].assign(z_new)
        self.accept["a_m"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_b_m(
            rng=self.rng,
            k=self.k["b_m"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=self.z["z_beta_dow_m"],
            z_beta_dow_j=self.z["z_beta_dow_j"],
            z_a_m=self.z["z_a_m"],
            z_b_m=z_b_m,
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_b_m"].assign(z_new)
        self.accept["b_m"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_a_j(
            rng=self.rng,
            k=self.k["a_j"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=self.z["z_beta_dow_m"],
            z_beta_dow_j=self.z["z_beta_dow_j"],
            z_a_m=self.z["z_a_m"],
            z_b_m=self.z["z_b_m"],
            z_a_j=z_a_j,
            z_b_j=z_b_j,
        )
        self.z["z_a_j"].assign(z_new)
        self.accept["a_j"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        z_new, accepted = update_z_b_j(
            rng=self.rng,
            k=self.k["b_j"],
            inputs=inputs,
            sigma_z=sigma_z,
            z_beta_market_mj=self.z["z_beta_market_mj"],
            z_beta_habit_j=self.z["z_beta_habit_j"],
            z_beta_peer_j=self.z["z_beta_peer_j"],
            z_decay_rate_j=self.z["z_decay_rate_j"],
            z_beta_dow_m=self.z["z_beta_dow_m"],
            z_beta_dow_j=self.z["z_beta_dow_j"],
            z_a_m=self.z["z_a_m"],
            z_b_m=self.z["z_b_m"],
            z_a_j=self.z["z_a_j"],
            z_b_j=z_b_j,
        )
        self.z["z_b_j"].assign(z_new)
        self.accept["b_j"].assign_add(tf.reduce_sum(tf.cast(accepted, tf.int32)))

        # ---- Accumulate posterior means (no burn-in/thinning; not returned) ----
        self.saved.assign_add(1)
        theta_curr = model.unconstrained_to_theta(self._current_z_dict())

        self.sums["beta_market_mj"].assign_add(theta_curr["beta_market_mj"])
        self.sums["beta_habit_j"].assign_add(theta_curr["beta_habit_j"])
        self.sums["beta_peer_j"].assign_add(theta_curr["beta_peer_j"])
        self.sums["decay_rate_j"].assign_add(theta_curr["decay_rate_j"])
        self.sums["beta_dow_m"].assign_add(theta_curr["beta_dow_m"])
        self.sums["beta_dow_j"].assign_add(theta_curr["beta_dow_j"])
        self.sums["a_m"].assign_add(theta_curr["a_m"])
        self.sums["b_m"].assign_add(theta_curr["b_m"])
        self.sums["a_j"].assign_add(theta_curr["a_j"])
        self.sums["b_j"].assign_add(theta_curr["b_j"])

        report_iteration_progress(z=self._current_z_dict(), it=it)
