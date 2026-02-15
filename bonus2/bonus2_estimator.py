"""
bonus2/bonus2_estimator.py

Bonus Q2 estimator (updated spec): RW-MH over unconstrained z blocks.

Updated model inputs (known to estimator):
  y_mit          (M,N,T) int32  choices; 0=outside, c=j+1=inside product j
  delta_mj       (M,J)   f64    Phase-1 baseline utilities (fixed)
  weekend_t      (T,)    int32  weekend indicator in {0,1}
  season_sin_kt  (K,T)   f64
  season_cos_kt  (K,T)   f64
  neighbors      neighbors[m][i] -> list[int] (within-market)
  L              scalar int32    peer lookback window length (known)
  decay          scalar f64      known habit decay in (0,1)

Unconstrained sampler blocks z (all float64):
  z_beta_market_j  (J,)
  z_beta_habit_j   (J,)
  z_beta_peer_j    (J,)
  z_beta_dow_j     (J,2)
  z_a_m            (M,K)
  z_b_m            (M,K)

Step sizes k passed to fit() keyed by:
  {"beta_market","beta_habit","beta_peer","beta_dow_j","a_m","b_m"}

Acceptance accounting:
  - One accept/reject per block per sweep.
  - Reported acceptance rates are accept_count / n_saved per block.

Notes:
  - No traces are stored; get_results() returns theta_init and theta_hat (last draw).
  - _mcmc_iteration_step remains a compiled tf.function graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from bonus2 import bonus2_model as model
from bonus2.bonus2_updates import (
    update_z_a_m,
    update_z_b_m,
    update_z_beta_dow_j,
    update_z_beta_habit_j,
    update_z_beta_market_j,
    update_z_beta_peer_j,
)

try:
    from bonus2.bonus2_diagnostics import report_iteration_progress
except Exception:  # pragma: no cover

    def report_iteration_progress(z: dict[str, tf.Tensor], it: tf.Tensor) -> None:
        means = []
        for k in sorted(z.keys()):
            x = tf.cast(z[k], tf.float64)
            means.append(tf.reduce_mean(x))
        tf.print("[Bonus2] it=", it, "| mean(z)=", means)


def _coerce_neighbors_to_list(neighbors: Any, M: int, N: int) -> list[list[list[int]]]:
    """Coerce neighbor structure to nested Python lists: neighbors[m][i] -> list[int]."""
    nb = neighbors
    if isinstance(nb, tf.RaggedTensor):
        nb = nb.to_list()
    elif isinstance(nb, np.ndarray):
        nb = nb.tolist()

    if not isinstance(nb, (list, tuple)) or len(nb) != M:
        raise ValueError("neighbors must be a nested list/tuple with outer length M")

    out: list[list[list[int]]] = []
    for m in range(M):
        rows = nb[m]
        if not isinstance(rows, (list, tuple)) or len(rows) != N:
            raise ValueError("neighbors[m] must have length N for each market m")

        rows_out: list[list[int]] = []
        for i in range(N):
            neigh_i = rows[i]
            if neigh_i is None:
                rows_out.append([])
                continue

            if isinstance(neigh_i, np.ndarray):
                neigh_i = neigh_i.tolist()
            if isinstance(neigh_i, (list, tuple)):
                neigh_list = [int(k) for k in neigh_i]
            else:
                neigh_list = [int(neigh_i)]

            clean: list[int] = []
            seen = set()
            for k in neigh_list:
                if k == i:
                    continue
                if k < 0 or k >= N:
                    continue
                if k in seen:
                    continue
                seen.add(k)
                clean.append(k)
            rows_out.append(clean)

        out.append(rows_out)

    return out


class Bonus2Estimator:
    """Estimate Bonus Q2 parameters with RW-MH on unconstrained z-blocks."""

    def __init__(
        self,
        y_mit: Any,
        delta_mj: Any,
        weekend_t: Any,
        season_sin_kt: Any,
        season_cos_kt: Any,
        neighbors: Any,
        L: int,
        decay: float,
        init_theta: dict[str, float],
        sigmas: dict[str, float],
        seed: int,
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

        # ---- seasonal basis: accept (K,T) or transpose from (T,K) ----
        sin_np = np.asarray(season_sin_kt, dtype=np.float64)
        cos_np = np.asarray(season_cos_kt, dtype=np.float64)
        if sin_np.ndim != 2 or cos_np.ndim != 2:
            raise ValueError("season_sin_kt and season_cos_kt must be 2D")
        if sin_np.shape != cos_np.shape:
            raise ValueError(
                "season_sin_kt and season_cos_kt must have identical shape"
            )

        if int(sin_np.shape[1]) == self.T:
            self.K = int(sin_np.shape[0])
            sin_np_kt = sin_np
            cos_np_kt = cos_np
        elif int(sin_np.shape[0]) == self.T:
            self.K = int(sin_np.shape[1])
            sin_np_kt = sin_np.T
            cos_np_kt = cos_np.T
        else:
            raise ValueError(
                "season_sin_kt/season_cos_kt must have T on one axis "
                "(shape (K,T) or (T,K))"
            )

        # ---- weekend indicator ----
        w_np = np.asarray(weekend_t)
        if w_np.ndim != 1 or int(w_np.shape[0]) != self.T:
            raise ValueError("weekend_t must be 1D with length T")
        # minimal validation: values must be 0/1
        if not np.all((w_np == 0) | (w_np == 1)):
            raise ValueError("weekend_t must contain only 0/1 values")

        # ---- convert to TF tensors ----
        self.y_mit = tf.convert_to_tensor(y_np, dtype=tf.int32)  # (M,N,T)
        self.delta_mj = tf.convert_to_tensor(delta_np, dtype=tf.float64)  # (M,J)
        self.weekend_t = tf.convert_to_tensor(w_np, dtype=tf.int32)  # (T,)
        self.season_sin_kt = tf.convert_to_tensor(sin_np_kt, dtype=tf.float64)  # (K,T)
        self.season_cos_kt = tf.convert_to_tensor(cos_np_kt, dtype=tf.float64)  # (K,T)

        decay_f = float(decay)
        if not (0.0 < decay_f < 1.0):
            raise ValueError("decay must be in (0,1)")
        self.decay = tf.convert_to_tensor(decay_f, dtype=tf.float64)

        # ---- network: build peer_adj_m ----
        nbrs_m = _coerce_neighbors_to_list(neighbors, M=self.M, N=self.N)
        self.peer_adj_m = model.build_peer_adjacency(nbrs_m=nbrs_m, N=self.N)

        self.L = tf.convert_to_tensor(int(L), dtype=tf.int32)

        # ---- prior scales over z (scalar float64 tensors) ----
        self.sigma_z: dict[str, tf.Tensor] = {
            k: tf.convert_to_tensor(float(v), dtype=tf.float64)
            for k, v in sigmas.items()
        }

        # ---- pack constant model inputs for posterior views ----
        self.inputs: dict[str, Any] = {
            "y_mit": self.y_mit,
            "delta_mj": self.delta_mj,
            "weekend_t": self.weekend_t,
            "season_sin_kt": self.season_sin_kt,
            "season_cos_kt": self.season_cos_kt,
            "peer_adj_m": self.peer_adj_m,
            "L": self.L,
            "decay": self.decay,
        }

        # ---- RNG ----
        self.rng = tf.random.Generator.from_seed(int(seed))

        # ---- initialize z explicitly from init_theta (scalar fills) ----
        beta_market0 = tf.convert_to_tensor(
            float(init_theta["beta_market"]), tf.float64
        )
        beta_habit0 = tf.convert_to_tensor(float(init_theta["beta_habit"]), tf.float64)
        beta_peer0 = tf.convert_to_tensor(float(init_theta["beta_peer"]), tf.float64)
        beta_dow_j0 = tf.convert_to_tensor(float(init_theta["beta_dow_j"]), tf.float64)
        a_m0 = tf.convert_to_tensor(float(init_theta["a_m"]), tf.float64)
        b_m0 = tf.convert_to_tensor(float(init_theta["b_m"]), tf.float64)

        z_beta_market_j0 = tf.fill((self.J,), beta_market0)
        z_beta_habit_j0 = tf.fill((self.J,), beta_habit0)
        z_beta_peer_j0 = tf.fill((self.J,), beta_peer0)
        z_beta_dow_j0 = tf.fill((self.J, 2), beta_dow_j0)
        z_a_m0 = tf.fill((self.M, self.K), a_m0)
        z_b_m0 = tf.fill((self.M, self.K), b_m0)

        self.z: dict[str, tf.Variable] = {
            "z_beta_market_j": tf.Variable(
                z_beta_market_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_habit_j": tf.Variable(
                z_beta_habit_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_peer_j": tf.Variable(
                z_beta_peer_j0, trainable=False, dtype=tf.float64
            ),
            "z_beta_dow_j": tf.Variable(
                z_beta_dow_j0, trainable=False, dtype=tf.float64
            ),
            "z_a_m": tf.Variable(z_a_m0, trainable=False, dtype=tf.float64),
            "z_b_m": tf.Variable(z_b_m0, trainable=False, dtype=tf.float64),
        }

        # ---- store theta_init for evaluation printouts ----
        self.theta_init = self._theta_from_z(self._current_z_dict(), as_numpy=True)

        # ---- step sizes (set in fit) ----
        self.k: dict[str, tf.Variable] = {
            "beta_market": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_habit": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_peer": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "beta_dow_j": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "a_m": tf.Variable(0.0, dtype=tf.float64, trainable=False),
            "b_m": tf.Variable(0.0, dtype=tf.float64, trainable=False),
        }

        # ---- acceptance counters (one per block per sweep) ----
        self.accept: dict[str, tf.Variable] = {
            "beta_market": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_habit": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_peer": tf.Variable(0, dtype=tf.int32, trainable=False),
            "beta_dow_j": tf.Variable(0, dtype=tf.int32, trainable=False),
            "a_m": tf.Variable(0, dtype=tf.int32, trainable=False),
            "b_m": tf.Variable(0, dtype=tf.int32, trainable=False),
        }

        # ---- sweep counter ----
        self.saved = tf.Variable(0, dtype=tf.int64, trainable=False)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def fit(self, n_iter: int, k: dict[str, float]) -> None:
        """Run MCMC for n_iter sweeps (Python loop).

        Args:
          n_iter: number of sweeps
          k: RW step sizes keyed by:
             {"beta_market","beta_habit","beta_peer","beta_dow_j","a_m","b_m"}
        """
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("n_iter must be > 0")

        self.k["beta_market"].assign(float(k["beta_market"]))
        self.k["beta_habit"].assign(float(k["beta_habit"]))
        self.k["beta_peer"].assign(float(k["beta_peer"]))
        self.k["beta_dow_j"].assign(float(k["beta_dow_j"]))
        self.k["a_m"].assign(float(k["a_m"]))
        self.k["b_m"].assign(float(k["b_m"]))

        self._reset_chain_state()

        for it in range(n_iter):
            self._mcmc_iteration_step(it=tf.constant(it, dtype=tf.int32))

    def get_results(self) -> dict[str, object]:
        """Return theta_init, theta_hat (last draw), n_saved, accept (per-block rates)."""
        theta_last = self._theta_from_z(self._current_z_dict(), as_numpy=False)
        theta_hat = {k: v.numpy() for k, v in theta_last.items()}

        n_saved = int(self.saved.numpy())
        counts = {kk: int(vv.numpy()) for kk, vv in self.accept.items()}

        denom = max(1, n_saved)
        rates = {kk: counts[kk] / denom for kk in counts.keys()}

        return {
            "theta_init": self.theta_init,
            "theta_hat": theta_hat,
            "n_saved": n_saved,
            "accept": rates,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _current_z_dict(self) -> dict[str, tf.Tensor]:
        return {
            "z_beta_market_j": self.z["z_beta_market_j"],
            "z_beta_habit_j": self.z["z_beta_habit_j"],
            "z_beta_peer_j": self.z["z_beta_peer_j"],
            "z_beta_dow_j": self.z["z_beta_dow_j"],
            "z_a_m": self.z["z_a_m"],
            "z_b_m": self.z["z_b_m"],
        }

    def _theta_from_z(
        self, z_dict: dict[str, tf.Tensor], as_numpy: bool
    ) -> dict[str, Any]:
        """Convert z -> theta (no centering/identifiability constraints in updated spec)."""
        theta = model.unconstrained_to_theta(z_dict)
        if as_numpy:
            return {k: v.numpy() for k, v in theta.items()}
        return theta

    def _reset_chain_state(self) -> None:
        for v in self.accept.values():
            v.assign(0)
        self.saved.assign(0)

    # -------------------------------------------------------------------------
    # Only compiled method
    # -------------------------------------------------------------------------

    @tf.function(reduce_retracing=True)
    def _mcmc_iteration_step(self, it: tf.Tensor) -> None:
        inputs = self.inputs
        sigma_z = self.sigma_z

        def _acc(block: str, accepted: tf.Tensor) -> None:
            self.accept[block].assign_add(tf.cast(accepted, tf.int32))

        # ---- beta_market_j ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_beta_market_j(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["beta_market"],
            rng=self.rng,
        )
        self.z["z_beta_market_j"].assign(z_new["z_beta_market_j"])
        _acc("beta_market", accepted)

        # ---- beta_habit_j ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_beta_habit_j(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["beta_habit"],
            rng=self.rng,
        )
        self.z["z_beta_habit_j"].assign(z_new["z_beta_habit_j"])
        _acc("beta_habit", accepted)

        # ---- beta_peer_j ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_beta_peer_j(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["beta_peer"],
            rng=self.rng,
        )
        self.z["z_beta_peer_j"].assign(z_new["z_beta_peer_j"])
        _acc("beta_peer", accepted)

        # ---- beta_dow_j ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_beta_dow_j(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["beta_dow_j"],
            rng=self.rng,
        )
        self.z["z_beta_dow_j"].assign(z_new["z_beta_dow_j"])
        _acc("beta_dow_j", accepted)

        # ---- a_m ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_a_m(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["a_m"],
            rng=self.rng,
        )
        self.z["z_a_m"].assign(z_new["z_a_m"])
        _acc("a_m", accepted)

        # ---- b_m ----
        z_dict = self._current_z_dict()
        z_new, accepted = update_z_b_m(
            z=z_dict,
            inputs=inputs,
            sigma_z=sigma_z,
            step_size=self.k["b_m"],
            rng=self.rng,
        )
        self.z["z_b_m"].assign(z_new["z_b_m"])
        _acc("b_m", accepted)

        self.saved.assign_add(1)
        report_iteration_progress(z=self._current_z_dict(), it=it)
