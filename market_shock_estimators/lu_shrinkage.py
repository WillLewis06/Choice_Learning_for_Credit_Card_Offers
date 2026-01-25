# market_shock_estimators/lu_shrinkage.py
#
# Updated to:
#   - use LuPosteriorTF API (Ebar, eta passed explicitly)
#   - call external kernels: tmh_step (tmh.py) and rw_mh_step (rw_mh.py)
#   - remove inlined TMH and RW-MH implementations
#
# Source baseline was the prior lu_shrinkage.py you provided. :contentReference[oaicite:0]{index=0}

from __future__ import annotations

import tensorflow as tf

from market_shock_estimators.tmh import tmh_step
from market_shock_estimators.rw_mh import rw_mh_step


class LuShrinkageEstimator:
    """
    Shrinkage estimator sampler (Lu Section 4 simulation target).

    Blocking (Lu Section 4):
      - TMH (Laplace independence MH) for beta and eta_t
      - RW-MH for r and Ebar_t
      - Gibbs for gamma_t, phi_t

    Implementation:
      - TMH delegated to tmh_step (LBFGS mode-finding)
      - RW-MH delegated to rw_mh_step
      - Posterior delegated to LuPosteriorTF.market_logp(Ebar=..., eta=..., ...)
    """

    def __init__(
        self,
        *,
        posterior,
        x,
        Z,
        q,
        q0,
        beta_init,
        r_init,
        Ebar_init,
        eta_init,
        gamma_init,
        phi_init,
        r_step=0.05,
        Ebar_step=0.05,
        ridge=1e-6,
        max_lbfgs_iters=100,
        seed=0,
        dtype=tf.float64,
    ):
        self.posterior = posterior
        self.dtype = dtype

        # Data
        self.x = tf.convert_to_tensor(x, dtype=dtype)
        self.Z = tf.convert_to_tensor(Z, dtype=dtype)
        self.q = tf.convert_to_tensor(q, dtype=dtype)
        self.q0 = tf.convert_to_tensor(q0, dtype=dtype)

        self.T = int(self.x.shape[0])
        self.J = int(self.x.shape[1])

        # State
        self.beta = tf.Variable(beta_init, dtype=dtype, trainable=False)
        self.r = tf.Variable(r_init, dtype=dtype, trainable=False)
        self.Ebar = tf.Variable(Ebar_init, dtype=dtype, trainable=False)  # (T,)
        self.eta = tf.Variable(eta_init, dtype=dtype, trainable=False)  # (T,J)
        self.gamma = tf.Variable(gamma_init, dtype=tf.int32, trainable=False)  # (T,J)

        phi_init = tf.convert_to_tensor(phi_init, dtype=dtype)
        phi_init = tf.reshape(phi_init, [self.T])  # force (T,)
        self.phi = tf.Variable(phi_init, dtype=dtype, trainable=False)  # (T,)

        # Kernel params
        self.r_step = tf.convert_to_tensor(r_step, dtype=dtype)
        self.Ebar_step = tf.convert_to_tensor(Ebar_step, dtype=dtype)
        self.ridge = tf.convert_to_tensor(ridge, dtype=dtype)
        self.max_lbfgs_iters = int(max_lbfgs_iters)

        # RNG
        self.rng = tf.random.Generator.from_seed(int(seed))

        # Acceptance counters
        self.acc_beta = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.acc_r = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.acc_eta = tf.Variable(tf.zeros([self.T], dtype=tf.int64), trainable=False)
        self.acc_Ebar = tf.Variable(tf.zeros([self.T], dtype=tf.int64), trainable=False)

    # ================================================================
    # Main step
    # ================================================================

    def step(self):
        beta_old = tf.identity(self.beta)
        r_old = tf.identity(self.r)
        eta_old = tf.identity(self.eta)

        # 1) beta — TMH
        beta_new, acc = self._tmh_beta()
        self.beta.assign(beta_new)
        self.acc_beta.assign_add(tf.cast(acc, tf.int64))

        # 2) r — RW-MH
        r_new, acc = rw_mh_step(
            theta0=self.r,
            logp_fn=self._r_logp,
            step_size=self.r_step,
            rng=self.rng,
        )
        self.r.assign(r_new)
        self.acc_r.assign_add(tf.cast(acc, tf.int64))

        # 3) Market-by-market
        for t in range(self.T):
            # Ebar_t — RW-MH
            Ebar_new, acc = self._rw_mh_Ebar(t)
            _set_1d_at(self.Ebar, t, Ebar_new)
            _add_1d_at(self.acc_Ebar, t, tf.cast(acc, tf.int64))

            # eta_t — TMH
            eta_new, acc = self._tmh_eta(t)
            _set_row_at(self.eta, t, eta_new)
            _add_1d_at(self.acc_eta, t, tf.cast(acc, tf.int64))

        # 4) gamma, phi — Gibbs
        for t in range(self.T):
            self._gibbs_gamma_phi_market(t)

        _print_iteration_diagnostics_tf(
            beta_old, r_old, eta_old, self.beta, self.r, self.eta
        )

    # ================================================================
    # Log posteriors
    # ================================================================

    def _market_logp(self, t, Ebar_t, eta_t, beta=None, r=None):
        """
        Market log posterior contribution (likelihood + priors on Ebar and eta).

        Uses LuPosteriorTF.market_logp(Ebar=..., eta=..., ...).
        """
        beta_use = self.beta if beta is None else beta
        r_use = self.r if r is None else r

        return self.posterior.market_logp(
            Ebar=Ebar_t,
            eta=eta_t,
            q=self.q[t],
            q0=self.q0[t],
            x=self.x[t],
            Z=self.Z[t],
            beta=beta_use,
            r=r_use,
            gamma=self.gamma[t],
        )

    def _beta_logp(self, beta):
        lp = tf.constant(0.0, self.dtype)
        for t in range(self.T):
            lp += self._market_logp(t, self.Ebar[t], self.eta[t], beta=beta)
        # Global prior on beta (kept outside market_logp)
        return lp - 0.5 * tf.reduce_sum(beta * beta) / self.posterior.beta_var

    def _r_logp(self, r):
        lp = tf.constant(0.0, self.dtype)
        for t in range(self.T):
            lp += self._market_logp(t, self.Ebar[t], self.eta[t], r=r)
        # Global prior on r (kept outside market_logp)
        return lp - 0.5 * tf.reduce_sum(r * r) / self.posterior.r_var

    # ================================================================
    # RW-MH wrappers
    # ================================================================

    def _rw_mh_Ebar(self, t):
        def logp_Ebar(val):
            return self._market_logp(t, val, self.eta[t])

        return rw_mh_step(
            theta0=self.Ebar[t],
            logp_fn=logp_Ebar,
            step_size=self.Ebar_step,
            rng=self.rng,
        )

    # ================================================================
    # TMH wrappers
    # ================================================================

    def _tmh_beta(self):
        return tmh_step(
            theta0=self.beta,
            logp_fn=self._beta_logp,
            ridge=self.ridge,
            max_lbfgs_iters=self.max_lbfgs_iters,
            rng=self.rng,
        )

    def _tmh_eta(self, t):
        def logp_eta(eta_t):
            return self._market_logp(t, self.Ebar[t], eta_t)

        return tmh_step(
            theta0=self.eta[t],
            logp_fn=logp_eta,
            ridge=self.ridge,
            max_lbfgs_iters=self.max_lbfgs_iters,
            rng=self.rng,
        )

    # ================================================================
    # Gibbs: gamma, phi
    # ================================================================

    def _gibbs_gamma_phi_market(self, t):
        eta_t = self.eta[t]
        phi_t = self.phi[t]

        T1 = self.posterior.T1_sq
        T0 = self.posterior.T0_sq

        p1 = phi_t * tf.exp(-0.5 * eta_t * eta_t / T1) / tf.sqrt(T1)
        p0 = (1.0 - phi_t) * tf.exp(-0.5 * eta_t * eta_t / T0) / tf.sqrt(T0)
        prob = p1 / (p1 + p0)

        u = self.rng.uniform([self.J], dtype=self.dtype)
        gamma_t = tf.cast(u < prob, tf.int32)

        # TF2-safe write: gamma[t, :] = gamma_t
        _set_row_at(self.gamma, t, gamma_t)

        a = self.posterior.a_phi + tf.reduce_sum(tf.cast(gamma_t, self.dtype))
        b = (
            self.posterior.b_phi
            + tf.cast(self.J, self.dtype)
            - tf.reduce_sum(tf.cast(gamma_t, self.dtype))
        )

        phi_new = _sample_beta_tf(self.rng, a, b, dtype=self.dtype)

        # TF2-safe write: phi[t] = phi_new
        _set_1d_at(self.phi, t, phi_new)

    # ================================================================
    # State export
    # ================================================================

    def state(self):
        return {
            "beta": self.beta.numpy(),
            "r": self.r.numpy(),
            "Ebar": self.Ebar.numpy(),
            "eta": self.eta.numpy(),
            "gamma": self.gamma.numpy(),
            "phi": self.phi.numpy(),
        }


def _set_1d_at(var: tf.Variable, t: int, value: tf.Tensor) -> None:
    """var[t] = value (TF2-safe)."""
    value = tf.cast(value, var.dtype)
    updated = tf.tensor_scatter_nd_update(var, indices=[[t]], updates=[value])
    var.assign(updated)


def _add_1d_at(var: tf.Variable, t: int, delta: tf.Tensor) -> None:
    """var[t] += delta (TF2-safe)."""
    delta = tf.cast(delta, var.dtype)
    updated = tf.tensor_scatter_nd_add(var, indices=[[t]], updates=[delta])
    var.assign(updated)


def _set_row_at(var: tf.Variable, t: int, row: tf.Tensor) -> None:
    """var[t, :] = row (TF2-safe)."""
    row = tf.cast(row, var.dtype)
    updated = tf.tensor_scatter_nd_update(var, indices=[[t]], updates=[row])
    var.assign(updated)


def _print_iteration_diagnostics_tf(beta_old, r_old, eta_old, beta, r, eta):
    print(
        "[LuShrinkage] "
        f"||Δβ||={tf.linalg.norm(beta - beta_old).numpy():.3e} "
        f"|Δr|={tf.linalg.norm(r - r_old).numpy():.3e} "
        f"mean||Δη||={tf.reduce_mean(tf.linalg.norm(eta - eta_old, axis=1)).numpy():.3e}"
    )


def _sample_beta_tf(rng: tf.random.Generator, a: tf.Tensor, b: tf.Tensor, dtype):
    """
    Sample Beta(a,b) using Gamma draws:
        X ~ Gamma(a, 1), Y ~ Gamma(b, 1), return X / (X + Y).
    Uses stateless Gamma draws seeded from the provided tf.random.Generator.
    """
    a = tf.convert_to_tensor(a, dtype=dtype)
    b = tf.convert_to_tensor(b, dtype=dtype)

    # Derive two independent stateless seeds from the generator
    seeds = rng.make_seeds(2)  # shape (2, 2), dtype int32
    seed_x = seeds[0]
    seed_y = seeds[1]

    # tf.random.stateless_gamma uses alpha (shape) and beta (rate)
    x = tf.random.stateless_gamma(
        shape=[],
        seed=seed_x,
        alpha=a,
        beta=tf.cast(1.0, dtype),
        dtype=dtype,
    )
    y = tf.random.stateless_gamma(
        shape=[],
        seed=seed_y,
        alpha=b,
        beta=tf.cast(1.0, dtype),
        dtype=dtype,
    )

    return x / (x + y)
