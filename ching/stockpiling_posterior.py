"""Posterior evaluation for the Ching-style stockpiling sampler."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from ching.stockpiling_model import InventoryMaps, solve_ccp_buy, unconstrained_to_theta

__all__ = [
    "StockpilingPosteriorConfig",
    "StockpilingPosteriorTF",
]


@dataclass(frozen=True)
class StockpilingPosteriorConfig:
    """Store posterior hyperparameters and numerical settings."""

    tol: float
    max_iter: int
    eps: float
    sigma_z_beta: float
    sigma_z_alpha: float
    sigma_z_v: float
    sigma_z_fc: float
    sigma_z_u_scale: float


class StockpilingPosteriorTF:
    """Evaluate likelihood, prior, and block log posteriors."""

    def __init__(
        self,
        config: StockpilingPosteriorConfig,
        a_mnjt: tf.Tensor,
        s_mjt: tf.Tensor,
        u_mj: tf.Tensor,
        P_price_mj: tf.Tensor,
        price_vals_mj: tf.Tensor,
        lambda_mn: tf.Tensor,
        waste_cost: tf.Tensor,
        inventory_maps: InventoryMaps,
    ):
        """Cache observed tensors, prior scales, and a fixed uniform inventory prior."""
        self.tol = float(config.tol)
        self.max_iter = int(config.max_iter)
        self.eps = tf.constant(config.eps, dtype=tf.float64)

        self.sigma_z_beta = tf.constant(config.sigma_z_beta, dtype=tf.float64)
        self.sigma_z_alpha = tf.constant(config.sigma_z_alpha, dtype=tf.float64)
        self.sigma_z_v = tf.constant(config.sigma_z_v, dtype=tf.float64)
        self.sigma_z_fc = tf.constant(config.sigma_z_fc, dtype=tf.float64)
        self.sigma_z_u_scale = tf.constant(config.sigma_z_u_scale, dtype=tf.float64)

        self.a_mnjt = a_mnjt
        self.s_mjt = s_mjt
        self.u_mj = u_mj
        self.P_price_mj = P_price_mj
        self.price_vals_mj = price_vals_mj
        self.lambda_mn = lambda_mn
        self.waste_cost = waste_cost
        self.inventory_maps = inventory_maps

        i_vals, _, _, _, _ = self.inventory_maps
        n_inventory_states = tf.shape(i_vals)[0]
        self.pi_I0 = tf.fill(
            tf.reshape(n_inventory_states, (1,)),
            tf.constant(1.0, dtype=tf.float64)
            / tf.cast(n_inventory_states, tf.float64),
        )
        self.lambda_mn_11 = self.lambda_mn[:, :, None, None]

        log_two_pi = tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))
        self._lp0_beta = -0.5 * (log_two_pi + 2.0 * tf.math.log(self.sigma_z_beta))
        self._lp0_alpha = -0.5 * (log_two_pi + 2.0 * tf.math.log(self.sigma_z_alpha))
        self._lp0_v = -0.5 * (log_two_pi + 2.0 * tf.math.log(self.sigma_z_v))
        self._lp0_fc = -0.5 * (log_two_pi + 2.0 * tf.math.log(self.sigma_z_fc))
        self._lp0_u_scale = -0.5 * (
            log_two_pi + 2.0 * tf.math.log(self.sigma_z_u_scale)
        )

    def _theta_from_z(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Map unconstrained sampler state to constrained parameters."""
        return unconstrained_to_theta(
            {
                "z_beta": z_beta,
                "z_alpha": z_alpha,
                "z_v": z_v,
                "z_fc": z_fc,
                "z_u_scale": z_u_scale,
            }
        )

    @tf.function(jit_compile=True)
    def _ccp_buy_from_z(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Solve buy CCPs for the current sampler state."""
        theta = self._theta_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        ccp_buy, _, _ = solve_ccp_buy(
            u_mj=self.u_mj,
            price_vals_mj=self.price_vals_mj,
            P_price_mj=self.P_price_mj,
            theta=theta,
            lambda_mn=self.lambda_mn,
            waste_cost=self.waste_cost,
            maps=self.inventory_maps,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        return ccp_buy

    @tf.function(jit_compile=True)
    def _shift_down(self, post: tf.Tensor) -> tf.Tensor:
        """Move probability mass through I' = max(I - 1, 0)."""
        n_states = tf.shape(post)[-1]

        def case_one() -> tf.Tensor:
            return post

        def case_many() -> tf.Tensor:
            zeros = tf.zeros_like(post[..., :1])
            first = post[..., :1] + post[..., 1:2]
            middle = post[..., 2:]
            return tf.concat([first, middle, zeros], axis=-1)

        return tf.cond(tf.equal(n_states, 1), case_one, case_many)

    @tf.function(jit_compile=True)
    def _shift_up(self, post: tf.Tensor) -> tf.Tensor:
        """Move probability mass through I' = min(I + 1, I_max)."""
        n_states = tf.shape(post)[-1]

        def case_one() -> tf.Tensor:
            return post

        def case_many() -> tf.Tensor:
            zeros = tf.zeros_like(post[..., :1])
            middle = post[..., :-2]
            last = post[..., -2:-1] + post[..., -1:]
            return tf.concat([zeros, middle, last], axis=-1)

        return tf.cond(tf.equal(n_states, 1), case_one, case_many)

    @tf.function(jit_compile=True)
    def _transition_inventory(self, post: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
        """Propagate the latent inventory distribution one period ahead."""
        lam = self.lambda_mn_11
        stay_or_restock = (1.0 - lam) * post + lam * self._shift_down(post)
        consume_after_buy = lam * post + (1.0 - lam) * self._shift_up(post)
        return tf.where(tf.equal(a_t[..., None], 1), consume_after_buy, stay_or_restock)

    @tf.function(jit_compile=True)
    def _filter_step_core(
        self,
        t: tf.Tensor,
        pi_acc: tf.Tensor,
        ccp_buy: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply one forward-filter update over latent inventory."""
        mnj_shape = tf.shape(pi_acc)[:3]
        s_t_mj = self.s_mjt[:, :, t]
        s_idx = tf.broadcast_to(s_t_mj[:, None, :], mnj_shape)

        p_buy_I = tf.gather(ccp_buy, s_idx, axis=3, batch_dims=3)
        a_t_mnj = self.a_mnjt[:, :, :, t]
        observed_action_prob = tf.where(
            tf.equal(a_t_mnj[..., None], 1),
            p_buy_I,
            1.0 - p_buy_I,
        )

        weighted = pi_acc * observed_action_prob
        action_prob = tf.reduce_sum(weighted, axis=3)
        action_prob = tf.maximum(action_prob, self.eps)

        post = weighted / action_prob[..., None]
        pi_next = self._transition_inventory(post=post, a_t=a_t_mnj)
        return p_buy_I, action_prob, pi_next

    @tf.function(jit_compile=True)
    def logprior_beta(self, z_beta: tf.Tensor) -> tf.Tensor:
        """Return the scalar Normal prior contribution for z_beta."""
        return tf.reduce_sum(
            self._lp0_beta - 0.5 * tf.square(z_beta / self.sigma_z_beta)
        )

    @tf.function(jit_compile=True)
    def logprior_alpha_vec(self, z_alpha: tf.Tensor) -> tf.Tensor:
        """Return the per-product Normal prior contribution for z_alpha."""
        return self._lp0_alpha - 0.5 * tf.square(z_alpha / self.sigma_z_alpha)

    @tf.function(jit_compile=True)
    def logprior_v_vec(self, z_v: tf.Tensor) -> tf.Tensor:
        """Return the per-product Normal prior contribution for z_v."""
        return self._lp0_v - 0.5 * tf.square(z_v / self.sigma_z_v)

    @tf.function(jit_compile=True)
    def logprior_fc_vec(self, z_fc: tf.Tensor) -> tf.Tensor:
        """Return the per-product Normal prior contribution for z_fc."""
        return self._lp0_fc - 0.5 * tf.square(z_fc / self.sigma_z_fc)

    @tf.function(jit_compile=True)
    def logprior_u_scale_vec(self, z_u_scale: tf.Tensor) -> tf.Tensor:
        """Return the per-market Normal prior contribution for z_u_scale."""
        return self._lp0_u_scale - 0.5 * tf.square(z_u_scale / self.sigma_z_u_scale)

    @tf.function(jit_compile=True)
    def logprior(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total prior contribution."""
        return (
            self.logprior_beta(z_beta)
            + tf.reduce_sum(self.logprior_alpha_vec(z_alpha))
            + tf.reduce_sum(self.logprior_v_vec(z_v))
            + tf.reduce_sum(self.logprior_fc_vec(z_fc))
            + tf.reduce_sum(self.logprior_u_scale_vec(z_u_scale))
        )

    @tf.function(jit_compile=True)
    def loglik_mnj(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return log-likelihood contributions for each (m, n, j)."""
        M = tf.shape(self.a_mnjt)[0]
        N = tf.shape(self.a_mnjt)[1]
        J = tf.shape(self.a_mnjt)[2]
        T = tf.shape(self.a_mnjt)[3]
        I = tf.shape(self.pi_I0)[0]

        ccp_buy = self._ccp_buy_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        pi0 = tf.broadcast_to(self.pi_I0[None, None, None, :], tf.stack([M, N, J, I]))
        ll0 = tf.zeros((M, N, J), dtype=tf.float64)

        def cond(t: tf.Tensor, ll_acc: tf.Tensor, pi_acc: tf.Tensor) -> tf.Tensor:
            return t < T

        def body(
            t: tf.Tensor,
            ll_acc: tf.Tensor,
            pi_acc: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            _, action_prob, pi_next = self._filter_step_core(
                t=t,
                pi_acc=pi_acc,
                ccp_buy=ccp_buy,
            )
            return t + 1, ll_acc + tf.math.log(action_prob), pi_next

        _, ll_final, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(tf.constant(0, dtype=tf.int32), ll0, pi0),
        )
        return ll_final

    @tf.function(jit_compile=True)
    def loglik(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total log likelihood after integrating out inventory."""
        return tf.reduce_sum(self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale))

    @tf.function(jit_compile=True)
    def beta_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the scalar block log posterior for beta."""
        return self.loglik(z_beta, z_alpha, z_v, z_fc, z_u_scale) + self.logprior_beta(
            z_beta
        )

    @tf.function(jit_compile=True)
    def alpha_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for alpha."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_alpha_vec(z_alpha)

    @tf.function(jit_compile=True)
    def v_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for v."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_v_vec(z_v)

    @tf.function(jit_compile=True)
    def fc_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for fc."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_fc_vec(z_fc)

    @tf.function(jit_compile=True)
    def u_scale_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-market block log posterior for u_scale."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[1, 2]) + self.logprior_u_scale_vec(z_u_scale)

    @tf.function(jit_compile=True)
    def joint_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the full joint log posterior."""
        return self.loglik(z_beta, z_alpha, z_v, z_fc, z_u_scale) + self.logprior(
            z_beta=z_beta,
            z_alpha=z_alpha,
            z_v=z_v,
            z_fc=z_fc,
            z_u_scale=z_u_scale,
        )

    @tf.function(jit_compile=True)
    def predict_p_buy_mnjt(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return filtered predictive buy probabilities over observed periods."""
        M = tf.shape(self.a_mnjt)[0]
        N = tf.shape(self.a_mnjt)[1]
        J = tf.shape(self.a_mnjt)[2]
        T = tf.shape(self.a_mnjt)[3]
        I = tf.shape(self.pi_I0)[0]

        ccp_buy = self._ccp_buy_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        pi0 = tf.broadcast_to(self.pi_I0[None, None, None, :], tf.stack([M, N, J, I]))
        out = tf.TensorArray(dtype=tf.float64, size=T)

        def cond(
            t: tf.Tensor,
            pi_acc: tf.Tensor,
            out_acc: tf.TensorArray,
        ) -> tf.Tensor:
            return t < T

        def body(
            t: tf.Tensor,
            pi_acc: tf.Tensor,
            out_acc: tf.TensorArray,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.TensorArray]:
            p_buy_I, _, pi_next = self._filter_step_core(
                t=t,
                pi_acc=pi_acc,
                ccp_buy=ccp_buy,
            )
            p_hat = tf.reduce_sum(pi_acc * p_buy_I, axis=3)
            return t + 1, pi_next, out_acc.write(t, p_hat)

        _, _, out_final = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(tf.constant(0, dtype=tf.int32), pi0, out),
        )
        return tf.transpose(out_final.stack(), perm=[1, 2, 3, 0])
