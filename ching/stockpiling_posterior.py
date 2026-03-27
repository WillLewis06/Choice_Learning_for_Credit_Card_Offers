from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from ching.stockpiling_model import InventoryMaps, solve_ccp_buy, unconstrained_to_theta

__all__ = [
    "StockpilingPosteriorConfig",
    "StockpilingPosteriorTF",
]

ONE_F64 = tf.constant(1.0, dtype=tf.float64)


@dataclass(frozen=True)
class StockpilingPosteriorConfig:
    """Store fixed hyperparameters and numerical settings for posterior evaluation."""

    tol: float
    max_iter: int
    eps: float

    sigma_z_beta: float
    sigma_z_alpha: float
    sigma_z_v: float
    sigma_z_fc: float
    sigma_z_u_scale: float

    fix_u_scale: bool
    fixed_z_u_scale: float


class StockpilingPosteriorTF:
    """Evaluate posterior terms for the Ching-style stockpiling sampler."""

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
        pi_I0: tf.Tensor,
    ):
        """Cache fixed observed tensors, prior scales, and numerical constants."""

        self.tol = float(config.tol)
        self.max_iter = int(config.max_iter)
        self.eps = tf.constant(config.eps, dtype=tf.float64)

        self.sigma_z_beta = tf.constant(config.sigma_z_beta, dtype=tf.float64)
        self.sigma_z_alpha = tf.constant(config.sigma_z_alpha, dtype=tf.float64)
        self.sigma_z_v = tf.constant(config.sigma_z_v, dtype=tf.float64)
        self.sigma_z_fc = tf.constant(config.sigma_z_fc, dtype=tf.float64)
        self.sigma_z_u_scale = tf.constant(config.sigma_z_u_scale, dtype=tf.float64)

        self.fix_u_scale = bool(config.fix_u_scale)
        self.fixed_z_u_scale = tf.constant(config.fixed_z_u_scale, dtype=tf.float64)

        self.a_mnjt = a_mnjt
        self.s_mjt = s_mjt
        self.u_mj = u_mj
        self.P_price_mj = P_price_mj
        self.price_vals_mj = price_vals_mj
        self.lambda_mn = lambda_mn
        self.waste_cost = waste_cost
        self.pi_I0 = tf.reshape(pi_I0, (-1,))

        self.inventory_maps = inventory_maps
        _, _, _, idx_down, idx_up = inventory_maps
        self.idx_down = idx_down
        self.idx_up = idx_up

        self.lambda_mn_11 = self.lambda_mn[:, :, None, None]

        self._log_two_pi = tf.math.log(
            tf.constant(2.0 * 3.141592653589793, dtype=tf.float64)
        )
        self._lp0_beta = -0.5 * (
            self._log_two_pi + 2.0 * tf.math.log(self.sigma_z_beta)
        )
        self._lp0_alpha = -0.5 * (
            self._log_two_pi + 2.0 * tf.math.log(self.sigma_z_alpha)
        )
        self._lp0_v = -0.5 * (self._log_two_pi + 2.0 * tf.math.log(self.sigma_z_v))
        self._lp0_fc = -0.5 * (self._log_two_pi + 2.0 * tf.math.log(self.sigma_z_fc))
        self._lp0_u_scale = -0.5 * (
            self._log_two_pi + 2.0 * tf.math.log(self.sigma_z_u_scale)
        )

    def _effective_z_u_scale(self, z_u_scale: tf.Tensor) -> tf.Tensor:
        """Return the active z_u_scale state, possibly fixed for this run."""
        if self.fix_u_scale:
            return tf.zeros_like(z_u_scale, dtype=tf.float64) + self.fixed_z_u_scale
        return z_u_scale

    def _theta_from_z(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Map unconstrained sampler state to constrained structural parameters."""
        return unconstrained_to_theta(
            {
                "z_beta": z_beta,
                "z_alpha": z_alpha,
                "z_v": z_v,
                "z_fc": z_fc,
                "z_u_scale": self._effective_z_u_scale(z_u_scale),
            }
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _ccp_buy_from_z(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Solve buy CCPs for the current unconstrained sampler state."""
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

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _shift_down(self, post: tf.Tensor) -> tf.Tensor:
        """Map mass via I' = max(I-1, 0)."""
        I = tf.shape(self.idx_up)[0]

        def case_I1() -> tf.Tensor:
            return post

        def case_Igt1() -> tf.Tensor:
            base = tf.gather(post, self.idx_up, axis=-1)
            first = base[..., :1] + post[..., :1]
            mid = base[..., 1:-1]
            last = base[..., -1:] - post[..., -1:]
            return tf.concat([first, mid, last], axis=-1)

        return tf.cond(tf.equal(I, 1), case_I1, case_Igt1)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _shift_up(self, post: tf.Tensor) -> tf.Tensor:
        """Map mass via I' = min(I+1, I_max)."""
        I = tf.shape(self.idx_down)[0]

        def case_I1() -> tf.Tensor:
            return post

        def case_Igt1() -> tf.Tensor:
            base = tf.gather(post, self.idx_down, axis=-1)
            first = base[..., :1] - post[..., :1]
            mid = base[..., 1:-1]
            last = base[..., -1:] + post[..., -1:]
            return tf.concat([first, mid, last], axis=-1)

        return tf.cond(tf.equal(I, 1), case_I1, case_Igt1)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _transition_inventory(self, post: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
        """Propagate the inventory distribution one step forward."""
        lam = self.lambda_mn_11
        one_minus = ONE_F64 - lam

        down = self._shift_down(post)
        up = self._shift_up(post)

        pi0 = one_minus * post + lam * down
        pi1 = lam * post + one_minus * up
        return tf.where(tf.equal(a_t[..., None], 1), pi1, pi0)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _filter_step_core(
        self,
        t: tf.Tensor,
        pi_acc: tf.Tensor,
        a_mnjt: tf.Tensor,
        s_mjt: tf.Tensor,
        ccp_buy: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform one forward-filter step over latent inventory."""
        mnj_shape = tf.shape(pi_acc)[:3]
        s_t_mj = s_mjt[:, :, t]
        s_idx = tf.broadcast_to(s_t_mj[:, None, :], mnj_shape)

        p_buy_I = tf.gather(ccp_buy, s_idx, axis=3, batch_dims=3)
        a_t_mnj = a_mnjt[:, :, :, t]
        emission = tf.where(tf.equal(a_t_mnj[..., None], 1), p_buy_I, ONE_F64 - p_buy_I)

        numer = pi_acc * emission
        denom = tf.reduce_sum(numer, axis=3)
        denom_safe = tf.maximum(denom, self.eps)
        post = numer / denom_safe[..., None]
        pi_next = self._transition_inventory(post=post, a_t=a_t_mnj)
        return p_buy_I, denom_safe, pi_next

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_beta(self, z_beta: tf.Tensor) -> tf.Tensor:
        """Return the scalar Normal prior term for z_beta."""
        return tf.reduce_sum(
            self._lp0_beta - 0.5 * tf.square(z_beta / self.sigma_z_beta)
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_alpha_vec(self, z_alpha: tf.Tensor) -> tf.Tensor:
        """Return the elementwise Normal prior terms for z_alpha."""
        return self._lp0_alpha - 0.5 * tf.square(z_alpha / self.sigma_z_alpha)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_v_vec(self, z_v: tf.Tensor) -> tf.Tensor:
        """Return the elementwise Normal prior terms for z_v."""
        return self._lp0_v - 0.5 * tf.square(z_v / self.sigma_z_v)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_fc_vec(self, z_fc: tf.Tensor) -> tf.Tensor:
        """Return the elementwise Normal prior terms for z_fc."""
        return self._lp0_fc - 0.5 * tf.square(z_fc / self.sigma_z_fc)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior_u_scale_vec(self, z_u_scale: tf.Tensor) -> tf.Tensor:
        """Return the elementwise Normal prior terms for z_u_scale."""
        z_use = self._effective_z_u_scale(z_u_scale)
        if self.fix_u_scale:
            return tf.zeros_like(z_use, dtype=tf.float64)
        return self._lp0_u_scale - 0.5 * tf.square(z_use / self.sigma_z_u_scale)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def logprior(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total prior contribution across all unconstrained blocks."""
        return (
            self.logprior_beta(z_beta)
            + tf.reduce_sum(self.logprior_alpha_vec(z_alpha))
            + tf.reduce_sum(self.logprior_v_vec(z_v))
            + tf.reduce_sum(self.logprior_fc_vec(z_fc))
            + tf.reduce_sum(self.logprior_u_scale_vec(z_u_scale))
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik_mnj(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return log-likelihood contributions per (m,n,j)."""
        a_mnjt = self.a_mnjt
        M = tf.shape(a_mnjt)[0]
        N = tf.shape(a_mnjt)[1]
        J = tf.shape(a_mnjt)[2]
        T = tf.shape(a_mnjt)[3]
        I = tf.shape(self.idx_up)[0]

        ccp_buy = self._ccp_buy_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        pi = tf.broadcast_to(self.pi_I0[None, None, None, :], tf.stack([M, N, J, I]))
        ll0 = tf.zeros((M, N, J), dtype=tf.float64)
        t0 = tf.constant(0, dtype=tf.int32)

        def cond(t: tf.Tensor, ll_acc: tf.Tensor, pi_acc: tf.Tensor) -> tf.Tensor:
            return t < T

        def body(
            t: tf.Tensor,
            ll_acc: tf.Tensor,
            pi_acc: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            _, denom_safe, pi_next = self._filter_step_core(
                t=t,
                pi_acc=pi_acc,
                a_mnjt=self.a_mnjt,
                s_mjt=self.s_mjt,
                ccp_buy=ccp_buy,
            )
            ll_acc = ll_acc + tf.math.log(denom_safe)
            return t + 1, ll_acc, pi_next

        _, ll_final, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(t0, ll0, pi),
        )
        return ll_final

    @tf.function(jit_compile=True, reduce_retracing=True)
    def loglik(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the total log-likelihood after integrating out inventory."""
        return tf.reduce_sum(self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale))

    @tf.function(jit_compile=True, reduce_retracing=True)
    def beta_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the scalar block log posterior for z_beta."""
        return self.loglik(z_beta, z_alpha, z_v, z_fc, z_u_scale) + self.logprior_beta(
            z_beta
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def alpha_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for z_alpha."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_alpha_vec(z_alpha)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def v_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for z_v."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_v_vec(z_v)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def fc_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-product block log posterior for z_fc."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[0, 1]) + self.logprior_fc_vec(z_fc)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def u_scale_block_logpost(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return the per-market block log posterior for z_u_scale."""
        ll_mnj = self.loglik_mnj(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        return tf.reduce_sum(ll_mnj, axis=[1, 2]) + self.logprior_u_scale_vec(z_u_scale)

    @tf.function(jit_compile=True, reduce_retracing=True)
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
            z_beta, z_alpha, z_v, z_fc, z_u_scale
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def predict_p_buy_mnjt(
        self,
        z_beta: tf.Tensor,
        z_alpha: tf.Tensor,
        z_v: tf.Tensor,
        z_fc: tf.Tensor,
        z_u_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Return predictive buy probabilities given filtered latent inventory."""
        a_mnjt = self.a_mnjt
        M = tf.shape(a_mnjt)[0]
        N = tf.shape(a_mnjt)[1]
        J = tf.shape(a_mnjt)[2]
        T = tf.shape(a_mnjt)[3]
        I = tf.shape(self.idx_up)[0]

        ccp_buy = self._ccp_buy_from_z(z_beta, z_alpha, z_v, z_fc, z_u_scale)
        pi = tf.broadcast_to(self.pi_I0[None, None, None, :], tf.stack([M, N, J, I]))
        out_ta = tf.TensorArray(dtype=tf.float64, size=T)
        t0 = tf.constant(0, dtype=tf.int32)

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
                a_mnjt=self.a_mnjt,
                s_mjt=self.s_mjt,
                ccp_buy=ccp_buy,
            )
            p_hat = tf.reduce_sum(pi_acc * p_buy_I, axis=3)
            out_acc = out_acc.write(t, p_hat)
            return t + 1, pi_next, out_acc

        _, _, out_final = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(t0, pi, out_ta),
        )
        p_buy_tmnj = out_final.stack()
        return tf.transpose(p_buy_tmnj, perm=[1, 2, 3, 0])
