"""
stockpiling_posterior.py

Likelihood and prediction utilities for the Phase-3 stockpiling model.

This module is intentionally stateless:
  - `inputs` is a typed mapping of fixed/known tensors and hyperparameters.
  - the current constrained parameter state is passed explicitly via `theta`.

Observed data:
  - purchases a_mnjt: (M,N,J,T) int32 in {0,1}
  - price states s_mjt: (M,J,T) int32 in {0,...,S-1}

Latent state:
  - per-(m,n,j) inventory I_t in {0,...,I_max}
  - per-period consumption is integrated out via known lambda_mn: (M,N) float64

Likelihood:
  - inventory is marginalized with a forward filter over the finite inventory state.

Required `inputs` keys:
  - a_mnjt: tf.Tensor (M,N,J,T) int32
  - s_mjt: tf.Tensor (M,J,T) int32
  - u_mj: tf.Tensor (M,J) float64
  - P_price_mj: tf.Tensor (M,J,S,S) float64
  - price_vals_mj: tf.Tensor (M,J,S) float64
  - lambda_mn: tf.Tensor (M,N) float64, in (0,1)
  - waste_cost: tf.Tensor () float64
  - inventory_maps: tuple returned by build_inventory_maps(I_max)
  - tol: float
  - max_iter: int
  - pi_I0: tf.Tensor (I_max+1,) float64 initial inventory distribution (sums to 1)

Notes:
  - This module does not clamp or repair invalid inputs. All external inputs must be
    validated upstream (configs / input validation).
"""

from __future__ import annotations

import math
from typing import TypedDict

import tensorflow as tf

from ching.stockpiling_model import solve_ccp_buy

__all__ = [
    "InventoryMaps",
    "StockpilingInputs",
    "logprior_normal",
    "loglik_mnj_from_theta",
    "predict_p_buy_mnjt_from_theta",
]

# (I_vals, stockout_mask, at_cap_mask, idx_down, idx_up)
InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]

TWO_PI_F64 = tf.constant(2.0 * math.pi, dtype=tf.float64)
ONE_F64 = tf.constant(1.0, dtype=tf.float64)
EPS_F64 = tf.constant(1e-12, dtype=tf.float64)


class StockpilingInputs(TypedDict):
    """Typed container for fixed inputs required by the stockpiling posterior."""

    a_mnjt: tf.Tensor
    s_mjt: tf.Tensor
    u_mj: tf.Tensor
    P_price_mj: tf.Tensor
    price_vals_mj: tf.Tensor
    lambda_mn: tf.Tensor
    waste_cost: tf.Tensor
    inventory_maps: InventoryMaps
    tol: float
    max_iter: int
    pi_I0: tf.Tensor


def logprior_normal(z: tf.Tensor, sigma_z: tf.Tensor) -> tf.Tensor:
    """Elementwise Normal(0, sigma_z^2) log density.

    Args:
      z: float64 tensor
      sigma_z: float64 tensor broadcastable to z, strictly positive

    Returns:
      logp: float64 tensor with the same shape as z
    """
    const = -0.5 * tf.math.log(TWO_PI_F64)
    return const - tf.math.log(sigma_z) - 0.5 * tf.square(z / sigma_z)


def _inventory_maps(inputs: StockpilingInputs) -> InventoryMaps:
    """Return precomputed inventory maps."""
    return inputs["inventory_maps"]


def _ccp_buy_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """Compute buy CCPs for the given theta and fixed inputs.

    Returns:
      ccp_buy: (M,N,J,S,I) float64
    """
    maps = _inventory_maps(inputs)

    ccp_buy, _, _ = solve_ccp_buy(
        u_mj=inputs["u_mj"],
        price_vals_mj=inputs["price_vals_mj"],
        P_price_mj=inputs["P_price_mj"],
        theta=theta,
        lambda_mn=inputs["lambda_mn"],
        waste_cost=inputs["waste_cost"],
        maps=maps,
        tol=float(inputs["tol"]),
        max_iter=int(inputs["max_iter"]),
    )
    return ccp_buy


def _shift_down(post: tf.Tensor, idx_up: tf.Tensor) -> tf.Tensor:
    """Map mass via I' = max(I-1, 0) (down-shift with absorption at 0)."""
    I = tf.shape(idx_up)[0]

    def case_I1() -> tf.Tensor:
        return post

    def case_Igt1() -> tf.Tensor:
        base = tf.gather(post, idx_up, axis=-1)  # (..., I)
        first = base[..., :1] + post[..., :1]
        mid = base[..., 1:-1]
        last = base[..., -1:] - post[..., -1:]
        return tf.concat([first, mid, last], axis=-1)

    return tf.cond(tf.equal(I, 1), case_I1, case_Igt1)


def _shift_up(post: tf.Tensor, idx_down: tf.Tensor) -> tf.Tensor:
    """Map mass via I' = min(I+1, I_max) (up-shift with absorption at I_max)."""
    I = tf.shape(idx_down)[0]

    def case_I1() -> tf.Tensor:
        return post

    def case_Igt1() -> tf.Tensor:
        base = tf.gather(post, idx_down, axis=-1)  # (..., I)
        first = base[..., :1] - post[..., :1]
        mid = base[..., 1:-1]
        last = base[..., -1:] + post[..., -1:]
        return tf.concat([first, mid, last], axis=-1)

    return tf.cond(tf.equal(I, 1), case_I1, case_Igt1)


def _transition_inventory(
    post: tf.Tensor,
    lambda_mn_11: tf.Tensor,
    a_t: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
) -> tf.Tensor:
    """Inventory transition pi_{t+1} from post(I_t | history, a_t) by marginalizing c_t."""
    lam = lambda_mn_11  # (M,N,1,1)
    one_minus = ONE_F64 - lam

    down = _shift_down(post, idx_up)  # I' = max(I-1, 0)
    up = _shift_up(post, idx_down)  # I' = min(I+1, I_max)

    # a=0: I' = I - c
    pi0 = one_minus * post + lam * down
    # a=1: I' = I + 1 - c
    pi1 = lam * post + one_minus * up

    return tf.where(tf.equal(a_t[..., None], 1), pi1, pi0)


def _filter_step_core(
    t: tf.Tensor,
    pi_acc: tf.Tensor,
    a_mnjt: tf.Tensor,
    s_mjt: tf.Tensor,
    ccp_buy: tf.Tensor,
    lambda_mn_11: tf.Tensor,
    idx_down: tf.Tensor,
    idx_up: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """One forward-filter step.

    Returns:
      p_buy_I:    (M,N,J,I) float64  CCP evaluated at current price state s_t
      denom_safe: (M,N,J) float64    P(a_t | history)
      pi_next:    (M,N,J,I) float64  next-period inventory prior
    """
    mnjs_shape = tf.shape(pi_acc)[:3]  # (3,) = (M,N,J)

    s_t_mj = s_mjt[:, :, t]  # (M,J)
    s_idx = tf.broadcast_to(s_t_mj[:, None, :], mnjs_shape)  # (M,N,J)

    # CCP evaluated at current price state: p_buy_I[m,n,j,I] = CCP[m,n,j,s_t,I]
    p_buy_I = tf.gather(ccp_buy, s_idx, axis=3, batch_dims=3)  # (M,N,J,I)

    a_t_mnj = a_mnjt[:, :, :, t]  # (M,N,J)
    e = tf.where(
        tf.equal(a_t_mnj[..., None], 1), p_buy_I, ONE_F64 - p_buy_I
    )  # (M,N,J,I)

    # Bayes update over inventory I_t
    numer = pi_acc * e
    denom = tf.reduce_sum(numer, axis=3)  # (M,N,J)
    denom_safe = tf.maximum(denom, EPS_F64)
    post = numer / denom_safe[..., None]

    # Propagate inventory prior using the structural transition and known consumption rate
    pi_next = _transition_inventory(
        post=post,
        lambda_mn_11=lambda_mn_11,
        a_t=a_t_mnj,
        idx_down=idx_down,
        idx_up=idx_up,
    )
    return p_buy_I, denom_safe, pi_next


def loglik_mnj_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """Log-likelihood per (m,n,j), integrating out latent inventory.

    Returns:
      ll_mnj: (M,N,J) float64
    """
    a_mnjt = inputs["a_mnjt"]
    s_mjt = inputs["s_mjt"]

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    _, _, _, idx_down, idx_up = _inventory_maps(inputs)
    I = tf.shape(idx_up)[0]

    ccp_buy = _ccp_buy_from_theta(theta=theta, inputs=inputs)  # (M,N,J,S,I)

    pi0 = tf.reshape(inputs["pi_I0"], (-1,))  # (I,)
    pi = tf.broadcast_to(pi0[None, None, None, :], tf.stack([M, N, J, I]))  # (M,N,J,I)

    lambda_mn_11 = inputs["lambda_mn"][:, :, None, None]  # (M,N,1,1)

    ll0 = tf.zeros((M, N, J), dtype=tf.float64)
    t0 = tf.constant(0, dtype=tf.int32)

    def body(
        t: tf.Tensor, ll_acc: tf.Tensor, pi_acc: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        _, denom_safe, pi_next = _filter_step_core(
            t=t,
            pi_acc=pi_acc,
            a_mnjt=a_mnjt,
            s_mjt=s_mjt,
            ccp_buy=ccp_buy,
            lambda_mn_11=lambda_mn_11,
            idx_down=idx_down,
            idx_up=idx_up,
        )
        ll_acc = ll_acc + tf.math.log(denom_safe)
        return t + 1, ll_acc, pi_next

    _, ll_final, _ = tf.while_loop(
        cond=lambda t, ll_acc, pi_acc: t < T,
        body=body,
        loop_vars=(t0, ll0, pi),
    )
    return ll_final


def predict_p_buy_mnjt_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """Predict P(a_t=1 | s_1:t, a_1:t-1) for each (m,n,j,t), integrating out inventory.

    Returns:
      p_buy_mnjt: (M,N,J,T) float64
    """
    a_mnjt = inputs["a_mnjt"]
    s_mjt = inputs["s_mjt"]

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    _, _, _, idx_down, idx_up = _inventory_maps(inputs)
    I = tf.shape(idx_up)[0]

    ccp_buy = _ccp_buy_from_theta(theta=theta, inputs=inputs)  # (M,N,J,S,I)

    pi0 = tf.reshape(inputs["pi_I0"], (-1,))  # (I,)
    pi = tf.broadcast_to(pi0[None, None, None, :], tf.stack([M, N, J, I]))  # (M,N,J,I)

    lambda_mn_11 = inputs["lambda_mn"][:, :, None, None]  # (M,N,1,1)

    out_ta = tf.TensorArray(tf.float64, size=T)
    t0 = tf.constant(0, dtype=tf.int32)

    def body(
        t: tf.Tensor, pi_acc: tf.Tensor, out_acc: tf.TensorArray
    ) -> tuple[tf.Tensor, tf.Tensor, tf.TensorArray]:
        p_buy_I, _, pi_next = _filter_step_core(
            t=t,
            pi_acc=pi_acc,
            a_mnjt=a_mnjt,
            s_mjt=s_mjt,
            ccp_buy=ccp_buy,
            lambda_mn_11=lambda_mn_11,
            idx_down=idx_down,
            idx_up=idx_up,
        )

        # Predictive probability of buying at t given history up to t-1:
        # p_hat = sum_I pi_t(I) * CCP(s_t, I)
        p_hat = tf.reduce_sum(pi_acc * p_buy_I, axis=3)  # (M,N,J)
        out_acc = out_acc.write(t, p_hat)
        return t + 1, pi_next, out_acc

    _, _, out_final = tf.while_loop(
        cond=lambda t, pi_acc, out_acc: t < T,
        body=body,
        loop_vars=(t0, pi, out_ta),
    )

    p_buy_tmnj = out_final.stack()  # (T,M,N,J)
    return tf.transpose(p_buy_tmnj, perm=[1, 2, 3, 0])
