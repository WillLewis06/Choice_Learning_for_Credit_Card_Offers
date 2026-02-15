"""
stockpiling_posterior.py

Likelihood and prediction utilities for the Phase-3 stockpiling model.

This module is intentionally "stateless":
  - `inputs` is a plain dict-like mapping of fixed/known tensors and hyperparameters.
  - The current parameter state is passed explicitly via `theta` (already constrained).

Observed data:
  - purchases a_mnjt: (M,N,J,T) in {0,1}
  - price states s_mjt: (M,J,T) in {0,...,S-1}

Latent state:
  - per-(m,n,j) inventory I_t in {0,...,I_max}
  - per-period consumption is integrated out via the known lambda_mn: (M,N)

Likelihood:
  - inventory is marginalized via a forward filter over the finite inventory state.

Expected `inputs` keys (dict):
  Required:
    - "a_mnjt": tf.Tensor (M,N,J,T)
    - "s_mjt": tf.Tensor (M,J,T)
    - "u_mj": tf.Tensor (M,J)
    - "P_price_mj": tf.Tensor (M,J,S,S)
    - "price_vals_mj": tf.Tensor (M,J,S)
    - "lambda_mn": tf.Tensor (M,N)
    - "I_max": int
    - "waste_cost": float
  Optional:
    - "tol": float (default 1e-6)
    - "max_iter": int (default 2000)
    - "init_I_dist": tf.Tensor (I_max+1,) initial inventory distribution
    - "inventory_maps": tuple of tensors returned by build_inventory_maps(I_max)
    - "use_ccp_cache": bool (default True)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from ching.stockpiling_model import (
    build_inventory_maps,
    solve_ccp_buy,
    solve_ccp_buy_cached,
)

__all__ = [
    "StockpilingInputs",
    "InventoryMaps",
    "logprior_normal",
    "loglik_mnj_from_theta",
    "predict_p_buy_mnjt_from_theta",
]

InventoryMaps = tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
StockpilingInputs = Mapping[str, Any]


def logprior_normal(z: tf.Tensor, sigma_z: float | tf.Tensor) -> tf.Tensor:
    """
    Elementwise Normal(0, sigma_z^2) log-density, returned with the same shape as `z`.
    """
    z = tf.convert_to_tensor(z, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma_z, dtype=z.dtype)
    const = -0.5 * tf.math.log(tf.constant(2.0 * np.pi, dtype=z.dtype))
    return const - tf.math.log(sigma) - 0.5 * tf.square(z / sigma)


def _inventory_maps(inputs: StockpilingInputs) -> InventoryMaps:
    """
    Return inventory maps, building them if absent.

    The maps are a tuple of tensors used by the CCP solver to vectorize
    inventory transitions and boundary conditions.
    """
    maps = inputs.get("inventory_maps", None)
    if maps is not None:
        return maps
    return build_inventory_maps(int(inputs["I_max"]))


def _ccp_buy_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """
    Compute buy CCPs for the given theta and fixed inputs.

    Returns:
      ccp_buy: (M,N,J,S,I) float64
    """
    maps = _inventory_maps(inputs)

    u_mj = tf.convert_to_tensor(inputs["u_mj"], dtype=tf.float64)
    price_vals_mj = tf.convert_to_tensor(inputs["price_vals_mj"], dtype=tf.float64)
    P_price_mj = tf.convert_to_tensor(inputs["P_price_mj"], dtype=tf.float64)
    lambda_mn = tf.convert_to_tensor(inputs["lambda_mn"], dtype=tf.float64)
    waste_cost = tf.convert_to_tensor(inputs["waste_cost"], dtype=tf.float64)

    tol = float(inputs.get("tol", 1e-6))
    max_iter = int(inputs.get("max_iter", 2000))
    use_ccp_cache = bool(inputs.get("use_ccp_cache", True))

    if use_ccp_cache:
        ccp_buy, _, _ = solve_ccp_buy_cached(
            u_mj=u_mj,
            price_vals_mj=price_vals_mj,
            P_price_mj=P_price_mj,
            theta=theta,
            lambda_mn=lambda_mn,
            waste_cost=waste_cost,
            maps=maps,
            tol=tol,
            max_iter=max_iter,
            use_cache=True,
        )
    else:
        ccp_buy, _, _ = solve_ccp_buy(
            u_mj=u_mj,
            price_vals_mj=price_vals_mj,
            P_price_mj=P_price_mj,
            theta=theta,
            lambda_mn=lambda_mn,
            waste_cost=waste_cost,
            maps=maps,
            tol=tol,
            max_iter=max_iter,
        )
    return ccp_buy


def _init_inventory_dist(inputs: StockpilingInputs) -> tf.Tensor:
    """
    Return an initial inventory distribution pi0 over I in {0,...,I_max}.

    Output:
      pi0: (I_max+1,) float64, sums to 1.
    """
    I = int(inputs["I_max"]) + 1
    init = inputs.get("init_I_dist", None)

    if init is None:
        pi0 = tf.fill((I,), 1.0 / float(I))
        return tf.cast(pi0, tf.float64)

    pi0 = tf.reshape(tf.cast(init, tf.float64), (I,))
    s = tf.reduce_sum(pi0)

    uniform = tf.fill((I,), tf.cast(1.0 / float(I), pi0.dtype))
    pi0 = tf.where(s > tf.cast(0.0, pi0.dtype), pi0 / s, uniform)
    return pi0


def _transition_inventory(
    post: tf.Tensor,
    lambda_mn: tf.Tensor,
    a_t: tf.Tensor,
    I_max: int,
) -> tf.Tensor:
    """
    Inventory transition: pi_{t+1} from post(I_t | history, a_t) by marginalizing c_t.

    Args:
      post:      (M,N,J,I) posterior over I_t after observing a_t
      lambda_mn: (M,N) consumption probability
      a_t:       (M,N,J) observed action in {0,1}
      I_max:     inventory cap

    Returns:
      pi_next: (M,N,J,I)
    """
    if I_max == 0:
        return tf.convert_to_tensor(post, dtype=tf.float64)

    post = tf.convert_to_tensor(post, dtype=tf.float64)
    lam = tf.clip_by_value(tf.cast(lambda_mn, tf.float64), 0.0, 1.0)[:, :, None, None]
    one_minus = 1.0 - lam

    I = I_max + 1
    idx = tf.range(I, dtype=tf.int32)
    idx_up = tf.minimum(idx + 1, I_max)
    idx_down = tf.maximum(idx - 1, 0)

    g_up = tf.gather(post, idx_up, axis=3)
    g_down = tf.gather(post, idx_down, axis=3)

    # a=0: I_{t+1} = clip(I_t - c_t, 0, I_max)
    base0 = one_minus * post + lam * g_up
    first0 = base0[..., :1] + lam * post[..., :1]
    mid0 = base0[..., 1:I_max]
    last0 = base0[..., I_max : I_max + 1] - lam * post[..., I_max : I_max + 1]
    pi0 = tf.concat([first0, mid0, last0], axis=3)

    # a=1: I_{t+1} = clip(I_t + 1 - c_t, 0, I_max)
    base1 = lam * post + one_minus * g_down
    first1 = base1[..., :1] - one_minus * post[..., :1]
    mid1 = base1[..., 1:I_max]
    last1 = base1[..., I_max : I_max + 1] + one_minus * post[..., I_max : I_max + 1]
    pi1 = tf.concat([first1, mid1, last1], axis=3)

    a_t = tf.cast(a_t, tf.int32)[..., None]
    return tf.where(tf.equal(a_t, 1), pi1, pi0)


def loglik_mnj_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """
    Compute log-likelihood contributions per (m,n,j), marginalizing over inventory.

    Returns:
      ll_mnj: (M,N,J) float64, sum over t of log P(a_t | s_1:t, a_1:t-1).
    """
    a_mnjt = tf.cast(inputs["a_mnjt"], tf.int32)
    s_mjt = tf.cast(inputs["s_mjt"], tf.int32)
    tf.debugging.assert_rank(a_mnjt, 4)
    tf.debugging.assert_rank(s_mjt, 3)

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    I_max = int(inputs["I_max"])
    I = I_max + 1

    ccp_buy = _ccp_buy_from_theta(theta=theta, inputs=inputs)  # (M,N,J,S,I)

    pi0 = _init_inventory_dist(inputs)  # (I,)
    pi = tf.broadcast_to(pi0[None, None, None, :], tf.stack([M, N, J, I]))  # (M,N,J,I)

    ll = tf.zeros((M, N, J), dtype=tf.float64)
    eps = tf.constant(1e-12, dtype=tf.float64)

    lambda_mn = tf.convert_to_tensor(inputs["lambda_mn"], dtype=tf.float64)

    def body(
        t: tf.Tensor, ll_acc: tf.Tensor, pi_acc: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        s_t_mj = s_mjt[:, :, t]  # (M,J)
        s_idx = tf.broadcast_to(s_t_mj[:, None, :], tf.stack([M, N, J]))  # (M,N,J)

        # (M,N,J,I)
        p_buy = tf.gather(ccp_buy, s_idx, axis=3, batch_dims=3)

        a_t_mnj = a_mnjt[:, :, :, t]  # (M,N,J)
        e = tf.where(tf.equal(a_t_mnj[..., None], 1), p_buy, 1.0 - p_buy)

        numer = pi_acc * tf.cast(e, tf.float64)
        denom = tf.reduce_sum(numer, axis=3)  # (M,N,J)
        denom_safe = tf.maximum(denom, eps)

        ll_acc = ll_acc + tf.math.log(denom_safe)
        post = numer / denom_safe[..., None]

        pi_next = _transition_inventory(
            post=post, lambda_mn=lambda_mn, a_t=a_t_mnj, I_max=I_max
        )
        return t + 1, ll_acc, pi_next

    t0 = tf.constant(0, dtype=tf.int32)
    _, ll_final, _ = tf.while_loop(
        cond=lambda t, ll_acc, pi_acc: t < T,
        body=body,
        loop_vars=(t0, ll, pi),
        parallel_iterations=1,
    )
    return ll_final


def predict_p_buy_mnjt_from_theta(
    theta: dict[str, tf.Tensor], inputs: StockpilingInputs
) -> tf.Tensor:
    """
    Predict P(a_t=1 | s_1:t, a_1:t-1) for each (m,n,j,t), integrating out latent inventory.

    Returns:
      p_buy_mnjt: (M,N,J,T) float64
    """
    a_mnjt = tf.cast(inputs["a_mnjt"], tf.int32)
    s_mjt = tf.cast(inputs["s_mjt"], tf.int32)

    M = tf.shape(a_mnjt)[0]
    N = tf.shape(a_mnjt)[1]
    J = tf.shape(a_mnjt)[2]
    T = tf.shape(a_mnjt)[3]

    I_max = int(inputs["I_max"])
    I = I_max + 1

    ccp_buy = _ccp_buy_from_theta(theta=theta, inputs=inputs)  # (M,N,J,S,I)

    pi0 = _init_inventory_dist(inputs)  # (I,)
    pi = tf.broadcast_to(pi0[None, None, None, :], tf.stack([M, N, J, I]))  # (M,N,J,I)

    lambda_mn = tf.convert_to_tensor(inputs["lambda_mn"], dtype=tf.float64)
    eps = tf.constant(1e-12, dtype=tf.float64)

    out_ta = tf.TensorArray(tf.float64, size=T)

    def body(
        t: tf.Tensor, pi_acc: tf.Tensor, out_acc: tf.TensorArray
    ) -> tuple[tf.Tensor, tf.Tensor, tf.TensorArray]:
        s_t_mj = s_mjt[:, :, t]  # (M,J)
        s_idx = tf.broadcast_to(s_t_mj[:, None, :], tf.stack([M, N, J]))  # (M,N,J)

        p_buy_I = tf.gather(ccp_buy, s_idx, axis=3, batch_dims=3)  # (M,N,J,I)

        p_buy_hat = tf.reduce_sum(
            pi_acc * tf.cast(p_buy_I, tf.float64), axis=3
        )  # (M,N,J)
        out_acc = out_acc.write(t, p_buy_hat)

        a_t = a_mnjt[:, :, :, t]  # (M,N,J)
        e = tf.where(tf.equal(a_t[..., None], 1), p_buy_I, 1.0 - p_buy_I)
        numer = pi_acc * tf.cast(e, tf.float64)
        denom = tf.reduce_sum(numer, axis=3)
        denom_safe = tf.maximum(denom, eps)
        post = numer / denom_safe[..., None]

        pi_next = _transition_inventory(
            post=post, lambda_mn=lambda_mn, a_t=a_t, I_max=I_max
        )
        return t + 1, pi_next, out_acc

    t0 = tf.constant(0, dtype=tf.int32)
    _, _, out_final = tf.while_loop(
        cond=lambda t, pi_acc, out_acc: t < T,
        body=body,
        loop_vars=(t0, pi, out_ta),
        parallel_iterations=1,
    )

    p_buy_tmnj = out_final.stack()  # (T,M,N,J)
    return tf.transpose(p_buy_tmnj, perm=[1, 2, 3, 0])
