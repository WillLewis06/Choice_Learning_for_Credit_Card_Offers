from __future__ import annotations

import tensorflow as tf


def init_progress_state(shrink: LuShrinkageEstimator) -> dict:
    """
    Initialize a lightweight snapshot of the current state.
    Scalars + cheap aggregates only.
    """
    return {
        "beta_p": float(shrink.beta_p.numpy()),
        "beta_w": float(shrink.beta_w.numpy()),
        "r": float(shrink.r.numpy()),
        "E_bar_norm": float(tf.norm(shrink.E_bar).numpy()),
        "njt_norm": float(tf.norm(shrink.njt).numpy()),
        "gamma_mean": float(tf.reduce_mean(tf.cast(shrink.gamma, tf.float64)).numpy()),
        "phi_mean": float(tf.reduce_mean(shrink.phi).numpy()),
    }


def report_iteration_progress(
    shrink: LuShrinkageEstimator, it: int, prev: dict
) -> dict:
    """
    Print current state values (scalars + cheap aggregates) at end of iteration.
    Returns updated snapshot for next iteration.
    """
    beta_p = float(shrink.beta_p.numpy())
    beta_w = float(shrink.beta_w.numpy())
    r_val = float(shrink.r.numpy())
    sigma = float(tf.exp(shrink.r).numpy())

    E_bar_norm = float(tf.norm(shrink.E_bar).numpy())
    njt_norm = float(tf.norm(shrink.njt).numpy())

    gamma_mean = float(tf.reduce_mean(tf.cast(shrink.gamma, tf.float64)).numpy())
    phi_mean = float(tf.reduce_mean(shrink.phi).numpy())

    print(
        f"[LuShrinkage] it={it} | "
        f"beta_p={beta_p:.4f}, beta_w={beta_w:.4f}, sigma={sigma:.4f} | "
        f"E_bar_norm={E_bar_norm:.4e}, njt_norm={njt_norm:.4e} | "
        f"mean(gamma)={gamma_mean:.4f}, mean(phi)={phi_mean:.4f}"
    )

    return {
        "beta_p": beta_p,
        "beta_w": beta_w,
        "r": r_val,
        "E_bar_norm": E_bar_norm,
        "njt_norm": njt_norm,
        "gamma_mean": gamma_mean,
        "phi_mean": phi_mean,
    }
