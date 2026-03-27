"""
Unit tests for lu.shrinkage.lu_gibbs.

Constraints
- No pytest fixture injection: all tests are plain functions with no fixture args.
- The public function under test is gibbs_gamma(...).
- Tests focus on the refactored contract: binary indicator support,
  stateless reproducibility, and the sequential collapsed-Gibbs logic.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from lu.lu_gibbs import gibbs_gamma

DTYPE = tf.float64
SEED_DTYPE = tf.int32


def _tf(x) -> tf.Tensor:
    """Create a tf.float64 constant."""
    return tf.constant(x, dtype=DTYPE)


def _seed(a: int, b: int) -> tf.Tensor:
    """Create a stateless seed tensor."""
    return tf.constant([a, b], dtype=SEED_DTYPE)


def _assert_binary_01_tf(x: tf.Tensor) -> None:
    """Assert that a tensor is finite and takes values only in {0, 1}."""
    x_np = tf.convert_to_tensor(x).numpy()
    if not np.all(np.isfinite(x_np)):
        raise AssertionError("Tensor contains non-finite values.")
    if not np.all((x_np == 0.0) | (x_np == 1.0)):
        raise AssertionError("Tensor contains values outside {0, 1}.")


def _tiny_inputs() -> dict:
    """Canonical small valid Gibbs input used across several tests."""
    return {
        "njt": _tf([[0.0, 0.2, -0.1], [0.05, -0.02, 0.0]]),
        "gamma": _tf([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        "a_phi": _tf(1.5),
        "b_phi": _tf(2.0),
        "T0_sq": _tf(0.25),
        "T1_sq": _tf(1.0),
        "seed": _seed(123, 456),
    }


def _conditional_prob1(
    njt_j: tf.Tensor,
    s_minus_j: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    J: int,
) -> tf.Tensor:
    """Compute P(gamma_j = 1 | rest) from the collapsed Gibbs update."""
    J_tf = tf.cast(J, DTYPE)

    logp0 = (
        tf.math.log(b_phi + (J_tf - 1.0 - s_minus_j))
        - 0.5 * tf.square(njt_j) / T0_sq
        - 0.5 * tf.math.log(T0_sq)
    )
    logp1 = (
        tf.math.log(a_phi + s_minus_j)
        - 0.5 * tf.square(njt_j) / T1_sq
        - 0.5 * tf.math.log(T1_sq)
    )
    return tf.math.sigmoid(logp1 - logp0)


def _manual_gibbs_sweep(
    njt: tf.Tensor,
    gamma: tf.Tensor,
    a_phi: tf.Tensor,
    b_phi: tf.Tensor,
    T0_sq: tf.Tensor,
    T1_sq: tf.Tensor,
    seed: tf.Tensor,
    sequential: bool,
) -> tuple[tf.Tensor, list[tf.Tensor]]:
    """
    Test-side implementation of the Gibbs sweep.

    sequential=True mirrors the algebra in gibbs_gamma(...).
    sequential=False computes a parallel-style counterfactual in which each
    column uses the original state counts rather than updated counts.

    Note: this helper runs eagerly. It is suitable for checking the update
    algebra and conditional probabilities, but exact draw-by-draw equality with
    the compiled XLA path should only be asserted when the conditional
    probabilities are numerically saturated near 0 or 1.
    """
    njt = tf.convert_to_tensor(njt, dtype=DTYPE)
    gamma = tf.convert_to_tensor(gamma, dtype=DTYPE)
    a_phi = tf.convert_to_tensor(a_phi, dtype=DTYPE)
    b_phi = tf.convert_to_tensor(b_phi, dtype=DTYPE)
    T0_sq = tf.convert_to_tensor(T0_sq, dtype=DTYPE)
    T1_sq = tf.convert_to_tensor(T1_sq, dtype=DTYPE)
    seed = tf.convert_to_tensor(seed, dtype=SEED_DTYPE)

    J_int = int(gamma.shape[-1])
    J = tf.cast(J_int, DTYPE)

    log_T0_sq = tf.math.log(T0_sq)
    log_T1_sq = tf.math.log(T1_sq)

    gamma_curr = tf.identity(gamma)
    s_init = tf.reduce_sum(gamma, axis=-1)
    s_curr = tf.identity(s_init)
    seed_curr = tf.identity(seed)

    prob_history: list[tf.Tensor] = []

    for j in range(J_int):
        if sequential:
            gamma_ref = gamma_curr
            s_ref = s_curr
        else:
            gamma_ref = gamma
            s_ref = s_init

        gamma_j = gamma_ref[:, j]
        njt_j = njt[:, j]
        s_minus_j = s_ref - gamma_j

        logp0 = (
            tf.math.log(b_phi + (J - 1.0 - s_minus_j))
            - 0.5 * tf.square(njt_j) / T0_sq
            - 0.5 * log_T0_sq
        )
        logp1 = (
            tf.math.log(a_phi + s_minus_j)
            - 0.5 * tf.square(njt_j) / T1_sq
            - 0.5 * log_T1_sq
        )
        prob1 = tf.math.sigmoid(logp1 - logp0)
        prob_history.append(prob1)

        seeds = tf.random.experimental.stateless_split(seed_curr, num=2)
        seed_curr = seeds[0]
        draw_seed = seeds[1]

        u = tf.random.stateless_uniform(
            shape=tf.shape(prob1),
            seed=draw_seed,
            dtype=DTYPE,
        )
        new_gamma_j = tf.cast(u < prob1, DTYPE)

        one_hot_j = tf.one_hot(j, depth=J_int, dtype=DTYPE)
        gamma_curr = (
            gamma_curr * (1.0 - one_hot_j[None, :])
            + new_gamma_j[:, None] * one_hot_j[None, :]
        )

        if sequential:
            s_curr = s_minus_j + new_gamma_j

    return gamma_curr, prob_history


def test_gibbs_gamma_preserves_shape_dtype_and_binary_support():
    tiny = _tiny_inputs()

    gamma_out = gibbs_gamma(
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        a_phi=tiny["a_phi"],
        b_phi=tiny["b_phi"],
        T0_sq=tiny["T0_sq"],
        T1_sq=tiny["T1_sq"],
        seed=tiny["seed"],
    )

    assert tuple(gamma_out.shape) == tuple(tiny["gamma"].shape)
    assert gamma_out.dtype == tiny["gamma"].dtype
    _assert_binary_01_tf(gamma_out)


def test_gibbs_gamma_is_deterministic_for_fixed_seed():
    tiny = _tiny_inputs()

    gamma_out_1 = gibbs_gamma(
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        a_phi=tiny["a_phi"],
        b_phi=tiny["b_phi"],
        T0_sq=tiny["T0_sq"],
        T1_sq=tiny["T1_sq"],
        seed=tiny["seed"],
    )
    gamma_out_2 = gibbs_gamma(
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        a_phi=tiny["a_phi"],
        b_phi=tiny["b_phi"],
        T0_sq=tiny["T0_sq"],
        T1_sq=tiny["T1_sq"],
        seed=tiny["seed"],
    )

    assert np.array_equal(gamma_out_1.numpy(), gamma_out_2.numpy())


def test_gibbs_gamma_changes_for_different_seeds():
    """
    Different seeds should generally produce different sampled indicators.

    The test uses a moderately large panel with non-extreme probabilities to
    make accidental equality across all entries negligibly unlikely.
    """
    njt = _tf(
        [
            [0.00, 0.10, -0.10, 0.05, -0.05, 0.00],
            [0.05, -0.05, 0.10, -0.10, 0.00, 0.05],
            [-0.10, 0.00, 0.05, -0.05, 0.10, -0.10],
            [0.10, -0.10, 0.00, 0.05, -0.05, 0.10],
            [0.00, 0.05, -0.05, 0.10, -0.10, 0.00],
            [-0.05, 0.10, 0.00, -0.10, 0.05, -0.05],
            [0.10, 0.00, -0.10, 0.05, 0.00, -0.05],
            [0.00, -0.05, 0.10, 0.00, -0.10, 0.05],
        ]
    )
    gamma = _tf(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )

    out_1 = gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=_tf(2.0),
        b_phi=_tf(2.0),
        T0_sq=_tf(0.50),
        T1_sq=_tf(1.00),
        seed=_seed(11, 17),
    )
    out_2 = gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=_tf(2.0),
        b_phi=_tf(2.0),
        T0_sq=_tf(0.50),
        T1_sq=_tf(1.00),
        seed=_seed(19, 23),
    )

    assert not np.array_equal(out_1.numpy(), out_2.numpy())


def test_gibbs_gamma_matches_manual_sequential_sweep_when_probabilities_saturate():
    """
    Exact equality between the eager manual sweep and the compiled Gibbs sweep is
    only safe to assert when the conditional probabilities are numerically
    saturated near 0 or 1.

    This removes dependence on any eager-versus-XLA RNG differences and checks
    that the sequential update algebra matches.
    """
    njt = _tf([[50.0, 0.0, 50.0], [0.0, 50.0, 0.0]])
    gamma = _tf([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    a_phi = _tf(1e-300)
    b_phi = _tf(1e300)
    T0_sq = _tf(1e-6)
    T1_sq = _tf(1.0)
    seed = _seed(123, 456)

    gamma_expected, prob_seq = _manual_gibbs_sweep(
        njt=njt,
        gamma=gamma,
        a_phi=a_phi,
        b_phi=b_phi,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        seed=seed,
        sequential=True,
    )
    gamma_actual = gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=a_phi,
        b_phi=b_phi,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        seed=seed,
    )

    for prob in prob_seq:
        prob_np = prob.numpy()
        assert np.all((prob_np < 1e-12) | (prob_np > 1.0 - 1e-12))

    tf.debugging.assert_equal(gamma_actual, gamma_expected)


def test_gibbs_gamma_sequential_counts_change_later_column_probability():
    """
    Later-column conditional probabilities should depend on the updated active
    count summary from earlier columns.

    This is a deterministic algebra test: it compares the second-column
    conditional probability under two different leave-one-out counts rather than
    relying on a realized random draw from the first column.
    """
    a_phi = _tf(1.0)
    b_phi = _tf(1.0)
    T0_sq = _tf(0.10)
    T1_sq = _tf(1.00)
    J = 3

    njt_j = _tf([0.5])

    prob_after_count_drop = _conditional_prob1(
        njt_j=njt_j,
        s_minus_j=_tf([0.0]),
        a_phi=a_phi,
        b_phi=b_phi,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        J=J,
    )
    prob_without_count_drop = _conditional_prob1(
        njt_j=njt_j,
        s_minus_j=_tf([1.0]),
        a_phi=a_phi,
        b_phi=b_phi,
        T0_sq=T0_sq,
        T1_sq=T1_sq,
        J=J,
    )

    assert not np.allclose(
        prob_after_count_drop.numpy(),
        prob_without_count_drop.numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert prob_without_count_drop.numpy()[0] > prob_after_count_drop.numpy()[0]


def test_gibbs_gamma_handles_single_column_case():
    njt = _tf([[0.0], [0.5], [-0.2]])
    gamma = _tf([[1.0], [0.0], [1.0]])

    gamma_out = gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=_tf(1.0),
        b_phi=_tf(1.0),
        T0_sq=_tf(0.25),
        T1_sq=_tf(1.00),
        seed=_seed(31, 37),
    )

    assert tuple(gamma_out.shape) == (3, 1)
    assert gamma_out.dtype == DTYPE
    _assert_binary_01_tf(gamma_out)


def test_gibbs_gamma_handles_single_market_case():
    njt = _tf([[0.0, 0.2, -0.1, 0.05]])
    gamma = _tf([[1.0, 0.0, 1.0, 0.0]])

    gamma_out = gibbs_gamma(
        njt=njt,
        gamma=gamma,
        a_phi=_tf(1.5),
        b_phi=_tf(2.0),
        T0_sq=_tf(0.25),
        T1_sq=_tf(1.00),
        seed=_seed(41, 43),
    )

    assert tuple(gamma_out.shape) == (1, 4)
    assert gamma_out.dtype == DTYPE
    _assert_binary_01_tf(gamma_out)


def test_gibbs_gamma_does_not_modify_input_tensors():
    tiny = _tiny_inputs()

    gamma_before = tf.identity(tiny["gamma"])
    njt_before = tf.identity(tiny["njt"])

    _ = gibbs_gamma(
        njt=tiny["njt"],
        gamma=tiny["gamma"],
        a_phi=tiny["a_phi"],
        b_phi=tiny["b_phi"],
        T0_sq=tiny["T0_sq"],
        T1_sq=tiny["T1_sq"],
        seed=tiny["seed"],
    )

    tf.debugging.assert_equal(tiny["gamma"], gamma_before)
    tf.debugging.assert_equal(tiny["njt"], njt_before)
