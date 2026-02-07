import tensorflow as tf

# -----------------------------------------------------------------------------
# TMH validation (TF-side) — callable inside @tf.function
# -----------------------------------------------------------------------------


@tf.function(reduce_retracing=True)
def tmh_step_validate_input_tf(
    theta0: tf.Tensor,
    k: tf.Tensor,
    ridge: tf.Tensor,
) -> None:
    """Validate TMH inputs inside a compiled TF graph.

    Enforced:
      - theta0: rank-1 tf.float64 tensor with finite entries
      - k: scalar tf.float64, finite, and k > 0
      - ridge: scalar tf.float64, finite, and ridge >= 0

    The TMH step relies on gradients/Hessians and linear algebra; these checks
    catch invalid inputs early and provide clear error messages.
    """
    tf.debugging.assert_type(theta0, tf.float64)
    tf.debugging.assert_rank(theta0, 1)
    tf.debugging.assert_all_finite(theta0, "theta0 contains non-finite values.")

    tf.debugging.assert_type(k, tf.float64)
    tf.debugging.assert_rank(k, 0)
    tf.debugging.assert_all_finite(k, "k is non-finite.")
    tf.debugging.assert_greater(k, tf.constant(0.0, tf.float64), "k must be > 0.")

    tf.debugging.assert_type(ridge, tf.float64)
    tf.debugging.assert_rank(ridge, 0)
    tf.debugging.assert_all_finite(ridge, "ridge is non-finite.")
    tf.debugging.assert_greater_equal(
        ridge, tf.constant(0.0, tf.float64), "ridge must be >= 0."
    )
