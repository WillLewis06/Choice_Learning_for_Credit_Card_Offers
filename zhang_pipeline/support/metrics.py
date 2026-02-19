import tensorflow as tf


def rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the RMSE.

    y_true, y_pred: (batch, J)
    returns: scalar tensor
    """

    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred)))


def rmse_from_sse(total_sq_err: float, total_elems: int) -> float:
    """
    Computes the Root mean squared error from aggregated squared error.

    total_sq_err: sum of squared differences over all elements
    total_elems:  num of elements (N * J)
    returns: RMSE as a float.
    """

    if total_elems <= 0:
        raise ValueError("In computing RMSE from SSE total_elems must be positive.")

    mse = total_sq_err / float(total_elems)
    return float(mse**0.5)
