import tensorflow as tf


def get_loss_fn(loss_name: str):
    """
    Maps config loss_name to tf loss function.
    Returns a tf function
    """
    if loss_name == "mse":
        return mse_loss
    elif loss_name == "nll":
        return nll_loss
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}")


def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Mean squared error function.

    y_true, y_pred: (batch, J)
    returns: scalar loss
    """
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))


def nll_loss(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-9) -> tf.Tensor:
    """
    Negative log-likelihood function.

    y_true: one-hot (batch, J)
    y_pred: probabilities (batch, J)
    """
    # Indices where y_true == 1
    idx = tf.where(y_true > 0.5)

    # Gather predicted probabilities for chosen items
    probs = tf.gather_nd(y_pred, idx)

    # Compute mean negative log probability
    return -tf.reduce_mean(tf.math.log(probs + eps))
