import tensorflow as tf
from typing import Callable

from support.metrics import rmse_from_sse


def train_one_epoch(
    model: tf.keras.Model,
    model_type: str,
    train_ds: tf.data.Dataset,
    loss_fn: Callable,
    optimizer: tf.keras.optimizers.Optimizer,
    l2_reg: float,
) -> float:
    """
    Train the model for one epoch.

    model: Keras model
    train_ds: tf.data.Dataset (X_batch, Y_batch)
    loss_fn: callable loss function
    optimizer: tf.keras optimizer
    l2_reg: L2 regularisation param

    returns: average loss over the epoch (float)
    """
    total_loss = 0.0
    num_batches = 0

    for X_batch, Y_batch in train_ds:
        with tf.GradientTape() as tape:
            Y_pred = model(X_batch, training=True)

            if model_type == "stacked":
                # stacked model outputs a prediction p for every layer
                # we sum the losses for each layer
                loss = tf.add_n([loss_fn(Y_batch, p) for p in Y_pred])
            else:
                loss = loss_fn(Y_batch, Y_pred)

            # regularisation
            if l2_reg > 0.0:
                # sum of squares
                reg_sum = tf.add_n(
                    [tf.reduce_sum(tf.square(v)) for v in model.trainable_variables]
                )
                # number of scalar weights
                num_weights = tf.add_n(
                    [tf.cast(tf.size(v), tf.float32) for v in model.trainable_variables]
                )
                reg_loss = reg_sum / num_weights  # mean(w^2)
                loss = loss + l2_reg * reg_loss

        grads = tape.gradient(loss, model.trainable_variables)

        # check no None gradients
        filtered = [
            (g, v) for g, v in zip(grads, model.trainable_variables) if g is not None
        ]
        optimizer.apply_gradients(filtered)

        total_loss += float(loss)
        num_batches += 1

    if num_batches == 0:
        raise ValueError("train_one_epoch received an empty train_ds.")

    return total_loss / num_batches


def evaluate_raw(model, model_type: str, X, Y, loss_fn, eval_batch_size: int) -> float:
    """
    Evaluate model on full tensors in batches.

    model: Keras model
    X: features tensor, shape (N, J)
    Y: targets tensor, shape (N, J)
    loss_fn: loss function
    eval_batch_size: batch size for evaluation

    returns avg loss
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    N = int(X.shape[0])
    if N == 0:
        raise ValueError("Empty evaluation set.")

    total_loss = 0.0
    total_obs = 0

    for start in range(0, N, eval_batch_size):
        end = min(start + eval_batch_size, N)

        X_b = X[start:end]
        Y_b = Y[start:end]

        Y_pred = model(X_b, training=False)

        if model_type == "stacked":
            # for logging purposes we just record the last layer output
            Y_pred_log_metric = Y_pred[-1]
        else:
            Y_pred_log_metric = Y_pred

        # loss_fn returns mean over batch; weight by batch size
        batch_loss = loss_fn(Y_b, Y_pred_log_metric)
        bsz = int(Y_b.shape[0])
        total_loss += float(batch_loss) * bsz
        total_obs += bsz

    avg_loss = total_loss / float(total_obs)

    return avg_loss


def evaluate_freq(model, model_type: str, X, Y_freq, eval_batch_size: int) -> float:
    """
    Evaluate model only on frequency-smoothed targets.

    model: Keras model
    X: (N, J) tensor of offer sets
    Y_freq: (N, J) frequency-smoothed targets
    eval_batch_size: batch size for evaluation

    returns freq based loss value
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_freq = tf.convert_to_tensor(Y_freq, dtype=tf.float32)

    N = int(X.shape[0])
    if N == 0:
        raise ValueError("Empty evaluation set.")

    total_sq_err_freq = 0.0
    total_elems = 0

    for start in range(0, N, eval_batch_size):
        end = min(start + eval_batch_size, N)

        X_b = X[start:end]
        Y_freq_b = Y_freq[start:end]

        Y_pred = model(X_b, training=False)

        if model_type == "stacked":
            # for logging purposes we just record the last layer output
            Y_pred_log_metric = Y_pred[-1]
        else:
            Y_pred_log_metric = Y_pred

        sq_err_freq = tf.reduce_sum(
            tf.math.squared_difference(Y_freq_b, Y_pred_log_metric)
        )
        total_sq_err_freq += float(sq_err_freq)
        total_elems += int(tf.size(Y_freq_b))

    freq_rmse = rmse_from_sse(total_sq_err_freq, total_elems)

    return freq_rmse
