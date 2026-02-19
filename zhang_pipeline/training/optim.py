import tensorflow as tf
from typing import Dict


def get_optimizer(cfg: Dict) -> tf.keras.optimizers.Optimizer:
    """
    Build optimizer from config.

    cfg: config
    returns: tf.keras optimizer
    """

    name = cfg["train"]["optimizer"]
    lr = cfg["train"]["lr"]

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)

    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)

    raise ValueError(f"Unsupported optimizer: {name!r}. Use 'adam' or 'sgd'.")
