import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="zhang25")
class QuadraticResidualBlock(tf.keras.layers.Layer):
    """
    Quadratic residual block implementing:
        y_out = y_in + W * y_in^2
    """

    def __init__(self, width: int) -> None:
        """
        width: J' in the paper, the shape of W is J'*J'
        """

        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive int, got {width!r}")

        super().__init__()

        self.width = width
        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={-1: width})

        # This Dense layer represents the matrix W in the formula.
        self.linear = tf.keras.layers.Dense(units=width, use_bias=False)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"width": self.width})
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the quadratic residual block.

        x: input tensor of shape (batch_size, width).
        y: output tensor of shape (batch_size, width) y = x + W * x^2
        """

        # Ensure floating dtype
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)

        # Residual connection: y = W x^2 + x
        y = x + self.linear(tf.math.square(x))

        return y


@tf.keras.utils.register_keras_serializable(package="zhang25")
class Phi_func(tf.keras.layers.Layer):
    """
    Phi function implementation of 2 layer MLP.

    Phi_l(z) = LayerNorm(Ws_l * Relu (Wh_l * z0 + bh_l) + bs_l)
    We apply:
        fc1: Dense layer with bias - maps d -> d*H
        ReLU
        fc2: Dense layer with bias - maps d -> d (shared across heads)
        LayerNorm
    """

    def __init__(self, heads: int, width: int, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.width = width

        self.fc1 = tf.keras.layers.Dense(self.width * self.heads, use_bias=True)
        self.fc2 = tf.keras.layers.Dense(self.width, use_bias=True)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"heads": self.heads, "width": self.width})
        return cfg

    def call(self, x, training=None):
        # x: (B, J, d)
        B = tf.shape(x)[0]
        J = tf.shape(x)[1]
        x = self.fc1(x)  # (B, J, d*H)
        x = tf.reshape(x, [B, J, self.heads, self.width])  # (B, J, H, d)
        x = tf.nn.relu(x)
        x = self.fc2(x)  # (B, J, H, d)
        x = self.norm(x)
        return x
