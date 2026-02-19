import tensorflow as tf


def build_stacked_from_config(cfg: dict, num_items: int) -> tf.keras.Model:
    """
    Build a Stacked model using cfg["model"] params.

    num_items: J
    layer width equals num_items
    """

    depth = cfg["model"]["depth"]
    return Stacked(num_items, depth)


@tf.keras.utils.register_keras_serializable(package="zhang25")
class Stacked(tf.keras.Model):
    """
    Layer-wise model with logit representations at every layer

    layer width equals the num_items

    Base embedding:
        z_1 = W_1 * x
    Layer-wise recursion:
        z_{l+1} = z_l + W_{l+1} * (z_l * x)

    call(x) method returns a list of per-layer probabilities:
        [p_1, p_2, ..., p_L]
    p_l = softmax(z_l) and has shape (batch_size, num_items)
    """

    def __init__(self, num_items: int, depth: int) -> None:
        super().__init__()
        self.num_items = num_items
        self.depth = depth

        # Enforce the correct input shape
        self.input_spec = tf.keras.layers.InputSpec(ndim=2, axes={1: self.num_items})

        # First layer: z_1 = W_1 * x
        self.first_linear = tf.keras.layers.Dense(
            units=num_items,
            use_bias=False,
            name="stacked_first_linear",
        )

        # Residual layers:  z_{l+1} = z_l + W_{l+1} * (z_l * x)
        self.residual_linears = []
        for ell in range(1, depth):
            layer = tf.keras.layers.Dense(units=num_items, use_bias=False)
            self.residual_linears.append(layer)

    def get_config(self) -> dict:
        """
        Returns the class' config.
        """
        return {
            "num_items": self.num_items,
            "depth": self.depth,
        }

    def call(self, x: tf.Tensor, training: bool | None = None):
        """
        Forward pass.

        x: tensor of shape (batch_size, num_items)

        Returns a list of per-layer probabilities:
            [p_1, p_2, ..., p_L]
        """

        # Ensure type is float32
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)

        probs_list = []

        # z_1 = W_1 * x
        z = self.first_linear(x)
        p = tf.nn.softmax(z, axis=-1)
        probs_list.append(p)

        # Residual recursion: z_{l+1} = z_l + W_{l+1} * (z_l * x)
        for layer in self.residual_linears:
            z_times_x = z * x  # elementwise
            increment = layer(z_times_x)
            z = z + increment
            p = tf.nn.softmax(z, axis=-1)
            probs_list.append(p)

        return probs_list
