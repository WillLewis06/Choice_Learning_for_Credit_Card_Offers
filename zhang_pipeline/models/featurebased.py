import tensorflow as tf

from models.blocks import Phi_func


def build_featurebased_from_config(cfg: dict, num_items: int) -> tf.keras.Model:
    """
    Build a FeatureBasedDeepHalo model using cfg["model"] params.

    num_items: J
    """
    depth = cfg["model"]["depth"]
    width = cfg["model"]["width"]
    heads = cfg["model"]["heads"]

    return FeatureBasedDeepHalo(
        num_items=num_items,
        depth=depth,
        width=width,
        heads=heads,
    )


@tf.keras.utils.register_keras_serializable(package="zhang25")
class FeatureBasedDeepHalo(tf.keras.Model):
    """
    Minimal feature-based Zhang25 model:

    input: x - shape: (batch, J, dx)
    depth: L - layers l = 1 to L
    layer width: d
    num heads: H

    1) Base embedding:
       z_0 = X(x) - maps (J,dx) to (J,d)

    2) Each layer l:
       linear encoding: s_l = W_l * z_{l-1} - we use a dense layer - maps (J,d) to (J,H)
       item average signal: Zbar_l = (1/|S|) * sum s_l - maps (J,H) to H
       heads representation: h_l = phi_l( z_0 ) - we use dense + Relu - maps (J,d) to (J,d*H)
       layer representation:  z_l = z_{l-1} + (1/H) * sum Zbar_l * h_l - maps (J,d) to (J,d)

    3) Output probabilities:
       logits: u = Beta * z_L - maps (J,d) to J
       probs: p = softmax_j(u) - maps J to J
    """

    def __init__(self, num_items: int, depth: int, width: int, heads: int, **kwargs):
        super().__init__(**kwargs)

        self.num_items = num_items  # J
        self.depth = depth  # L
        self.width = width  # d
        self.heads = heads  # H

        # Enforce input shape with axis 1 = num_items
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={1: self.num_items})

        # base embedding layer - maps the input to width (d_x → d)

        # χ: 3-layer MLP + LayerNorm
        self.base_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.width, activation="relu", use_bias=True),
                tf.keras.layers.Dense(self.width, activation="relu", use_bias=True),
                tf.keras.layers.Dense(self.width, use_bias=True),
            ]
        )
        self.base_norm = tf.keras.layers.LayerNormalization(axis=-1)

        # each layer l stores:
        self.head_projs = []
        self.phi_layers = []
        for _ in range(self.depth):
            # linear encoding: s_l = W_l * z_{l-1} - we use a dense layer - maps (J,d) to (J,H)
            self.head_projs.append(tf.keras.layers.Dense(self.heads, use_bias=True))
            # heads representation: h_l = phi_l( z_0 ) - we use MLP - maps (J,d) to (J,d*H)
            self.phi_layers.append(Phi_func(self.heads, self.width))

        # output layer
        self.beta_layer = tf.keras.layers.Dense(1, use_bias=True)

    def get_config(self):
        """
        Returns the class' config
        """
        config = super().get_config()
        config.update(
            {
                "num_items": self.num_items,
                "depth": self.depth,
                "width": self.width,
                "heads": self.heads,
            }
        )
        return config

    def call(self, x, training=None):
        """
        x: Tensor with shape (batch_size, J, dx)

        returns: probabilities with shape (batch_size, J)
        """

        # check input is float32
        x = tf.convert_to_tensor(x)
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)

        batch_size = tf.shape(x)[0]

        # Mask padded (dummy) items.
        # Convention: padded items are all-zeros across feature dimension.
        # mask: (B, J) boolean - True for non-zeros - False for zeros
        mask_bool = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        # mask: (B, J, 1) float32 for broadcasting
        mask = tf.cast(mask_bool, tf.float32)[:, :, tf.newaxis]
        # valid_count: (B, 1) - num of valid items
        valid_count = tf.reduce_sum(mask, axis=1)
        # Avoid division by zero - set to 1 if null
        valid_count = tf.maximum(valid_count, 1.0)

        # 1) Base embedding:
        # z_0 = X(x) - maps (J,dx) to (J,d)
        z0 = self.base_norm(self.base_encoder(x, training=training))

        # Start recursion from z_prev = z^0
        z_prev = z0

        # 2) compute each layer l = 1 to L
        for head_proj, phi_layer in zip(self.head_projs, self.phi_layers):

            # linear encoding: s_l = W_l * z_{l-1} - we use a dense layer - maps (J,d) to (J,H)
            head_scores = head_proj(z_prev)

            # item average signal: Zbar_l = (1/|S|) * sum s_l - maps (J,H) to H
            Zbar = tf.reduce_sum(head_scores * mask, axis=1) / valid_count
            # reshape so that we can multiply by phi
            Zbar_exp = Zbar[:, tf.newaxis, :, tf.newaxis]

            # heads representation: h_l = phi_l( z_0 ) - we use dense + Relu - maps (J,d) to (J,d*H)
            phi = phi_layer(z0, training=training)
            # Mask φ for padded items (author code multiplies by valid mask each layer)
            phi = phi * mask[:, :, tf.newaxis]

            # layer representation:  z_l = z_{l-1} + (1/H) * sum Zbar_l * h_l - maps (J,d) to (J,d)
            z_prev = z_prev + tf.reduce_sum(Zbar_exp * phi, axis=2) / float(self.heads)

        # 3) Utilities and probabilities
        # logits shape: (B, J)
        logits = self.beta_layer(z_prev)
        logits = tf.squeeze(logits, axis=-1)

        # Force padded (dummy) items to have probability 0.
        neg_inf = tf.constant(-1e9, dtype=logits.dtype)
        logits = tf.where(mask_bool, logits, neg_inf)

        # Softmax over items j
        probs = tf.nn.softmax(logits, axis=-1)

        return probs
