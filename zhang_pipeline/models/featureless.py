import tensorflow as tf
from models.blocks import QuadraticResidualBlock


def build_featureless_from_config(cfg: dict, num_items: int) -> tf.keras.Model:
    """
    Builds a FeaturelessDeepHalo model using cfg['model'] params.
    """
    model_cfg = cfg.get("model", {})
    depth = int(model_cfg.get("depth", 3))
    width = int(model_cfg.get("width", 64))
    return FeaturelessDeepHalo(num_items=num_items, depth=depth, width=width)


@tf.keras.utils.register_keras_serializable(package="zhang25")
class FeaturelessDeepHalo(tf.keras.Model):
    """
    Featureless DeepHalo model made up of:

    1 input linear layer - embedding function (maps J -> J')
    (depth - 1) * QuadraticResidualBlock's from the class above (maps J' -> J')
    1 output linear layer - maps from y_L to logits U (J' -> J)
    """

    def __init__(self, num_items: int, depth: int, width: int, **kwargs) -> None:
        """
        Inputs to the model have shape (batch_size, num_items)

        num_items: num of choices J
        depth: number of layers (as per the paper's meaning of layers l)
        width: J' in the paper, the shape of W is J'*J'
        """

        # Initialize base Keras model
        super().__init__()

        # Store config for validation and debugging
        self.num_items = num_items
        self.depth = depth
        self.width = width

        # Input embedding function - (maps J -> J')
        self.in_linear = tf.keras.layers.Dense(
            units=width,
            use_bias=False,
        )

        # Residual stack: (depth - 1) quadratic residual blocks
        # maps from a layer's deep representations to another
        self.blocks = [QuadraticResidualBlock(width) for _ in range(depth - 1)]

        # Output projection - maps R^width to R^num_items
        self.out_linear = tf.keras.layers.Dense(
            units=num_items,
            use_bias=False,
        )

    def get_config(self) -> dict:
        """
        Returns the class' config
        """
        config = super().get_config()
        config.update(
            {
                "num_items": self.num_items,
                "depth": self.depth,
                "width": self.width,
            }
        )
        return config

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """
        Forward pass of the featureless DeepHalo model.

        x: input tensor of shape (batch_size, num_items)
        training: unused, required for Keras API compatibility
        returns: probs - choice probabilities - shape (batch_size, num_items)
        """

        # Validate shape (batch_size, num_items)
        if x.shape.rank != 2 or (
            x.shape[-1] is not None and x.shape[-1] != self.num_items
        ):
            raise ValueError(
                f"x must have shape (batch_size, {self.num_items}), got {x.shape}"
            )

        # Ensure floating dtype
        if not x.dtype.is_floating:
            x = tf.cast(x, tf.float32)

        # bool mask of offered items:
        # entry is 1 -> item is on offer - entry is 0 -> item not on offer
        mask = x > 0.5

        # Check that each row has at least one offered item
        row_has_offer = tf.reduce_any(mask, axis=1)

        # Input projection
        h = self.in_linear(x)

        # Residual stack
        for block in self.blocks:
            h = block(h)

        # Output projection
        logits = self.out_linear(h)

        # enforce logits of items not on offer to a strong negative value
        # probs for that item will be zero after softmax evaluation
        neg_large = tf.constant(-1e9, dtype=logits.dtype)
        masked_logits = tf.where(mask, logits, neg_large)

        # Softmax probabilities
        probs = tf.nn.softmax(masked_logits, axis=-1)

        # for rows of empty available items -> set probs to zero
        probs = tf.where(
            row_has_offer[:, None],  # True: keep real probabilities
            probs,  # False: override
            tf.zeros_like(probs),  # rows with no offered items → all zeros
        )

        return probs
