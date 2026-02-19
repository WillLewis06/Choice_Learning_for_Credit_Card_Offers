import tensorflow as tf

from choice_learn.models.base_model import ChoiceModel
from .blocks import QuadraticResidualBlock


@tf.keras.utils.register_keras_serializable(package="zhang25")
class BaseFeaturelessDeepHalo(tf.keras.Model):
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
        super().__init__(**kwargs)

        # Store config for validation and debugging
        self.num_items = int(num_items)
        self.depth = int(depth)
        self.width = int(width)

        if depth <= 0:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if width <= 0:
            raise ValueError(f"width must be > 0, got {width}")

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
        returns: choice logits - shape (batch_size, num_items)
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

        return masked_logits


class FeaturelessDeepHalo(ChoiceModel):
    """
    Wrapper class of the Featureless DeepHalo model to integrate into choice-learn.

    Uses available_items_by_choice as the model input x
    compute_batch_utility returns logits (B, J)
    ChoiceModel handles masked softmax and NLL.
    """

    def __init__(
        self, num_items: int, depth: int = 3, width: int = 64, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.num_items = int(num_items)
        self.depth = int(depth)
        self.width = int(width)

        self.base_model = BaseFeaturelessDeepHalo(
            num_items=self.num_items,
            depth=self.depth,
            width=self.width,
        )

    @property
    def trainable_weights(self):
        """
        Returns weights - used by ChoiceModel.save_model
        """
        return list(self.base_model.trainable_weights)

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices,
    ) -> tf.Tensor:
        """
        Computes logits for a batch.

        available_items_by_choice gives us our featureless input
        Returns: logits (B, J)
        """
        if available_items_by_choice is None:
            raise ValueError(
                "FeaturelessDeepHalo requires available_items_by_choice (B, J) as input."
            )
        return self.base_model(available_items_by_choice)
