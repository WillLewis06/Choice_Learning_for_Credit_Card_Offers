import os
import tensorflow as tf
from typing import Any

from models.featureless import FeaturelessDeepHalo
from models.featurebased import FeatureBasedDeepHalo
from models.stacked import Stacked


def make_model_name(cfg: dict) -> str:
    """
    Construct a descriptive base name for the saved model from config.

    Uses model.type, model.depth, model.width if present.
    """
    model = cfg["model"]

    # explicit save name
    save_name = model.get("save_name")
    if save_name:
        name = save_name if save_name.endswith(".keras") else save_name + ".keras"
        return name

    # validated fields: always present
    model_type = model["type"]
    depth = model["depth"]
    width = model["width"]

    return f"{model_type}_d{depth}_w{width}.keras"


def make_continue_model_name(cfg: dict[str, Any]) -> str:
    """
    Construct a file name for saving a model after continued training.

    If model.save_name is provided, return that
    Else use the base name with '_continue' suffix.
    """
    model = cfg["model"]

    # explicit save name given
    save_name = model.get("save_name")
    if save_name:
        return save_name if save_name.endswith(".keras") else save_name + ".keras"

    # Otherwise extend the base model name
    base = make_model_name(cfg)
    stem = base[:-6]  # remove .keras
    return f"{stem}_continue.keras"


def save_full_model(
    model: tf.keras.Model,
    checkpoint_dir: str,
    model_name: str,
) -> str:
    """
    Save the full Keras model in .keras format.

    returns: full path used for saving.
    """
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be a non-empty string.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, model_name)

    # Full-model save (not weights-only).
    model.save(ckpt_path)
    return ckpt_path


def load_full_model_from_config(cfg: dict[str, Any]) -> tf.keras.Model:
    """
    Load a full Keras model using the path in the config.

    returns: loaded Keras model (uncompiled).
    """

    model_path = cfg["model"]["load_path"]

    import models.featureless
    import models.blocks
    import models.featurebased
    import models.stacked

    model = tf.keras.models.load_model(model_path, compile=False)
    return model
