"""
End-to-end pytest tests for featureless.py.

We check that:
- the base Keras model produces masked logits with the expected shape
- invalid input shapes are rejected early
- unavailable items are masked to a large negative logit
- the wrapper requires available_items_by_choice
- get_config is usable for reconstruction
- the base model can be saved and loaded
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

# If this file lives in choice_learn/models/, keep the import relative.
from .featureless import BaseFeaturelessDeepHalo, FeaturelessDeepHalo


def _make_base(
    num_items: int = 4, depth: int = 2, width: int = 8
) -> BaseFeaturelessDeepHalo:
    """
    Create a small BaseFeaturelessDeepHalo model for tests.
    """
    return BaseFeaturelessDeepHalo(num_items=num_items, depth=depth, width=width)


def _make_wrapper(
    num_items: int = 4, depth: int = 2, width: int = 8
) -> FeaturelessDeepHalo:
    """
    Create a small FeaturelessDeepHalo wrapper for tests.
    """
    return FeaturelessDeepHalo(num_items=num_items, depth=depth, width=width)


def test_base_featureless_shapes_and_mask() -> None:
    """
    Check BaseFeaturelessDeepHalo runs and masks unavailable items.

    Expected behaviour:
    - output shape is (batch_size, num_items)
    - items with availability <= 0.5 have logit = -1e9
    - offered items have finite logits
    """
    base = _make_base(num_items=4, depth=2, width=8)

    x = tf.constant(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=tf.float32,
    )

    logits = base(x, training=False).numpy()
    assert logits.shape == (2, 4)

    neg_large = -1e9

    # Row 0: items 2 and 4 unavailable
    assert logits[0, 1] == neg_large
    assert logits[0, 3] == neg_large
    assert np.isfinite(logits[0, 0])
    assert np.isfinite(logits[0, 2])

    # Row 1: items 3 and 4 unavailable
    assert logits[1, 2] == neg_large
    assert logits[1, 3] == neg_large
    assert np.isfinite(logits[1, 0])
    assert np.isfinite(logits[1, 1])


def test_base_featureless_accepts_integer_inputs() -> None:
    """
    Check BaseFeaturelessDeepHalo accepts integer availability input and runs.
    """
    base = _make_base(num_items=4, depth=2, width=8)

    x = tf.constant([[1, 0, 1, 0]], dtype=tf.int32)
    logits = base(x, training=False)

    assert logits.shape == (1, 4)
    assert tf.reduce_all(tf.math.is_finite(logits))


def test_base_featureless_raises_on_bad_rank() -> None:
    """
    Check BaseFeaturelessDeepHalo fails fast when input rank is not 2.
    """
    base = _make_base(num_items=4, depth=2, width=8)

    bad = tf.zeros((2, 4, 1), dtype=tf.float32)
    with pytest.raises(ValueError):
        _ = base(bad, training=False)


def test_base_featureless_raises_on_bad_num_items() -> None:
    """
    Check BaseFeaturelessDeepHalo fails fast when input last dimension != num_items.
    """
    base = _make_base(num_items=4, depth=2, width=8)

    bad = tf.zeros((2, 5), dtype=tf.float32)
    with pytest.raises(ValueError):
        _ = base(bad, training=False)


def test_base_featureless_get_config_round_trip() -> None:
    """
    Check BaseFeaturelessDeepHalo get_config supports reconstruction.
    """
    base = _make_base(num_items=4, depth=3, width=16)

    cfg = base.get_config()
    assert cfg["num_items"] == 4
    assert cfg["depth"] == 3
    assert cfg["width"] == 16

    rebuilt = BaseFeaturelessDeepHalo.from_config(cfg)
    assert rebuilt.num_items == 4
    assert rebuilt.depth == 3
    assert rebuilt.width == 16


def test_base_featureless_gradients_are_finite() -> None:
    """
    Check BaseFeaturelessDeepHalo produces finite gradients under a simple loss.
    """
    base = _make_base(num_items=4, depth=2, width=8)

    x = tf.constant(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        logits = base(x, training=True)
        # Only sum offered logits to avoid huge -1e9 dominating the loss.
        mask = tf.cast(x > 0.5, logits.dtype)
        loss = tf.reduce_sum(logits * mask)

    grads = tape.gradient(loss, base.trainable_variables)
    assert grads is not None
    assert all(g is not None for g in grads)

    for g in grads:
        g_np = g.numpy()
        assert np.isfinite(g_np).all()


def test_featureless_wrapper_requires_availability() -> None:
    """
    Check FeaturelessDeepHalo fails fast when available_items_by_choice is missing.
    """
    wrapper = _make_wrapper(num_items=4, depth=2, width=8)
    choices = tf.constant([0], dtype=tf.int32)

    with pytest.raises(ValueError):
        _ = wrapper.compute_batch_utility(
            shared_features_by_choice=None,
            items_features_by_choice=None,
            available_items_by_choice=None,
            choices=choices,
        )


def test_featureless_wrapper_masks_unavailable() -> None:
    """
    Check FeaturelessDeepHalo wrapper returns masked logits for unavailable items.
    """
    wrapper = _make_wrapper(num_items=4, depth=2, width=8)
    choices = tf.constant([0], dtype=tf.int32)

    avail = tf.constant([[1.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
    logits = wrapper.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=None,
        available_items_by_choice=avail,
        choices=choices,
    ).numpy()[0]

    assert logits.shape == (4,)
    assert logits[1] == -1e9
    assert logits[3] == -1e9
    assert np.isfinite(logits[0])
    assert np.isfinite(logits[2])


def test_base_featureless_can_be_saved_and_loaded(tmp_path: Path) -> None:
    """
    Check BaseFeaturelessDeepHalo can be saved and loaded as a .keras model.
    """
    base = _make_base(num_items=4, depth=2, width=8)

    # Build weights
    x = tf.constant([[1.0, 0.0, 1.0, 0.0]], dtype=tf.float32)
    _ = base(x, training=False)

    save_path = tmp_path / "base_featureless.keras"
    base.save(save_path)

    loaded = tf.keras.models.load_model(save_path)

    logits = loaded(x, training=False).numpy()[0]
    assert logits.shape == (4,)
    assert logits[1] == -1e9
    assert logits[3] == -1e9
    assert np.isfinite(logits[0])
    assert np.isfinite(logits[2])
