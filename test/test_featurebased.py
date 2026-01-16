"""
End-to-end pytest tests for featurebased.py.

We check that:
- the base Keras model produces masked logits with the expected shape
- invalid input shapes are rejected early
- padded (all-zero) items are masked to a large negative logit
- the wrapper requires items_features_by_choice
- the wrapper concatenates tuple/list feature blocks correctly
- the wrapper masks unavailable items when availability is provided
- get_config is usable for reconstruction
- the base model can be saved and loaded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# If this file lives in choice_learn/models/, keep the import relative.
from .featurebased import BaseFeatureBasedDeepHalo, FeatureBasedDeepHalo


def _make_base(
    num_items: int = 4, depth: int = 2, width: int = 8, heads: int = 2
) -> BaseFeatureBasedDeepHalo:
    """
    Create a small BaseFeatureBasedDeepHalo model for tests.
    """
    return BaseFeatureBasedDeepHalo(
        num_items=num_items,
        depth=depth,
        width=width,
        heads=heads,
    )


def _make_wrapper(
    num_items: int = 4, depth: int = 2, width: int = 8, heads: int = 2
) -> FeatureBasedDeepHalo:
    """
    Create a small FeatureBasedDeepHalo wrapper for tests.
    """
    return FeatureBasedDeepHalo(
        num_items=num_items,
        depth=depth,
        width=width,
        heads=heads,
    )


def test_base_featurebased_shapes_and_padding_mask() -> None:
    """
    Check BaseFeatureBasedDeepHalo runs and masks padded items.

    Expected behaviour:
    - output shape is (batch_size, num_items)
    - items with all-zero features have logit = -1e9
    - non-padded items have finite logits
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    # batch=2, J=4, dx=3
    # Row 0: item 2 and 4 are padded (all zeros)
    # Row 1: item 3 is padded (all zeros)
    x = tf.constant(
        [
            [
                [1.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.2, 0.1, 0.0],
                [0.4, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
            ],
        ],
        dtype=tf.float32,
    )

    logits = base(x, training=False).numpy()
    assert logits.shape == (2, 4)

    neg_large = -1e9

    assert logits[0, 1] == neg_large
    assert logits[0, 3] == neg_large
    assert np.isfinite(logits[0, 0])
    assert np.isfinite(logits[0, 2])

    assert logits[1, 2] == neg_large
    assert np.isfinite(logits[1, 0])
    assert np.isfinite(logits[1, 1])
    assert np.isfinite(logits[1, 3])


def test_base_featurebased_accepts_integer_inputs() -> None:
    """
    Check BaseFeatureBasedDeepHalo accepts integer inputs and runs.
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    x = tf.constant(
        [
            [
                [1, 0, 0],
                [0, 0, 0],  # padded
                [1, 1, 0],
                [0, 0, 0],  # padded
            ]
        ],
        dtype=tf.int32,
    )
    logits = base(x, training=False)

    assert logits.shape == (1, 4)
    assert tf.reduce_all(tf.math.is_finite(tf.where(logits > -1e8, logits, 0.0)))


def test_base_featurebased_raises_on_bad_rank() -> None:
    """
    Check BaseFeatureBasedDeepHalo fails fast when input rank is not 3.
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    bad = tf.zeros((2, 4), dtype=tf.float32)  # rank 2
    with pytest.raises(Exception):
        _ = base(bad, training=False)


def test_base_featurebased_raises_on_bad_num_items() -> None:
    """
    Check BaseFeatureBasedDeepHalo fails fast when input axis 1 != num_items.
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    bad = tf.zeros((2, 5, 3), dtype=tf.float32)  # J=5 but model expects J=4
    with pytest.raises(Exception):
        _ = base(bad, training=False)


def test_base_featurebased_get_config_round_trip() -> None:
    """
    Check BaseFeatureBasedDeepHalo get_config supports reconstruction.
    """
    base = _make_base(num_items=4, depth=3, width=16, heads=4)

    cfg = base.get_config()
    assert cfg["num_items"] == 4
    assert cfg["depth"] == 3
    assert cfg["width"] == 16
    assert cfg["heads"] == 4

    rebuilt = BaseFeatureBasedDeepHalo.from_config(cfg)
    assert rebuilt.num_items == 4
    assert rebuilt.depth == 3
    assert rebuilt.width == 16
    assert rebuilt.heads == 4


def test_base_featurebased_gradients_are_finite() -> None:
    """
    Check BaseFeatureBasedDeepHalo produces finite gradients under a simple loss.
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    x = tf.constant(
        [
            [
                [1.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],  # padded
                [0.3, 0.1, 0.0],
                [0.0, 0.0, 0.0],  # padded
            ],
            [
                [0.2, 0.1, 0.0],
                [0.4, 0.0, 0.2],
                [0.0, 0.0, 0.0],  # padded
                [0.1, 0.1, 0.1],
            ],
        ],
        dtype=tf.float32,
    )

    # valid mask: item has any non-zero feature
    valid = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
    valid_f = tf.cast(valid, tf.float32)

    with tf.GradientTape() as tape:
        logits = base(x, training=True)
        # Only include valid logits to avoid -1e9 dominating.
        loss = tf.reduce_sum(logits * valid_f)

    grads = tape.gradient(loss, base.trainable_variables)
    assert grads is not None
    assert all(g is not None for g in grads)

    for g in grads:
        g_np = g.numpy()
        assert np.isfinite(g_np).all()


def test_featurebased_wrapper_requires_items_features() -> None:
    """
    Check FeatureBasedDeepHalo fails fast when items_features_by_choice is missing.
    """
    wrapper = _make_wrapper(num_items=4, depth=2, width=8, heads=2)
    choices = tf.constant([0], dtype=tf.int32)

    with pytest.raises(ValueError):
        _ = wrapper.compute_batch_utility(
            shared_features_by_choice=None,
            items_features_by_choice=None,
            available_items_by_choice=None,
            choices=choices,
        )


def test_featurebased_wrapper_concats_tuple_inputs() -> None:
    """
    Check FeatureBasedDeepHalo concatenates tuple/list feature blocks on the last axis.
    """
    wrapper = _make_wrapper(num_items=4, depth=2, width=8, heads=2)
    choices = tf.constant([0], dtype=tf.int32)

    x1 = tf.ones((1, 4, 2), dtype=tf.float32)
    x2 = tf.zeros((1, 4, 3), dtype=tf.float32)
    x2 = tf.tensor_scatter_nd_update(
        x2, indices=[[0, 1, 0]], updates=[1.0]
    )  # make one entry non-zero

    logits = wrapper.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=(x1, x2),
        available_items_by_choice=None,
        choices=choices,
    ).numpy()[0]

    assert logits.shape == (4,)
    assert np.isfinite(logits).all()


def test_featurebased_wrapper_masks_unavailable_items() -> None:
    """
    Check FeatureBasedDeepHalo wrapper masks logits when availability is provided.
    """
    wrapper = _make_wrapper(num_items=4, depth=2, width=8, heads=2)
    choices = tf.constant([0], dtype=tf.int32)

    x = tf.constant(
        [
            [
                [1.0, 0.0, 0.2],
                [0.2, 0.0, 0.1],
                [0.3, 0.1, 0.0],
                [0.1, 0.0, 0.0],
            ]
        ],
        dtype=tf.float32,
    )
    avail = tf.constant([[1.0, 0.0, 1.0, 0.0]], dtype=tf.float32)

    logits = wrapper.compute_batch_utility(
        shared_features_by_choice=None,
        items_features_by_choice=x,
        available_items_by_choice=avail,
        choices=choices,
    ).numpy()[0]

    assert logits.shape == (4,)
    assert logits[1] == -1e9
    assert logits[3] == -1e9
    assert np.isfinite(logits[0])
    assert np.isfinite(logits[2])


def test_base_featurebased_can_be_saved_and_loaded(tmp_path: Path) -> None:
    """
    Check BaseFeatureBasedDeepHalo can be saved and loaded as a .keras model.
    """
    base = _make_base(num_items=4, depth=2, width=8, heads=2)

    x = tf.constant(
        [
            [
                [1.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],  # padded
                [0.3, 0.1, 0.0],
                [0.0, 0.0, 0.0],  # padded
            ]
        ],
        dtype=tf.float32,
    )
    _ = base(x, training=False)  # build

    save_path = tmp_path / "base_featurebased.keras"
    base.save(save_path)

    loaded = tf.keras.models.load_model(save_path)

    logits = loaded(x, training=False).numpy()[0]
    assert logits.shape == (4,)
    assert logits[1] == -1e9
    assert logits[3] == -1e9
    assert np.isfinite(logits[0])
    assert np.isfinite(logits[2])
