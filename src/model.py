"""
ResNet50 backbone for bounding-box regression (4 outputs: xmin, ymin, xmax, ymax).

Beginner notes:
- include_top=False removes the original ImageNet classifier head.
- GlobalAveragePooling2D turns the 7x7x2048 feature map into a single vector per image.
- The final Dense(4) layer outputs box coordinates; linear activation is the default for regression.
"""

from __future__ import annotations

import logging
from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

logger = logging.getLogger(__name__)


def build_bbox_resnet50(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    trainable_base: bool = False,
    weights: str = "imagenet",
    dense_units: tuple[int, ...] = (256, 128),
) -> keras.Model:
    """
    Build Keras Functional model: ResNet50 -> GAP -> Dense blocks -> 4 outputs.

    Parameters
    ----------
    trainable_base : bool
        If False, freeze ResNet50 weights (faster training, less overfit on small data).
    weights : str
        'imagenet' loads pretrained weights; None trains from scratch.
    """
    try:
        base = ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as e:
        logger.exception("Failed to create ResNet50 backbone")
        raise RuntimeError(
            "Could not load ResNet50. Check TensorFlow/Keras install and network for weights."
        ) from e

    base.trainable = trainable_base
    if not trainable_base:
        logger.info("ResNet50 base is frozen (trainable_base=False)")

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
    # Linear outputs for regression (explicit for teaching)
    outputs = layers.Dense(4, activation="linear", name="bbox")(x)

    model = keras.Model(inputs=base.input, outputs=outputs, name="resnet50_bbox")
    return model


def compile_bbox_model(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    loss: str = "mse",
    metrics: Optional[list] = None,
) -> keras.Model:
    """Attach optimizer and MSE loss (mean squared error on box coordinates)."""
    if metrics is None:
        metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    except Exception as e:
        raise RuntimeError(f"model.compile failed: {e}") from e
    return model
