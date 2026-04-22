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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

logger = logging.getLogger(__name__)


def bbox_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """IoU for boxes in [xmin, ymin, xmax, ymax] format."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_x1 = tf.minimum(y_true[:, 0], y_true[:, 2])
    true_y1 = tf.minimum(y_true[:, 1], y_true[:, 3])
    true_x2 = tf.maximum(y_true[:, 0], y_true[:, 2])
    true_y2 = tf.maximum(y_true[:, 1], y_true[:, 3])

    pred_x1 = tf.minimum(y_pred[:, 0], y_pred[:, 2])
    pred_y1 = tf.minimum(y_pred[:, 1], y_pred[:, 3])
    pred_x2 = tf.maximum(y_pred[:, 0], y_pred[:, 2])
    pred_y2 = tf.maximum(y_pred[:, 1], y_pred[:, 3])

    inter_x1 = tf.maximum(true_x1, pred_x1)
    inter_y1 = tf.maximum(true_y1, pred_y1)
    inter_x2 = tf.minimum(true_x2, pred_x2)
    inter_y2 = tf.minimum(true_y2, pred_y2)

    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)
    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    true_area = tf.maximum(0.0, (true_x2 - true_x1) * (true_y2 - true_y1))
    pred_area = tf.maximum(0.0, (pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    union = true_area + pred_area - inter_area

    return inter_area / (union + 1e-7)


def bbox_iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean IoU metric for training logs."""
    return tf.reduce_mean(bbox_iou(y_true, y_pred))


def combined_bbox_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Robust regression objective:
    - Huber loss keeps coordinate learning stable.
    - IoU penalty encourages better overlap (localization quality).
    """
    huber = keras.losses.Huber(delta=8.0, reduction=keras.losses.Reduction.NONE)
    huber_loss = huber(y_true, y_pred)
    iou_penalty = 1.0 - bbox_iou(y_true, y_pred)
    return huber_loss + (20.0 * iou_penalty)


def build_bbox_resnet50(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    trainable_base: bool = False,
    weights: str = "imagenet",
    dense_units: tuple[int, ...] = (512, 256, 128),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Build Keras Functional model: ResNet50 -> GAP -> Dense blocks with regularization -> 4 outputs.

    Parameters
    ----------
    trainable_base : bool
        If False, freeze ResNet50 weights. If True, fine-tune the last layers.
    weights : str
        'imagenet' loads pretrained weights; None trains from scratch.
    dense_units : tuple
        Sizes of dense layers before output.
    dropout_rate : float
        Dropout rate for regularization (default: 0.3).
    l2_reg : float
        L2 regularization coefficient (default: 1e-4).
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

    # Fine-tune: freeze early layers, train later layers
    if trainable_base:
        # Freeze first 100 layers, train last 50+ layers for fine-tuning
        for layer in base.layers[:-50]:
            layer.trainable = False
        logger.info("ResNet50 base: freezing first layers, training last 50+ layers (fine-tuning)")
    else:
        base.trainable = False
        logger.info("ResNet50 base is frozen (trainable_base=False)")

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    
    # Add dense layers with dropout and L2 regularization
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units, 
            activation="relu", 
            name=f"dense_{i}",
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    
    # Linear outputs for regression (explicit for teaching)
    outputs = layers.Dense(4, activation="linear", name="bbox")(x)

    model = keras.Model(inputs=base.input, outputs=outputs, name="resnet50_bbox")
    return model


def compile_bbox_model(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    loss=combined_bbox_loss,
    metrics: Optional[list] = None,
) -> keras.Model:
    """Attach optimizer and bbox regression losses/metrics."""
    if metrics is None:
        metrics = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            bbox_iou_metric,
        ]
    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    except Exception as e:
        raise RuntimeError(f"model.compile failed: {e}") from e
    return model
