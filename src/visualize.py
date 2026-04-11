"""
Draw bounding boxes with OpenCV and save prediction visualizations.

Beginner notes:
- OpenCV uses BGR color order; we convert when loading RGB-only arrays if needed.
- Ground-truth and prediction boxes are drawn on the same resized 224×224 image for fair comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

logger = logging.getLogger(__name__)

# BGR colors for drawing
COLOR_GT = (0, 200, 0)  # green — ground truth
COLOR_PRED = (0, 0, 255)  # red — prediction


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    b = box.astype(np.float32).copy()
    b[0] = np.clip(b[0], 0, width - 1)
    b[1] = np.clip(b[1], 0, height - 1)
    b[2] = np.clip(b[2], 0, width - 1)
    b[3] = np.clip(b[3], 0, height - 1)
    return b


def draw_boxes_overlay(
    image_bgr: np.ndarray,
    gt_box: Optional[np.ndarray],
    pred_box: Optional[np.ndarray],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw optional ground-truth (green) and prediction (red) rectangles on a copy of the image.
    Boxes are [xmin, ymin, xmax, ymax] in pixel coords of this image.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    if gt_box is not None:
        b = _clip_box(np.asarray(gt_box), w, h)
        p1 = (int(round(b[0])), int(round(b[1])))
        p2 = (int(round(b[2])), int(round(b[3])))
        cv2.rectangle(out, p1, p2, COLOR_GT, thickness)
        cv2.putText(out, "GT", (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GT, 1)

    if pred_box is not None:
        b = _clip_box(np.asarray(pred_box), w, h)
        p1 = (int(round(b[0])), int(round(b[1])))
        p2 = (int(round(b[2])), int(round(b[3])))
        cv2.rectangle(out, p1, p2, COLOR_PRED, thickness)
        cv2.putText(
            out,
            "Pred",
            (p1[0], min(h - 2, p2[1] + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_PRED,
            1,
        )
    return out


def load_image_bgr_resized(path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Load image from disk, return BGR uint8 resized to (width, height)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    tw, th = size
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)


def preprocess_bgr_for_model(image_bgr: np.ndarray) -> np.ndarray:
    """Resize is assumed done; convert BGR->RGB and apply ResNet50 preprocess_input."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    # preprocess_input expects a batch-like array; add batch dim for one image
    x = preprocess_input(rgb)
    return np.expand_dims(x, axis=0)


def save_validation_prediction_grid(
    model: keras.Model,
    images_dir: Path,
    val_image_names: Iterable[str],
    y_val: np.ndarray,
    target_size: Tuple[int, int],
    outputs_dir: Path,
    max_images: int = 8,
    prefix: str = "val_pred",
) -> None:
    """
    For each validation image (up to max_images), draw GT vs prediction and save to outputs_dir.
    y_val rows must align with val_image_names order (same as data_loader split).
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)
    names = list(val_image_names)
    n = min(max_images, len(names), len(y_val))

    for i in range(n):
        name = names[i]
        path = images_dir / name
        if not path.is_file():
            logger.warning("Skip visualization, missing file: %s", path)
            continue
        try:
            bgr = load_image_bgr_resized(path, target_size)
            x = preprocess_bgr_for_model(bgr)
            pred = model.predict(x, verbose=0)[0]
            gt = y_val[i]
            vis = draw_boxes_overlay(bgr, gt_box=gt, pred_box=pred)
            out_path = outputs_dir / f"{prefix}_{i:02d}_{Path(name).stem}.png"
            if not cv2.imwrite(str(out_path), vis):
                raise RuntimeError(f"cv2.imwrite failed: {out_path}")
            logger.info("Wrote %s", out_path)
        except Exception as e:
            logger.error("Visualization failed for %s: %s", name, e)
            raise


def save_test_predictions(
    model: keras.Model,
    testing_dir: Path,
    outputs_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
    max_images: Optional[int] = None,
    prefix: str = "test_pred",
) -> None:
    """Run inference on images in testing_dir; draw predicted box only; save PNGs."""
    testing_dir = Path(testing_dir)
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in testing_dir.iterdir() if p.suffix.lower() in exts and p.is_file())
    if max_images is not None:
        paths = paths[:max_images]

    for i, path in enumerate(paths):
        try:
            bgr = load_image_bgr_resized(path, target_size)
            x = preprocess_bgr_for_model(bgr)
            pred = model.predict(x, verbose=0)[0]
            vis = draw_boxes_overlay(bgr, gt_box=None, pred_box=pred)
            out_path = outputs_dir / f"{prefix}_{i:03d}_{path.stem}.png"
            if not cv2.imwrite(str(out_path), vis):
                raise RuntimeError(f"cv2.imwrite failed: {out_path}")
            logger.info("Wrote %s", out_path)
        except Exception as e:
            logger.warning("Test inference failed for %s: %s", path, e)
