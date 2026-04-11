"""
Load training images and bounding boxes from CSV.

Beginner notes:
- Bounding boxes are in pixel coordinates: xmin, ymin, xmax, ymax (inclusive-ish corners).
- After resizing an image, box coordinates must be scaled by the same factors as width/height.
- ResNet50 expects a specific pixel normalization; we use keras.applications.resnet50.preprocess_input.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV is required for image loading. Install with: pip install opencv-python"
    ) from exc

logger = logging.getLogger(__name__)

# Default layout (override with arguments or env CAR_OD_DATA_ROOT)
DEFAULT_CSV_NAMES: Tuple[str, ...] = (
    "train_solution_bounding_boxes.csv",
    "train_solution_bounding_boxes (1).csv",
)


def _resolve_data_root(data_root: Optional[Path]) -> Path:
    """Pick project data folder: explicit > env > ../data from this file."""
    if data_root is not None:
        return Path(data_root).resolve()
    env = os.environ.get("CAR_OD_DATA_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def find_labels_csv(data_root: Path, csv_path: Optional[Path] = None) -> Path:
    """
    Return path to labels CSV.

    If csv_path is given, it must exist.
    Otherwise try DEFAULT_CSV_NAMES under data_root.
    """
    if csv_path is not None:
        p = Path(csv_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Labels CSV not found: {p}")
        return p
    for name in DEFAULT_CSV_NAMES:
        candidate = data_root / name
        if candidate.is_file():
            logger.info("Using labels file: %s", candidate)
            return candidate.resolve()
    raise FileNotFoundError(
        f"No labels CSV found in {data_root}. Tried: {', '.join(DEFAULT_CSV_NAMES)}"
    )


def load_annotations_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read CSV with columns: image, xmin, ymin, xmax, ymax.

    Multiple rows per image (several cars) are merged into one **union** box
    so each image has a single regression target (4 numbers).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}") from e

    required = {"image", "xmin", "ymin", "xmax", "ymax"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")

    for col in ("xmin", "ymin", "xmax", "ymax"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["image", "xmin", "ymin", "xmax", "ymax"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with non-numeric box values", dropped)

    # Fix inverted boxes if any
    swap_x = df["xmax"] < df["xmin"]
    if swap_x.any():
        logger.warning("Swapping xmin/xmax for %d rows", int(swap_x.sum()))
        df.loc[swap_x, ["xmin", "xmax"]] = df.loc[swap_x, ["xmax", "xmin"]].values
    swap_y = df["ymax"] < df["ymin"]
    if swap_y.any():
        logger.warning("Swapping ymin/ymax for %d rows", int(swap_y.sum()))
        df.loc[swap_y, ["ymin", "ymax"]] = df.loc[swap_y, ["ymax", "ymin"]].values

    # One row per image: union of all boxes
    grouped = (
        df.groupby("image", as_index=False)
        .agg(
            xmin=("xmin", "min"),
            ymin=("ymin", "min"),
            xmax=("xmax", "max"),
            ymax=("ymax", "max"),
        )
    )
    return grouped


def _read_image_bgr(path: Path) -> np.ndarray:
    """Load image as BGR uint8 (OpenCV default)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _resize_image_and_box(
    image_bgr: np.ndarray,
    box: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image to target_size (width, height) and scale box [xmin, ymin, xmax, ymax].

    Returns RGB image uint8 for further preprocess_input, and scaled float box.
    """
    h, w = image_bgr.shape[:2]
    tw, th = target_size
    sx = tw / float(w)
    sy = th / float(h)
    scaled = box.astype(np.float32).copy()
    scaled[0] *= sx
    scaled[1] *= sy
    scaled[2] *= sx
    scaled[3] *= sy
    # Clip to image bounds
    scaled[0] = np.clip(scaled[0], 0, tw - 1)
    scaled[1] = np.clip(scaled[1], 0, th - 1)
    scaled[2] = np.clip(scaled[2], 0, tw - 1)
    scaled[3] = np.clip(scaled[3], 0, th - 1)
    resized = cv2.resize(image_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb, scaled


def build_training_arrays(
    images_dir: Path,
    annotations: pd.DataFrame,
    target_size: Tuple[int, int] = (224, 224),
    skip_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load every image listed in annotations, resize, preprocess for ResNet50.

    Returns
    -------
    X : np.ndarray, float32, shape (N, H, W, 3) — after preprocess_input
    y : np.ndarray, float32, shape (N, 4) — scaled boxes in resized coordinates
    used_images : list of filenames that were successfully loaded
    """
    images_dir = Path(images_dir).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Training images directory not found: {images_dir}")

    tw, th = target_size
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    used: list[str] = []

    for _, row in annotations.iterrows():
        name = str(row["image"]).strip()
        if not name:
            continue
        path = images_dir / name
        if not path.is_file():
            msg = f"Missing image file: {path}"
            if skip_missing:
                logger.warning(msg)
                continue
            raise FileNotFoundError(msg)

        box = np.array(
            [row["xmin"], row["ymin"], row["xmax"], row["ymax"]], dtype=np.float32
        )
        try:
            bgr = _read_image_bgr(path)
            rgb, box_scaled = _resize_image_and_box(bgr, box, target_size)
            # Lazy import: avoids loading TensorFlow when only parsing CSV.
            from tensorflow.keras.applications.resnet50 import preprocess_input

            # preprocess_input applies ImageNet mean/std scaling expected by ResNet50
            batch_ready = preprocess_input(rgb.astype(np.float32))
            xs.append(batch_ready)
            ys.append(box_scaled)
            used.append(name)
        except Exception as e:
            if skip_missing:
                logger.warning("Skipping %s: %s", name, e)
                continue
            raise

    if not xs:
        raise RuntimeError(
            "No training samples loaded. Check images_dir and CSV image names."
        )

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.stack(ys, axis=0).astype(np.float32)
    return X, y, used


def train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    image_names: Optional[list[str]] = None,
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Stratify-free split (regression). Optionally splits image_names the same way."""
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1")
    if image_names is not None and len(image_names) != len(X):
        raise ValueError("image_names length must match number of samples")
    idx = np.arange(len(X))
    i_train, i_val = train_test_split(
        idx,
        test_size=validation_fraction,
        random_state=random_state,
    )
    X_train, X_val = X[i_train], X[i_val]
    y_train, y_val = y[i_train], y[i_val]
    if image_names is None:
        return X_train, X_val, y_train, y_val, [], []
    names = list(image_names)
    train_names = [names[i] for i in i_train]
    val_names = [names[i] for i in i_val]
    return X_train, X_val, y_train, y_val, train_names, val_names


def load_training_data(
    images_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    target_size: Tuple[int, int] = (224, 224),
    validation_fraction: float = 0.2,
    random_state: int = 42,
    skip_missing: bool = True,
) -> dict:
    """
    End-to-end: resolve paths, load CSV, load images, split train/val.

    Default folders under data_root:
    - training_images/
    - labels CSV (see find_labels_csv)
    """
    root = _resolve_data_root(data_root)
    if images_dir is None:
        images_dir = root / "training_images"
    else:
        images_dir = Path(images_dir).resolve()

    labels_path = find_labels_csv(root, csv_path)
    ann = load_annotations_csv(labels_path)
    X, y, used = build_training_arrays(
        images_dir, ann, target_size=target_size, skip_missing=skip_missing
    )
    X_train, X_val, y_train, y_val, train_names, val_names = train_validation_split(
        X,
        y,
        image_names=used,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_image_names": train_names,
        "val_image_names": val_names,
        "labels_path": labels_path,
        "images_dir": Path(images_dir).resolve(),
        "target_size": target_size,
        "n_samples": len(used),
        "used_images": used,
    }


def list_test_images(testing_dir: Path) -> Iterable[Path]:
    """Yield image paths from testing folder (common extensions)."""
    testing_dir = Path(testing_dir).resolve()
    if not testing_dir.is_dir():
        raise FileNotFoundError(f"Testing images directory not found: {testing_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in sorted(testing_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
