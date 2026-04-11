"""
Train bbox regression models with different optimizers and save artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from src.model import build_bbox_resnet50, compile_bbox_model

logger = logging.getLogger(__name__)

# SGD needs a smaller LR than raw SGD defaults: with frozen ResNet + MSE on ~0–224
# coordinates, lr=1e-3 often blows up gradients (weights → NaN, loss → NaN). Adam/RMSprop
# adapt per-parameter; SGD does not, so match the ~1e-4 scale used for Adam.
OPTIMIZER_CONFIGS: Dict[str, keras.optimizers.Optimizer] = {
    "adam": keras.optimizers.Adam(learning_rate=1e-4),
    "sgd": keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    "rmsprop": keras.optimizers.RMSprop(learning_rate=1e-4),
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_one_optimizer(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    models_dir: Path,
    weights: str = "imagenet",
    trainable_base: bool = False,
) -> keras.callbacks.History:
    """Train a single model; save .keras weights and history JSON."""
    if name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"Unknown optimizer name: {name}. Use {list(OPTIMIZER_CONFIGS)}")

    _ensure_dir(models_dir)
    model = build_bbox_resnet50(weights=weights, trainable_base=trainable_base)
    compile_bbox_model(model, optimizer=OPTIMIZER_CONFIGS[name])

    logger.info("Training with optimizer=%s, samples=%d", name, len(X_train))
    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
    except Exception as e:
        raise RuntimeError(f"Training failed for optimizer {name}: {e}") from e

    out_path = models_dir / f"resnet50_bbox_{name}.keras"
    try:
        model.save(out_path)
        logger.info("Saved model: %s", out_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {out_path}: {e}") from e

    hist_path = models_dir / f"history_{name}.json"

    def _json_safe_float(x: Any) -> Optional[float]:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf

    serializable = {
        k: [_json_safe_float(x) for x in v] for k, v in history.history.items()
    }
    hist_path.write_text(
        json.dumps(serializable, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return history


def train_all_optimizers(
    data: Dict[str, Any],
    epochs: int = 15,
    batch_size: int = 16,
    models_dir: Optional[Path] = None,
    weights: str = "imagenet",
    trainable_base: bool = False,
) -> Dict[str, keras.callbacks.History]:
    """
    Train adam, sgd, and rmsprop models on the same split.

    `data` must contain X_train, y_train, X_val, y_val (from data_loader.load_training_data).
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir = Path(models_dir).resolve()

    required = ("X_train", "y_train", "X_val", "y_val")
    for k in required:
        if k not in data:
            raise KeyError(f"data dict missing key '{k}'")

    histories: Dict[str, keras.callbacks.History] = {}
    for opt_name in OPTIMIZER_CONFIGS:
        histories[opt_name] = train_one_optimizer(
            opt_name,
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            epochs=epochs,
            batch_size=batch_size,
            models_dir=models_dir,
            weights=weights,
            trainable_base=trainable_base,
        )
    return histories


def plot_loss_comparison(
    histories: Dict[str, keras.callbacks.History],
    output_path: Path,
    title: str = "Training loss (MSE) — optimizer comparison",
) -> None:
    """Save a single figure with train/val loss curves for each optimizer."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    plt.figure(figsize=(10, 6))
    for name, h in histories.items():
        if "loss" in h.history:
            plt.plot(h.history["loss"], label=f"{name} train", linestyle="-")
        if "val_loss" in h.history:
            plt.plot(h.history["val_loss"], label=f"{name} val", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
        logger.info("Saved loss plot: %s", output_path)
    finally:
        plt.close()


def load_histories_from_json(models_dir: Path) -> Dict[str, Dict[str, List[Any]]]:
    """Reload histories saved as history_<name>.json (no Keras objects)."""
    models_dir = Path(models_dir)
    out: Dict[str, Dict[str, List[Any]]] = {}
    for name in OPTIMIZER_CONFIGS:
        p = models_dir / f"history_{name}.json"
        if p.is_file():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _final_val_loss(history_obj: Any) -> Optional[float]:
    """
    Last finite validation loss from a Keras History or a dict (e.g. loaded JSON).
    Skips trailing None/NaN (e.g. failed training epochs saved as null).
    """
    if history_obj is None:
        return None
    if hasattr(history_obj, "history"):
        vals = history_obj.history.get("val_loss")
    elif isinstance(history_obj, dict):
        vals = history_obj.get("val_loss")
    else:
        return None
    if not vals:
        return None
    for x in reversed(vals):
        if x is None:
            continue
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf):
            return float(xf)
    return None


def print_optimizer_comparison(histories: Dict[str, Any]) -> None:
    """
    Print final val_loss for Adam, SGD, RMSprop (stdout).
    Accepts Keras History objects or dicts from load_histories_from_json.
    """
    rows = [
        ("Adam", "adam"),
        ("SGD", "sgd"),
        ("RMSprop", "rmsprop"),
    ]
    print("\nOptimizer Comparison:")
    for label, key in rows:
        final = _final_val_loss(histories.get(key))
        if final is None:
            print(f"{label:<9} → Final Val Loss: N/A")
        else:
            print(f"{label:<9} → Final Val Loss: {final:.4f}")


def plot_loss_from_saved_histories(
    histories_json: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title: str = "Training loss (MSE) — optimizer comparison (from saved JSON)",
) -> None:
    """Plot comparison using dicts like {'loss': [...], 'val_loss': [...]}."""
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 6))
    for name, h in histories_json.items():
        if "loss" in h:
            plt.plot(h["loss"], label=f"{name} train", linestyle="-")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label=f"{name} val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        plt.savefig(output_path, dpi=150)
    finally:
        plt.close()
