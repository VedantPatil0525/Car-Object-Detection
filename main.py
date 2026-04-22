#!/usr/bin/env python3
"""
Car bounding-box regression with ResNet50 — entry point.

Typical layout (override with flags or CAR_OD_DATA_ROOT):
  data/training_images/
  data/train_solution_bounding_boxes.csv
  data/testing_images/

Outputs:
  models/resnet50_bbox_<optimizer>.keras
  outputs/loss_comparison.png
  outputs/val_pred_*.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tensorflow import keras

from src.data_loader import find_labels_csv, load_training_data
from src.train import (
    OPTIMIZER_CONFIGS,
    load_histories_from_json,
    plot_loss_comparison,
    print_optimizer_comparison,
    train_all_optimizers,
)
from src.visualize import save_test_predictions, save_validation_prediction_grid


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train ResNet50 bbox regressors (TensorFlow/Keras)")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing training_images, testing_images, and CSV (default: ./data)",
    )
    p.add_argument("--images-dir", type=Path, default=None, help="Override training images directory")
    p.add_argument("--csv", type=Path, default=None, help="Override path to train_solution_bounding_boxes.csv")
    p.add_argument("--testing-dir", type=Path, default=None, help="Folder with test images")
    p.add_argument("--models-dir", type=Path, default=root / "models", help="Where to save .keras models")
    p.add_argument("--outputs-dir", type=Path, default=root / "outputs", help="Plots and prediction images")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per optimizer (default: 50)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-trainable-base", action="store_true", help="Freeze ResNet50 (default: fine-tune)")
    p.add_argument("--no-weights", action="store_true", help="Do not use ImageNet weights (random init)")
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; only visualize using saved models in --models-dir",
    )
    p.add_argument("--viz-samples", type=int, default=8, help="Validation images to save with boxes")
    p.add_argument(
        "--optimizer-viz",
        type=str,
        default="adam",
        choices=list(OPTIMIZER_CONFIGS.keys()),
        help="Which saved model to use for sample prediction images",
    )
    return p.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")
    args = _parse_args()

    try:
        data_root = args.data_root
        if args.images_dir is None and data_root is not None:
            train_img_dir = Path(data_root) / "training_images"
        elif args.images_dir is not None:
            train_img_dir = Path(args.images_dir)
        else:
            train_img_dir = None

        csv_path = args.csv
        if csv_path is None and data_root is not None:
            try:
                csv_path = find_labels_csv(Path(data_root), None)
            except FileNotFoundError:
                csv_path = None

        if args.testing_dir is not None:
            testing_dir = Path(args.testing_dir)
        elif data_root is not None:
            testing_dir = Path(data_root) / "testing_images"
        else:
            testing_dir = Path(__file__).resolve().parent / "data" / "testing_images"

        args.models_dir.mkdir(parents=True, exist_ok=True)
        args.outputs_dir.mkdir(parents=True, exist_ok=True)

        weights = None if args.no_weights else "imagenet"

        log.info("Loading dataset...")
        data = load_training_data(
            images_dir=train_img_dir,
            csv_path=csv_path,
            data_root=data_root,
            validation_fraction=args.val_fraction,
            random_state=args.seed,
        )
        log.info("Loaded %d samples (train/val split done).", data["n_samples"])

        histories: Optional[Dict[str, Any]] = None
        if not args.skip_training:
            histories = train_all_optimizers(
                data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                models_dir=args.models_dir,
                weights=weights,
                trainable_base=not args.no_trainable_base,
            )
            plot_loss_comparison(histories, args.outputs_dir / "loss_comparison.png")
        else:
            log.info("Skipping training (--skip-training).")

        model_path = args.models_dir / f"resnet50_bbox_{args.optimizer_viz}.keras"
        if not model_path.is_file():
            log.error("Model not found for visualization: %s", model_path)
            return 1

        log.info("Loading model for visualization: %s", model_path)
        model = keras.models.load_model(model_path)

        save_validation_prediction_grid(
            model,
            images_dir=data["images_dir"],
            val_image_names=data["val_image_names"],
            y_val=data["y_val"],
            target_size=tuple(data["target_size"]),
            outputs_dir=args.outputs_dir,
            max_images=args.viz_samples,
            prefix=f"val_pred_{args.optimizer_viz}",
        )

        if testing_dir.is_dir():
            log.info("Saving test-set prediction images from %s", testing_dir)
            save_test_predictions(
                model,
                testing_dir=testing_dir,
                outputs_dir=args.outputs_dir,
                target_size=(224, 224),
                prefix=f"test_pred_{args.optimizer_viz}",
            )
        else:
            log.warning("Testing directory not found (skip test viz): %s", testing_dir)

        histories_for_report: Dict[str, Any] = (
            histories if histories is not None else load_histories_from_json(args.models_dir)
        )
        print_optimizer_comparison(histories_for_report)

        log.info("Done. Models: %s | Outputs: %s", args.models_dir, args.outputs_dir)
        return 0

    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    except ValueError as e:
        log.error("%s", e)
        return 3
    except Exception as e:
        log.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
