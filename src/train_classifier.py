#!/usr/bin/env python3
"""
Transfer Learning script for Custom Image Classification on Raspberry Pi 4 & 5.
Fine-tunes a lightweight classification model on custom class folders and exports to ONNX.
"""

import os
import sys
import shutil
import logging
import argparse
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_classification(
    data_dir: str,
    base_model: str = "weights/yolov8n-cls.pt",
    epochs: int = 15,
    batch: int = 16,
    imgsz: int = 224,
    device: str = "cpu",
    output_dir: str = "models",
    export_onnx: bool = True
):
    """
    Runs transfer learning to fine-tune classifier on custom dataset directory.
    Dataset structure:
        data_dir/
            train/
                class_a/
                class_b/
            val/
                class_a/
                class_b/
    """
    logger.info("==================================================")
    logger.info("Starting Transfer Learning for Image Classification")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Dataset directory: {data_dir}")
    logger.info(f"Epochs: {epochs} | Batch size: {batch} | Img Size: {imgsz}")
    logger.info(f"Device: {device}")
    logger.info("==================================================")

    os.makedirs(output_dir, exist_ok=True)

    # Load base classification model
    model = YOLO(base_model)

    # Train with transfer learning
    results = model.train(
        data=data_dir,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=output_dir,
        name="custom_classifier_run",
        exist_ok=True,
        verbose=True
    )

    # Save best weights
    best_pt = os.path.join(output_dir, "custom_classifier_run", "weights", "best.pt")
    target_pt = os.path.join(output_dir, "custom_classifier.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        logger.info(f"Trained PyTorch weights saved to: {target_pt}")

        if export_onnx:
            logger.info("Exporting fine-tuned classifier to ONNX format...")
            trained_model = YOLO(target_pt)
            trained_model.export(format="onnx", imgsz=imgsz, simplify=True)
            exported_onnx = target_pt.replace(".pt", ".onnx")
            logger.info(f"Exported ONNX weights saved to: {exported_onnx}")
    else:
        logger.warning(f"Could not locate best.pt in {output_dir}/custom_classifier_run/weights/")

    logger.info("Classification transfer learning completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for Image Classification")
    parser.add_argument("--data", type=str, required=True, help="Path to classification dataset directory")
    parser.add_argument("--base-model", type=str, default="weights/yolov8n-cls.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--no-onnx", action="store_true", help="Do not export to ONNX")
    args = parser.parse_args()

    train_classification(
        data_dir=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        export_onnx=not args.no_onnx
    )


if __name__ == "__main__":
    main()
