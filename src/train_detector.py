#!/usr/bin/env python3
"""
Transfer Learning script for Custom Object Detection on Raspberry Pi 4 & 5.
Fine-tunes a lightweight YOLO model on custom annotated dataset and exports to ONNX.
"""

import os
import sys
import shutil
import logging
import argparse
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_detection(
    data_yaml: str,
    base_model: str = "weights/yolov8n.pt",
    epochs: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    freeze: int = 10,
    device: str = "cpu",
    output_dir: str = "models",
    export_onnx: bool = True
):
    """
    Runs transfer learning to fine-tune detector on custom dataset.
    Args:
        data_yaml: Path to dataset YAML configuration file.
        base_model: Pre-trained weights to start transfer learning from.
        epochs: Training epochs.
        batch: Batch size (use 4-8 on RPi 4/5).
        imgsz: Training image resolution.
        freeze: Number of backbone layers to freeze (transfers features, saves RAM/CPU).
        device: Device to train on ('cpu' for RPi).
        output_dir: Destination directory for trained weights.
        export_onnx: Whether to export trained model to ONNX for edge acceleration.
    """
    logger.info("==================================================")
    logger.info("Starting Transfer Learning for Object Detection")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Dataset config: {data_yaml}")
    logger.info(f"Epochs: {epochs} | Batch size: {batch} | Img Size: {imgsz}")
    logger.info(f"Frozen backbone layers: {freeze} | Device: {device}")
    logger.info("==================================================")

    os.makedirs(output_dir, exist_ok=True)

    # Load base pre-trained model
    model = YOLO(base_model)

    # Train with transfer learning
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        freeze=freeze,
        device=device,
        project=output_dir,
        name="custom_detector_run",
        exist_ok=True,
        verbose=True
    )

    # Locate best weights
    best_pt = os.path.join(output_dir, "custom_detector_run", "weights", "best.pt")
    target_pt = os.path.join(output_dir, "custom_detector.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        logger.info(f"Trained PyTorch weights saved to: {target_pt}")

        if export_onnx:
            logger.info("Exporting fine-tuned model to ONNX format for accelerated edge inference...")
            trained_model = YOLO(target_pt)
            trained_model.export(format="onnx", imgsz=imgsz, simplify=True)
            exported_onnx = target_pt.replace(".pt", ".onnx")
            logger.info(f"Exported ONNX weights saved to: {exported_onnx}")
    else:
        logger.warning(f"Could not locate best.pt in {output_dir}/custom_detector_run/weights/")

    logger.info("Transfer learning completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for Object Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset config")
    parser.add_argument("--base-model", type=str, default="weights/yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (e.g. 640 or 416)")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu')")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--no-onnx", action="store_true", help="Do not export to ONNX")
    args = parser.parse_args()

    train_detection(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        device=args.device,
        output_dir=args.output_dir,
        export_onnx=not args.no_onnx
    )


if __name__ == "__main__":
    main()
