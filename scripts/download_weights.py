#!/usr/bin/env python3
"""
Downloads edge-optimized pre-trained weights during Docker build or setup.
Bakes weights into /app/weights/ so the container can run 100% offline.
"""

import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))

MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8n-cls.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt",
    "midas_v21_small.onnx": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
}


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        logger.info(f"Already exists: {dest_path} ({os.path.getsize(dest_path)} bytes)")
        return

    logger.info(f"Downloading {url} -> {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded: {dest_path} ({os.path.getsize(dest_path)} bytes)")
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        # Try fallback if available
        if "yolov8" in dest_path:
            try:
                from ultralytics import YOLO
                model_name = os.path.basename(dest_path)
                logger.info(f"Attempting download via Ultralytics API for {model_name}...")
                model = YOLO(model_name)
                if os.path.exists(model_name) and model_name != dest_path:
                    os.rename(model_name, dest_path)
            except Exception as ex:
                logger.error(f"Fallback download also failed: {ex}")


def export_onnx_models():
    """Optionally export YOLO models to ONNX for optimized edge CPU inference."""
    try:
        from ultralytics import YOLO
        yolo_pt = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
        yolo_onnx = os.path.join(WEIGHTS_DIR, "yolov8n.onnx")
        if os.path.exists(yolo_pt) and not os.path.exists(yolo_onnx):
            logger.info("Exporting YOLOv8n to ONNX format for edge CPU acceleration...")
            model = YOLO(yolo_pt)
            model.export(format="onnx", imgsz=640, simplify=True)
            exported_file = yolo_pt.replace(".pt", ".onnx")
            if os.path.exists(exported_file) and exported_file != yolo_onnx:
                os.rename(exported_file, yolo_onnx)
            logger.info("YOLOv8n ONNX export complete.")
    except Exception as e:
        logger.info(f"Skipping ONNX export (will use PyTorch engine or export at runtime): {e}")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(WEIGHTS_DIR, filename)
        download_file(url, dest)

    export_onnx_models()
    logger.info("All base weights are ready in %s", WEIGHTS_DIR)


if __name__ == "__main__":
    main()
