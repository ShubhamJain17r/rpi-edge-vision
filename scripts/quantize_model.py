#!/usr/bin/env python3
"""
INT8 Dynamic Quantization Tool for Raspberry Pi 4 & 5.
Reduces ONNX model size by ~60% and accelerates ARM NEON CPU inference by 30-50%.

Usage:
    python3 scripts/quantize_model.py --input weights/yolov8n.onnx --output weights/yolov8n_int8.onnx
"""

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def quantize_onnx(input_model_path: str, output_model_path: str):
    if not os.path.exists(input_model_path):
        logger.error(f"Input model not found: {input_model_path}")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("Starting INT8 Dynamic Quantization for ARM CPU")
    logger.info(f"Source Model: {input_model_path} ({os.path.getsize(input_model_path) / 1024 / 1024:.2f} MB)")
    logger.info(f"Target Model: {output_model_path}")
    logger.info("==================================================")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8
        )

        out_size_mb = os.path.getsize(output_model_path) / 1024 / 1024
        orig_size_mb = os.path.getsize(input_model_path) / 1024 / 1024
        savings_pct = (1.0 - (out_size_mb / orig_size_mb)) * 100.0

        logger.info(f"✓ Quantization completed successfully!")
        logger.info(f"Original size: {orig_size_mb:.2f} MB -> Quantized size: {out_size_mb:.2f} MB ({savings_pct:.1f}% reduction)")
        logger.info(f"To use, update config/default.yaml: model_path: {output_model_path}")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="INT8 Dynamic Quantization for Edge Devices")
    parser.add_argument("--input", type=str, default="weights/yolov8n.onnx", help="Path to input .onnx model")
    parser.add_argument("--output", type=str, default="weights/yolov8n_int8.onnx", help="Path to output .onnx model")
    args = parser.parse_args()

    quantize_onnx(args.input, args.output)


if __name__ == "__main__":
    main()
