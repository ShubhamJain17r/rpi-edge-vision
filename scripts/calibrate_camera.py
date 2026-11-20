#!/usr/bin/env python3
"""
Camera Focal Length Calibration Tool for Distance Estimation.
Calculates the exact focal length (in pixels) for your specific camera sensor.

Mathematical principle:
    Focal_Length = (Pixel_Width * Known_Distance) / Real_Width

Usage Examples:
    # 1. Direct calculation from known measurements:
    python3 scripts/calibrate_camera.py --real-width 0.21 --distance 0.50 --pixel-width 252

    # 2. Interactive calibration using live camera feed:
    python3 scripts/calibrate_camera.py --interactive --real-width 0.21 --distance 0.50 --source 0
"""

import sys
import logging
import argparse
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_focal_length(pixel_width: float, known_distance_m: float, real_width_m: float) -> float:
    if real_width_m <= 0:
        raise ValueError("Real width must be greater than 0")
    f = (pixel_width * known_distance_m) / real_width_m
    return float(round(f, 2))


def interactive_calibration(source: int, real_width_m: float, known_distance_m: float):
    logger.info(f"Opening camera {source} for interactive calibration...")
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Could not open camera {source}")
        return

    logger.info("Place your object of known width at the exact known distance.")
    logger.info("Drag a bounding box around the target object and press ENTER or SPACE.")

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Failed to grab initial frame.")
        cap.release()
        return

    # OpenCV ROI Selector
    roi = cv2.selectROI("Camera Calibration - Select Target Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cap.release()

    x, y, w, h = roi
    if w <= 0:
        logger.warning("No valid bounding box selected. Calibration aborted.")
        return

    focal_length = calculate_focal_length(w, known_distance_m, real_width_m)
    logger.info("==================================================")
    logger.info("CALIBRATION RESULTS:")
    logger.info(f"Detected pixel width: {w} px")
    logger.info(f"Target real width:    {real_width_m:.3f} m")
    logger.info(f"Target distance:      {known_distance_m:.3f} m")
    logger.info(f"Calculated Focal Length (f): {focal_length:.2f} px")
    logger.info("==================================================")
    logger.info("Add this to your config/default.yaml:")
    logger.info(f"distance -> pinhole -> focal_length: {focal_length}")


def main():
    parser = argparse.ArgumentParser(description="Camera Focal Length Calibration Tool")
    parser.add_argument("--real-width", type=float, default=0.21, help="Real-world width of reference object in meters (e.g. 0.21 for A4 paper width)")
    parser.add_argument("--distance", type=float, default=0.50, help="Known distance from camera to object in meters (e.g. 0.50 m)")
    parser.add_argument("--pixel-width", type=float, default=None, help="Measured width in pixels")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive camera capture GUI")
    parser.add_argument("--source", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    if args.interactive:
        interactive_calibration(args.source, args.real_width, args.distance)
    elif args.pixel_width is not None:
        f = calculate_focal_length(args.pixel_width, args.distance, args.real_width)
        print("==================================================")
        print(f"Calculated Focal Length: {f:.2f} px")
        print(f"Set 'focal_length: {f:.2f}' in config/default.yaml")
        print("==================================================")
    else:
        print("Please provide --pixel-width or run with --interactive. Use -h for help.")


if __name__ == "__main__":
    main()
