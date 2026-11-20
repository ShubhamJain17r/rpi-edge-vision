"""
Master Edge AI Vision & Distance Pipeline.
Optimized for Webcam vision on Raspberry Pi 4 & 5 with:
- Decoupled depth estimation cadence (sub-sampled MiDaS inference for maximum FPS)
- Zero-latency hardware frame capture
- Latency profiling and telemetry streaming
"""

import os
import sys
import time
import signal
import logging
import argparse
from typing import Dict, Any, List
import yaml
import numpy as np
import cv2

from src.camera import CameraStream
from src.detector import EdgeDetector, DetectionResult
from src.distance import DistanceEstimator
from src.web_streamer import WebStreamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isabs(config_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(base_dir, config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def annotate_frame(
    frame: np.ndarray,
    detections: List[DetectionResult],
    fps: float,
    inference_ms: float,
    depth_preview: np.ndarray = None
) -> np.ndarray:
    """Draws rich bounding boxes, class labels, distance metrics, and PIP depth preview."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    box_color = (0, 220, 100)      # Bright Green
    text_bg_color = (25, 25, 25)   # Dark Charcoal
    dist_color = (0, 195, 255)     # Amber / Gold

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Build label: "person 87% | 1.45m"
        label_parts = [f"{det.class_name} {int(det.confidence * 100)}%"]
        if det.distance_m is not None:
            label_parts.append(f"dist: {det.distance_m:.2f}m")
        if det.relative_depth is not None:
            label_parts.append(f"d_idx: {det.relative_depth}")
        label = " | ".join(label_parts)

        # Label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - text_h - 8)
        bg_y2 = y1
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), text_bg_color, -1)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), box_color, 1)

        # Text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Center indicator dot
        cx, cy = det.center
        cv2.circle(annotated, (cx, cy), 3, dist_color, -1)

    # Top status bar with FPS and latency profiling
    status_bar = f"FPS: {fps:.1f} | Latency: {inference_ms:.0f}ms | Objects: {len(detections)} | RPi Edge Vision"
    cv2.rectangle(annotated, (0, 0), (w, 24), (20, 20, 20), -1)
    cv2.putText(annotated, status_bar, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

    # Picture-in-picture depth map preview in top-right corner
    if depth_preview is not None:
        pip_w = 160
        pip_h = 120
        pip_depth = cv2.resize(depth_preview, (pip_w, pip_h))
        cv2.rectangle(pip_depth, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 1)
        cv2.putText(pip_depth, "AI Depth Map", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        annotated[30:30 + pip_h, w - pip_w - 10:w - 10] = pip_depth

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 & 5 Edge AI Vision Pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML file")
    parser.add_argument("--source", type=str, default=None, help="Override camera source (0, video.mp4, etc.)")
    parser.add_argument("--model", type=str, default=None, help="Override detection model path (.pt or .onnx)")
    parser.add_argument("--no-web", action="store_true", help="Disable web streamer")
    parser.add_argument("--display", action="store_true", help="Show GUI window (requires active X11 display)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source is not None:
        config["camera"]["source"] = args.source
    if args.model is not None:
        config["detector"]["model_path"] = args.model

    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detector", {})
    dist_cfg = config.get("distance", {})
    sys_cfg = config.get("system", {})
    web_cfg = config.get("web_streamer", {})

    # Optimization: Decoupled depth cadence (e.g. compute MiDaS depth every Nth frame)
    depth_interval = dist_cfg.get("depth_interval", 3)

    logger.info("Initializing Raspberry Pi Edge AI Pipeline...")

    # 1. Initialize Zero-Latency Webcam Streamer
    camera = CameraStream(
        source=cam_cfg.get("source", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
        fourcc=cam_cfg.get("fourcc", "MJPG"),
        buffer_size=cam_cfg.get("buffer_size", 1),
        fallback_synthetic=cam_cfg.get("fallback_synthetic", True)
    )
    camera.start()

    # 2. Initialize Edge Detector (PyTorch or ONNX)
    detector = EdgeDetector(
        model_path=det_cfg.get("model_path", "weights/yolov8n.pt"),
        conf_threshold=det_cfg.get("conf_threshold", 0.35),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        imgsz=det_cfg.get("imgsz", 640),
        device=det_cfg.get("device", "cpu"),
        num_threads=sys_cfg.get("num_threads", 4)
    )

    # 3. Initialize Distance & Depth Engine with Temporal Smoother
    distance_engine = DistanceEstimator(config=config)

    # 4. Initialize Web Streamer
    streamer = None
    if web_cfg.get("enabled", True) and not args.no_web:
        streamer = WebStreamer(
            host=web_cfg.get("host", "0.0.0.0"),
            port=web_cfg.get("port", 8080),
            quality=web_cfg.get("quality", 80)
        )
        streamer.start()

    running = True

    def sig_handler(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received. Stopping pipeline...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    fps = 0.0
    frame_count = 0
    inference_ms = 0.0
    start_time = time.time()
    last_fps_time = time.time()

    logger.info(f"Pipeline running with depth_interval={depth_interval}! Press Ctrl+C to terminate.")

    try:
        while running:
            loop_start = time.time()
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            inf_start = time.time()

            # 1. Detect objects on every frame
            detections = detector.detect(frame)

            # 2. Compute distance (Optical pinhole on 100% of frames, MiDaS depth on every Nth frame)
            should_compute_depth = (frame_count % depth_interval == 0)
            detections, depth_preview = distance_engine.process(
                frame,
                detections,
                compute_depth=should_compute_depth
            )

            inference_ms = (time.time() - inf_start) * 1000.0

            # 3. Draw annotations
            annotated_frame = annotate_frame(frame, detections, fps, inference_ms, depth_preview)

            # 4. Push to Web Streamer
            if streamer is not None:
                streamer.update_frame(annotated_frame, fps, detections)

            # 5. Local display if requested
            if args.display and os.environ.get("DISPLAY"):
                cv2.imshow("RPi Edge Vision & Distance", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.002)

    except Exception as e:
        logger.error(f"Error in pipeline main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up pipeline resources...")
        camera.stop()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped cleanly.")


if __name__ == "__main__":
    main()
