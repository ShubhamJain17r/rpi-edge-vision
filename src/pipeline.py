"""
High-FPS Edge AI Vision & Distance Pipeline.
Optimized for Raspberry Pi 4 & 5 with:
- Asynchronous threaded inference (camera @ 30 FPS, detection decoupled)
- Safe GUI detection (guards against Qt xcb crashes)
- Real-time terminal console telemetry in --no-web mode
- Instantaneous (<0.05ms) optical pinhole metric distance estimation
"""

import os
import sys
import time
import signal
import logging
import argparse
import ctypes
import threading
from typing import Dict, Any, List, Optional
import yaml
import numpy as np
import cv2

from src.camera import CameraStream
from src.detector import EdgeDetector, DetectionResult
from src.distance import DistanceEstimator
from src.web_streamer import WebStreamer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_qt_gui_supported() -> bool:
    """Checks whether Qt X11 dependencies are satisfied before invoking OpenCV GUI."""
    if not os.environ.get("DISPLAY"):
        return False
    try:
        ctypes.CDLL("libSM.so.6")
        ctypes.CDLL("libICE.so.6")
        return True
    except OSError:
        return False


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
    display_fps: float,
    inf_fps: float,
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

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        label_parts = [f"{det.class_name} {int(det.confidence * 100)}%"]
        if det.distance_m is not None:
            label_parts.append(f"dist: {det.distance_m:.2f}m")
        if det.relative_depth is not None:
            label_parts.append(f"depth: {det.relative_depth}")
        label = " | ".join(label_parts)

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bg_y1 = max(0, y1 - text_h - 8)
        bg_y2 = y1
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), text_bg_color, -1)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), box_color, 1)

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

        cx, cy = det.center
        cv2.circle(annotated, (cx, cy), 3, dist_color, -1)

    status_bar = f"Video: {display_fps:.1f} FPS | AI: {inf_fps:.1f} FPS | Objects: {len(detections)}"
    cv2.rectangle(annotated, (0, 0), (w, 24), (20, 20, 20), -1)
    cv2.putText(annotated, status_bar, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

    if depth_preview is not None:
        pip_w = 140
        pip_h = 105
        pip_depth = cv2.resize(depth_preview, (pip_w, pip_h))
        cv2.rectangle(pip_depth, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 1)
        annotated[28:28 + pip_h, w - pip_w - 8:w - 8] = pip_depth

    return annotated


class AsyncPipeline:
    def __init__(self, config: Dict[str, Any], show_gui: bool = False, enable_web: bool = True):
        self.config = config
        self.show_gui = show_gui
        self.enable_web = enable_web

        cam_cfg = config.get("camera", {})
        det_cfg = config.get("detector", {})
        dist_cfg = config.get("distance", {})
        sys_cfg = config.get("system", {})
        web_cfg = config.get("web_streamer", {})

        # 1. Camera Capture
        self.camera = CameraStream(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            fourcc=cam_cfg.get("fourcc", "MJPG"),
            buffer_size=cam_cfg.get("buffer_size", 1),
            fallback_synthetic=cam_cfg.get("fallback_synthetic", True)
        )

        # 2. Edge Detector
        self.detector = EdgeDetector(
            model_path=det_cfg.get("model_path", "weights/yolov8n.pt"),
            conf_threshold=det_cfg.get("conf_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            imgsz=det_cfg.get("imgsz", 320),
            device=det_cfg.get("device", "cpu"),
            num_threads=sys_cfg.get("num_threads", 4)
        )

        # 3. Distance Engine
        self.distance_engine = DistanceEstimator(config=config)
        self.depth_interval = dist_cfg.get("depth_interval", 5)

        # 4. Web Streamer
        self.streamer = None
        if self.enable_web and web_cfg.get("enabled", True):
            self.streamer = WebStreamer(
                host=web_cfg.get("host", "0.0.0.0"),
                port=web_cfg.get("port", 8080),
                quality=web_cfg.get("quality", 70)
            )
            self.streamer.start()

        # Shared thread-safe state
        self.lock = threading.Lock()
        self.latest_detections: List[DetectionResult] = []
        self.latest_depth_vis: Optional[np.ndarray] = None
        self.inf_fps = 0.0

        self.running = False
        self.worker_thread = None

    def start(self):
        self.camera.start()
        self.running = True
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True, name="AI_InferenceWorker")
        self.worker_thread.start()
        logger.info("Asynchronous Edge AI Inference Worker started.")
        return self

    def _inference_loop(self):
        frame_idx = 0
        last_inf_time = time.time()
        inf_count = 0

        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_idx += 1
            inf_count += 1
            now = time.time()
            if now - last_inf_time >= 1.0:
                self.inf_fps = inf_count / (now - last_inf_time)
                inf_count = 0
                last_inf_time = now

            # 1. Run detection
            dets = self.detector.detect(frame)

            # 2. Run distance & depth
            compute_depth = (frame_idx % self.depth_interval == 0)
            dets, depth_vis = self.distance_engine.process(frame, dets, compute_depth=compute_depth)

            # 3. Update shared state
            with self.lock:
                self.latest_detections = dets
                if depth_vis is not None:
                    self.latest_depth_vis = depth_vis

            time.sleep(0.001)

    def run_display_loop(self):
        window_name = self.config.get("display", {}).get("window_name", "RPi Edge Vision & Distance")
        use_gui = False

        if self.show_gui:
            if is_qt_gui_supported():
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 640, 480)
                    use_gui = True
                    logger.info(f"✓ Native OpenCV GUI window active: '{window_name}'")
                except Exception as e:
                    logger.warning(f"Could not open GUI window ({e}). Running in terminal console mode.")
            else:
                logger.info("ℹ GUI window not available (Qt X11 libraries missing). Running in terminal console mode.")

        display_fps = 0.0
        frame_count = 0
        last_fps_time = time.time()
        last_console_time = time.time()

        if not use_gui and self.streamer is None:
            logger.info("Running in Maximum Performance Mode (--no-web).")
            logger.info("Streaming live detections and metric distance telemetry below:")

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    display_fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now

                with self.lock:
                    dets = list(self.latest_detections)
                    depth_preview = self.latest_depth_vis
                    inf_fps = self.inf_fps

                annotated = annotate_frame(frame, dets, display_fps, inf_fps, depth_preview)

                # 1. Web Streamer
                if self.streamer is not None:
                    self.streamer.update_frame(annotated, display_fps, dets)

                # 2. OpenCV GUI Window
                if use_gui:
                    try:
                        cv2.imshow(window_name, annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                    except Exception as e:
                        logger.warning(f"cv2.imshow failed: {e}. Switching to console output.")
                        use_gui = False

                # 3. Live Console Telemetry (when in --no-web and no GUI)
                if not use_gui and self.streamer is None:
                    if now - last_console_time >= 0.75:
                        det_strs = [f"{d.class_name}({int(d.confidence*100)}%, dist={d.distance_m}m)" for d in dets]
                        det_summary = ", ".join(det_strs) if det_strs else "No objects detected"
                        logger.info(f"[TELEMETRY] Video: {display_fps:.0f} FPS | AI: {inf_fps:.0f} FPS | {det_summary}")
                        last_console_time = now

                time.sleep(0.008)

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()
            if use_gui:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.stop()
        logger.info("Pipeline stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 & 5 High-FPS Edge Vision Pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML file")
    parser.add_argument("--source", type=str, default=None, help="Override camera source (0, video.mp4, etc.)")
    parser.add_argument("--model", type=str, default=None, help="Override detection model path (.pt or .onnx)")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference size (e.g. 320 for max speed)")
    parser.add_argument("--mode", type=str, default=None, help="Distance mode: 'pinhole' (fastest), 'both', or 'midas'")
    parser.add_argument("--no-web", action="store_true", help="Disable web streamer to save CPU cycles")
    parser.add_argument("--gui", action="store_true", help="Enable OpenCV GUI window (requires working Qt/X11)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source is not None:
        config["camera"]["source"] = args.source
    if args.model is not None:
        config["detector"]["model_path"] = args.model
    if args.imgsz is not None:
        config["detector"]["imgsz"] = args.imgsz
    if args.mode is not None:
        config["distance"]["mode"] = args.mode

    pipeline = AsyncPipeline(
        config=config,
        show_gui=args.gui or config.get("display", {}).get("gui_window", False),
        enable_web=not args.no_web
    )
    pipeline.start()
    pipeline.run_display_loop()


if __name__ == "__main__":
    main()
