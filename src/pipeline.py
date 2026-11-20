"""
Full-Screen Edge AI Vision & Distance Pipeline.
Optimized for Raspberry Pi 4 & 5 (and local testing on Linux PC) with:
- Dedicated Full-Screen OpenCV GUI window (cv2.imshow)
- Real-time YOLOv8 object detection + optical pinhole metric distance estimation
- Asynchronous decoupled worker architecture for 30 FPS camera capture and high-speed inference
- Clean exit on "ESC" or "q" without container respawn
- Press "f" to toggle full-screen / windowed mode
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
    """Draws bounding boxes, class labels, metric distance tags, and HUD status bar."""
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

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bg_y1 = max(0, y1 - text_h - 8)
        bg_y2 = y1
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), text_bg_color, -1)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, bg_y2), box_color, 1)

        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cx, cy = det.center
        cv2.circle(annotated, (cx, cy), 3, dist_color, -1)

    # Top HUD Status Bar
    status_bar = f"Video: {display_fps:.1f} FPS | AI: {inf_fps:.1f} FPS | Objects: {len(detections)} | [ESC/q: Quit | f: Fullscreen]"
    cv2.rectangle(annotated, (0, 0), (w, 26), (20, 20, 20), -1)
    cv2.putText(annotated, status_bar, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

    # Monocular depth map Picture-in-Picture preview (if enabled)
    if depth_preview is not None:
        pip_w = 140
        pip_h = 105
        pip_depth = cv2.resize(depth_preview, (pip_w, pip_h))
        cv2.rectangle(pip_depth, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 1)
        annotated[30:30 + pip_h, w - pip_w - 8:w - 8] = pip_depth

    return annotated


class AsyncPipeline:
    def __init__(self, config: Dict[str, Any], fullscreen: bool = True):
        self.config = config
        self.fullscreen = fullscreen

        cam_cfg = config.get("camera", {})
        det_cfg = config.get("detector", {})
        dist_cfg = config.get("distance", {})
        sys_cfg = config.get("system", {})

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

        if is_qt_gui_supported():
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.resizeWindow(window_name, 640, 480)
                use_gui = True
                mode_str = "Full Screen" if self.fullscreen else "Windowed"
                logger.info(f"✓ Native OpenCV GUI window active: '{window_name}' ({mode_str} mode)")
                logger.info("ℹ Controls: Press 'q' or ESC in the window to quit. Press 'f' to toggle fullscreen.")
            except Exception as e:
                logger.warning(f"Could not open GUI window ({e}). Running in terminal console mode.")
        else:
            logger.info("ℹ Display not available (Qt X11 libraries or $DISPLAY missing). Running in terminal console mode.")

        display_fps = 0.0
        frame_count = 0
        last_fps_time = time.time()
        last_console_time = time.time()

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

                # Render to local OpenCV GUI Window
                if use_gui:
                    try:
                        cv2.imshow(window_name, annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q"), ord("Q")):
                            logger.info("Exit requested by user ('q' or ESC pressed). Shutting down...")
                            break
                        elif key in (ord("f"), ord("F")):
                            self.fullscreen = not self.fullscreen
                            prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                            mode = "Enabled" if self.fullscreen else "Disabled"
                            logger.info(f"Fullscreen toggled: {mode}")
                    except Exception as e:
                        logger.warning(f"cv2.imshow failed: {e}. Switching to console output.")
                        use_gui = False

                # Live Console Telemetry (every ~2 seconds)
                if now - last_console_time >= 2.0:
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
    parser.add_argument("--windowed", action="store_true", help="Launch in windowed mode instead of full screen")
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

    fullscreen_default = config.get("display", {}).get("fullscreen", True)
    is_fullscreen = False if args.windowed else fullscreen_default

    pipeline = AsyncPipeline(
        config=config,
        fullscreen=is_fullscreen
    )
    pipeline.start()
    pipeline.run_display_loop()
    sys.exit(0)


if __name__ == "__main__":
    main()
