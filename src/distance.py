"""
Dual Distance Calculation Engine for Edge Raspberry Pi (RPi 4 & RPi 5):
1. Geometric Optical Pinhole Model: Calculates real-world metric distance (meters/cm)
   from camera focal length and known real-world class dimensions.
2. AI Monocular Depth Model (MiDaS v2.1 Small): Generates dense relative depth maps
   using ONNX Runtime CPU.
3. Temporal Distance Smoothing: Removes webcam pixel jitter via Exponential Moving Average (EMA).
"""

import os
import math
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

from src.detector import DetectionResult

logger = logging.getLogger(__name__)


class TemporalDistanceSmoother:
    """
    Stabilizes distance readings on live webcams by filtering out single-frame pixel jitter.
    Tracks detection centers across frames and applies an Exponential Moving Average (EMA).
    """

    def __init__(self, alpha: float = 0.6, max_distance_jump_m: float = 1.0, memory_frames: int = 15):
        self.alpha = alpha  # Weight for new measurement (0.0 to 1.0)
        self.max_distance_jump = max_distance_jump_m
        self.memory_frames = memory_frames
        self.tracked_objects: Dict[str, Dict[str, Any]] = {}

    def smooth(self, detection: DetectionResult) -> float:
        if detection.distance_m is None:
            return 0.0

        raw_dist = detection.distance_m
        cx, cy = detection.center
        cls_name = detection.class_name

        # Find closest tracked object of the same class within reasonable pixel radius
        best_key = None
        best_dist_px = float("inf")

        for key, track in list(self.tracked_objects.items()):
            if track["class"] == cls_name:
                dx = cx - track["center"][0]
                dy = cy - track["center"][1]
                dist_px = math.hypot(dx, dy)
                if dist_px < 80 and dist_px < best_dist_px:
                    best_dist_px = dist_px
                    best_key = key

        if best_key is not None:
            prev_dist = self.tracked_objects[best_key]["distance"]
            # Apply EMA
            smoothed_dist = self.alpha * raw_dist + (1.0 - self.alpha) * prev_dist
            self.tracked_objects[best_key] = {
                "class": cls_name,
                "center": (cx, cy),
                "distance": smoothed_dist,
                "age": 0
            }
            return float(round(smoothed_dist, 2))
        else:
            # New track
            new_key = f"{cls_name}_{len(self.tracked_objects)}_{int(cx)}_{int(cy)}"
            self.tracked_objects[new_key] = {
                "class": cls_name,
                "center": (cx, cy),
                "distance": raw_dist,
                "age": 0
            }
            return float(round(raw_dist, 2))

    def update_age(self):
        """Ages out stale tracks not seen in recent frames."""
        keys_to_remove = []
        for key in self.tracked_objects:
            self.tracked_objects[key]["age"] += 1
            if self.tracked_objects[key]["age"] > self.memory_frames:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.tracked_objects[key]


class PinholeDistanceEstimator:
    """
    Computes metric distance (meters) using camera focal length and perspective geometry.
    Formula: Distance = (Real_Dimension * Focal_Length) / Pixel_Dimension
    """

    def __init__(self, focal_length: float = 600.0, class_dimensions: Optional[Dict[str, Dict[str, float]]] = None):
        self.focal_length = focal_length
        self.class_dimensions = class_dimensions or {}

    def estimate_distance(self, detection: DetectionResult) -> float:
        cls_name = detection.class_name.lower()
        dims = self.class_dimensions.get(cls_name, self.class_dimensions.get("default", {"width": 0.30, "height": 0.30}))

        real_w = dims.get("width", 0.30)
        real_h = dims.get("height", 0.30)

        # For upright objects like persons or bottles, height is generally less sensitive to yaw rotation
        if cls_name in ["person", "bottle", "wine glass", "vase"]:
            dist = (real_h * self.focal_length) / max(1, detection.height_px)
        elif cls_name in ["car", "bus", "truck", "couch", "bed"]:
            dist = (real_w * self.focal_length) / max(1, detection.width_px)
        else:
            dist_w = (real_w * self.focal_length) / max(1, detection.width_px)
            dist_h = (real_h * self.focal_length) / max(1, detection.height_px)
            dist = (dist_w + dist_h) / 2.0

        return float(round(dist, 2))


class MidasDepthEstimator:
    """
    Monocular Depth Estimation using MiDaS v2.1 Small ONNX model.
    Optimized for multi-core ARM NEON CPU inference.
    """

    def __init__(
        self,
        model_path: str = "weights/midas_v21_small.onnx",
        input_size: Tuple[int, int] = (256, 256),
        num_threads: int = 4
    ):
        self.input_size = input_size
        self.num_threads = num_threads
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_loaded = False

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.isabs(model_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            full_path = os.path.join(base_dir, model_path)
        else:
            full_path = model_path

        if not os.path.exists(full_path):
            logger.warning(f"MiDaS ONNX model not found at {full_path}. Operating in fallback mode.")
            return

        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                full_path,
                sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.is_loaded = True
            logger.info("MiDaS Depth Estimation model loaded successfully in ONNX Runtime.")
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS ONNX model: {e}")

    def predict_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Runs monocular depth estimation on the input BGR frame."""
        if not self.is_loaded or self.session is None or frame is None:
            return None

        orig_h, orig_w = frame.shape[:2]

        # Fast SIMD BGR -> RGB resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: img})
        disparity = outputs[0][0]

        disparity_resized = cv2.resize(disparity, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return disparity_resized

    def colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        if depth_map is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-5:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)

        depth_u8 = (depth_norm * 255.0).astype(np.uint8)
        colorized = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        return colorized


class DistanceEstimator:
    """
    Unified Distance & Depth Engine combining Geometric Pinhole, MiDaS AI Depth, and Temporal Smoothing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dist_cfg = config.get("distance", {})
        self.mode = dist_cfg.get("mode", "both")

        # Initialize Pinhole
        pinhole_cfg = dist_cfg.get("pinhole", {})
        focal_length = pinhole_cfg.get("focal_length", 600.0)
        class_dims = pinhole_cfg.get("class_dimensions", {})
        self.pinhole = PinholeDistanceEstimator(focal_length=focal_length, class_dimensions=class_dims)

        # Initialize MiDaS
        midas_cfg = dist_cfg.get("midas", {})
        model_path = midas_cfg.get("model_path", "weights/midas_v21_small.onnx")
        input_size = tuple(midas_cfg.get("input_size", [256, 256]))
        num_threads = config.get("system", {}).get("num_threads", 4)
        self.midas = MidasDepthEstimator(model_path=model_path, input_size=input_size, num_threads=num_threads)

        # Initialize Temporal Smoother for webcam stability
        smoothing_enabled = dist_cfg.get("enable_smoothing", True)
        self.smoother = TemporalDistanceSmoother(alpha=0.65) if smoothing_enabled else None

        # Cached depth map for decoupled cadence
        self.cached_depth_map = None
        self.cached_depth_vis = None

    def process(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        compute_depth: bool = True
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """
        Calculates distance for all detections and produces an optional colorized depth map.
        Supports compute_depth=False to reuse cached depth map (decoupled cadence).
        """
        if self.mode in ["midas", "both"] and self.midas.is_loaded:
            if compute_depth:
                self.cached_depth_map = self.midas.predict_depth_map(frame)
                if self.cached_depth_map is not None:
                    self.cached_depth_vis = self.midas.colorize_depth(self.cached_depth_map)

        depth_map = self.cached_depth_map
        colorized_depth = self.cached_depth_vis

        for det in detections:
            # 1. Optical Pinhole Metric Distance
            if self.mode in ["pinhole", "both"]:
                raw_dist = self.pinhole.estimate_distance(det)
                det.distance_m = raw_dist

                # Apply temporal webcam smoothing
                if self.smoother is not None:
                    det.distance_m = self.smoother.smooth(det)

            # 2. MiDaS AI Depth (median relative disparity inside bounding box)
            if depth_map is not None:
                x1, y1, x2, y2 = det.bbox
                h, w = depth_map.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                if x2 > x1 and y2 > y1:
                    box_depth = depth_map[y1:y2, x1:x2]
                    median_val = float(np.median(box_depth))
                    det.relative_depth = round(median_val, 2)

        if self.smoother is not None:
            self.smoother.update_age()

        return detections, colorized_depth
