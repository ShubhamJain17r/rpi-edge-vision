"""
High-performance Object Detection and Classification inference engine.
Optimized for Raspberry Pi 4 and Raspberry Pi 5 with:
- Dual-engine: Ultralytics PyTorch (.pt) and Native ONNX Runtime (.onnx)
- ARM NEON SIMD acceleration via ONNX Runtime CPUExecutionProvider
- Ultra-fast C++ OpenCV NMS post-processing
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Standardized detection result container."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    width_px: int
    height_px: int
    center: Tuple[int, int]
    # Distance estimation will attach to this:
    distance_m: Optional[float] = None
    relative_depth: Optional[float] = None


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resizes and pads image to new_shape while preserving aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class EdgeDetector:
    """
    Object detector and classifier wrapping Ultralytics and native ONNX Runtime.
    Automatically handles PyTorch .pt models and high-speed ONNX .onnx models.
    """

    def __init__(
        self,
        model_path: str = "weights/yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        num_threads: int = 4
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.num_threads = num_threads

        self.model = None
        self.is_onnx = False
        self.onnx_session = None
        self.input_name = None
        self.output_name = None
        self.class_names: Dict[int, str] = {}

        self._load_model()

    def _load_model(self):
        """Loads model based on file extension (.pt or .onnx)."""
        if not os.path.isabs(self.model_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            full_path = os.path.join(base_dir, self.model_path)
        else:
            full_path = self.model_path

        # If .onnx counterpart exists, prefer ONNX for edge speed
        onnx_counterpart = full_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_counterpart) and not full_path.endswith(".onnx"):
            logger.info(f"⚡ Detected optimized ONNX counterpart: {onnx_counterpart}. Using ONNX Runtime for edge acceleration.")
            full_path = onnx_counterpart

        if not os.path.exists(full_path):
            logger.warning(f"Model file not found at {full_path}. Attempting default fallback.")
            full_path = "yolov8n.pt"

        logger.info(f"Loading detection model: {full_path}")
        self.is_onnx = full_path.endswith(".onnx")

        if self.is_onnx:
            self._load_onnx(full_path)
        else:
            self._load_ultralytics(full_path)

    def _load_ultralytics(self, model_path: str):
        """Loads model using Ultralytics YOLO API with CPU thread tuning."""
        try:
            import torch
            torch.set_num_threads(self.num_threads)
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            logger.info(f"Ultralytics model loaded successfully. Classes: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"Failed to load Ultralytics model: {e}")
            raise

    def _load_onnx(self, model_path: str):
        """Loads model using ONNX Runtime with optimized multi-threading."""
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.onnx_session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_name = self.onnx_session.get_outputs()[0].name
            self._init_coco_names()
            logger.info(f"✓ Native ONNX Runtime engine active (threads={self.num_threads}, input={self.input_name})")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def _init_coco_names(self):
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        self.class_names = {i: name for i, name in enumerate(coco_classes)}

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Runs object detection on input BGR frame and returns structured detections."""
        if frame is None:
            return []

        h, w = frame.shape[:2]
        results_list: List[DetectionResult] = []

        if self.is_onnx and self.onnx_session is not None:
            # ⚡ OPTIMIZED NATIVE ONNX RUNTIME PIPELINE FOR ARM NEON
            img_padded, ratio, (dw, dh) = letterbox(frame, new_shape=(self.imgsz, self.imgsz))
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            blob = np.transpose(img_float, (2, 0, 1))[np.newaxis, ...]

            # Run inference
            outputs = self.onnx_session.run([self.output_name], {self.input_name: blob})[0]
            # YOLOv8 output shape is [1, 84, 8400]
            preds = np.transpose(outputs[0])  # [8400, 84]

            # Vectorized thresholding
            scores = np.max(preds[:, 4:], axis=1)
            mask = scores > self.conf_threshold

            if np.any(mask):
                valid_preds = preds[mask]
                valid_scores = scores[mask]
                class_ids = np.argmax(valid_preds[:, 4:], axis=1)

                # Extract boxes: cx, cy, w_box, h_box
                cx = valid_preds[:, 0]
                cy = valid_preds[:, 1]
                w_box = valid_preds[:, 2]
                h_box = valid_preds[:, 3]

                # Convert to corners and rescale from letterbox back to original image
                x1 = (cx - w_box / 2 - dw) / ratio
                y1 = (cy - h_box / 2 - dh) / ratio
                x2 = (cx + w_box / 2 - dw) / ratio
                y2 = (cy + h_box / 2 - dh) / ratio

                # Clip coordinates
                x1 = np.clip(x1, 0, w - 1).astype(int)
                y1 = np.clip(y1, 0, h - 1).astype(int)
                x2 = np.clip(x2, 0, w - 1).astype(int)
                y2 = np.clip(y2, 0, h - 1).astype(int)
                box_w = np.maximum(1, x2 - x1)
                box_h = np.maximum(1, y2 - y1)

                # OpenCV C++ NMS
                cv_boxes = [[int(x1[i]), int(y1[i]), int(box_w[i]), int(box_h[i])] for i in range(len(x1))]
                indices = cv2.dnn.NMSBoxes(cv_boxes, valid_scores.tolist(), self.conf_threshold, self.iou_threshold)

                if len(indices) > 0:
                    for i in indices.flatten():
                        bx1, by1, bw, bh = cv_boxes[i]
                        bx2, by2 = bx1 + bw, by1 + bh
                        cls_id = int(class_ids[i])
                        results_list.append(DetectionResult(
                            bbox=(bx1, by1, bx2, by2),
                            class_id=cls_id,
                            class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                            confidence=float(valid_scores[i]),
                            width_px=bw,
                            height_px=bh,
                            center=(bx1 + bw // 2, by1 + bh // 2)
                        ))

        elif self.model is not None:
            # Ultralytics PyTorch inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    center = (x1 + box_w // 2, y1 + box_h // 2)

                    results_list.append(DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        width_px=box_w,
                        height_px=box_h,
                        center=center
                    ))

        return results_list

    def classify_crop(self, crop: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Classifies an individual image crop."""
        if self.model is None or crop is None or crop.size == 0:
            return []

        try:
            results = self.model(crop, verbose=False)
            if len(results) > 0 and hasattr(results[0], "probs") and results[0].probs is not None:
                probs = results[0].probs.data.cpu().numpy()
                top_indices = np.argsort(probs)[::-1][:top_k]
                return [(self.class_names.get(idx, str(idx)), float(probs[idx])) for idx in top_indices]
        except Exception as e:
            logger.debug(f"Classification on crop: {e}")
        return []
