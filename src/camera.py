"""
Robust, zero-latency threaded webcam reader for Raspberry Pi (V4L2, USB, CSI).
Includes:
- Hardware buffer size clamping (CAP_PROP_BUFFERSIZE = 1) to eliminate video lag.
- Hardware MJPEG stream decoding via V4L2 for high-FPS USB webcams.
- Thread-safe latest-frame grabber that drops stale queued frames.
- Automatic fallback to synthetic test frames if no hardware webcam is plugged in.
"""

import time
import math
import logging
import threading
from typing import Optional, Tuple, Union
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SyntheticCamera:
    """Generates synthetic video frames with moving objects for headless testing."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_idx = 0
        self.start_time = time.time()

    def read(self) -> Tuple[bool, np.ndarray]:
        self.frame_idx += 1
        t = (time.time() - self.start_time)

        # Create gradient background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = 30
        frame[:, :, 1] = 35
        frame[:, :, 2] = 40

        # Moving simulated "person" (bounding box test target)
        cx = int(self.width / 2 + (self.width / 4) * math.sin(t * 1.2))
        cy = int(self.height / 2 + 30 * math.cos(t * 0.8))
        box_w, box_h = 100, 220
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(self.width, cx + box_w // 2)
        y2 = min(self.height, cy + box_h // 2)

        # Draw simulated human torso & head
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 110, 180), -1)
        cv2.circle(frame, (cx, max(0, y1 - 25)), 25, (210, 180, 140), -1)
        cv2.putText(frame, "Simulated Person", (x1, max(20, y1 - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Moving simulated "bottle" (closer target)
        bx = int(120 + 40 * math.cos(t * 2.0))
        by = int(self.height - 150)
        cv2.rectangle(frame, (bx, by), (bx + 40, by + 120), (60, 180, 75), -1)
        cv2.putText(frame, "Bottle", (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

        # Grid lines for visual depth cue
        for y in range(0, self.height, 60):
            cv2.line(frame, (0, y), (self.width, y), (50, 55, 60), 1)
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (50, 55, 60), 1)

        # Banner indicating synthetic mode
        cv2.rectangle(frame, (0, 0), (self.width, 32), (0, 0, 150), -1)
        cv2.putText(frame, "SYNTHETIC TEST CAMERA (No physical webcam attached)",
                    (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        time.sleep(1.0 / self.fps)
        return True, frame

    def release(self):
        pass


class CameraStream:
    """
    Zero-latency, threaded webcam capture optimized for Raspberry Pi.
    Uses V4L2 single-frame buffer and hardware MJPEG decompression.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
        buffer_size: int = 1,
        fallback_synthetic: bool = True
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.buffer_size = buffer_size
        self.fallback_synthetic = fallback_synthetic

        self.cap = None
        self.is_synthetic = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.grabbed = False

        self._init_camera()

    def _init_camera(self):
        """Initializes USB webcam or V4L2 device with buffer clamping."""
        logger.info(f"Opening webcam source: {self.source} ({self.width}x{self.height} @ {self.fps}fps, format={self.fourcc})")
        
        is_device_index = False
        try:
            dev_idx = int(self.source)
            is_device_index = True
        except (ValueError, TypeError):
            dev_idx = None

        if is_device_index:
            # On Linux/RPi, V4L2 backend gives direct hardware access
            self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(dev_idx)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if self.cap.isOpened():
            # 1. OPTIMIZATION: Set buffer size to 1 to eliminate pipeline queue latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # 2. OPTIMIZATION: Hardware MJPEG decoding offloads CPU on USB webcams
            if len(self.fourcc) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify initial frame capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                self.grabbed = True
                self.latest_frame = test_frame
                actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"✓ Hardware webcam initialized: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS (buffer={self.buffer_size})")
                return

        # Fallback if hardware webcam is not connected
        logger.warning(f"Could not open hardware webcam: {self.source}")
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.fallback_synthetic:
            logger.info("Engaging Synthetic Webcam simulation fallback.")
            self.cap = SyntheticCamera(width=self.width, height=self.height, fps=self.fps)
            self.is_synthetic = True
            self.grabbed = True
            _, self.latest_frame = self.cap.read()
        else:
            raise RuntimeError(f"Failed to open webcam source: {self.source}")

    def start(self):
        """Starts asynchronous frame reading thread."""
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True, name="WebcamWorker")
        self.thread.start()
        logger.info("Webcam grabber thread started.")
        return self

    def _update_loop(self):
        """Continuously grabs newest frames, discarding stale buffered frames."""
        while self.running:
            if self.cap is None:
                time.sleep(0.05)
                continue

            # Grabbing immediately drops any stale queued frames in hardware buffer
            if not self.is_synthetic:
                grabbed = self.cap.grab()
                if grabbed:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.grabbed = True
                else:
                    time.sleep(0.005)
            else:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.grabbed = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the latest grabbed frame with zero latency."""
        with self.lock:
            if not self.grabbed or self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Webcam capture stopped.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
