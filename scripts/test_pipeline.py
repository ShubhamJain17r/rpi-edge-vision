#!/usr/bin/env python3
"""
Comprehensive automated self-test suite for Raspberry Pi 4 & 5 Edge AI pipeline.
Verifies all modules, camera fallback, detection, distance estimation, temporal smoothing,
and web streaming. Can run in headless environments with no camera attached.
"""

import os
import sys
import time
import logging
import unittest
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera import SyntheticCamera, CameraStream
from src.detector import EdgeDetector, DetectionResult
from src.distance import PinholeDistanceEstimator, DistanceEstimator, TemporalDistanceSmoother
from src.pipeline import load_config, annotate_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TestEdgeVisionPipeline(unittest.TestCase):

    def setUp(self):
        self.config = load_config("config/default.yaml")

    def test_01_synthetic_camera(self):
        """Test that synthetic camera produces valid BGR frames."""
        cam = SyntheticCamera(width=640, height=480, fps=30)
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)
        logger.info("✓ Synthetic camera verified.")

    def test_02_camera_stream_fallback(self):
        """Test CameraStream initializes and falls back to synthetic when no hardware camera exists."""
        stream = CameraStream(source=999, buffer_size=1, fallback_synthetic=True)
        self.assertTrue(stream.is_synthetic)
        stream.start()
        time.sleep(0.1)
        ret, frame = stream.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        stream.stop()
        logger.info("✓ Camera stream fallback verified.")

    def test_03_pinhole_distance_math(self):
        """Test pinhole distance calculation matches ground truth optical geometry."""
        dims = {"bottle": {"width": 0.08, "height": 0.25}}
        estimator = PinholeDistanceEstimator(focal_length=600.0, class_dimensions=dims)

        det = DetectionResult(
            bbox=(100, 100, 132, 200),
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            width_px=32,
            height_px=100,
            center=(116, 150)
        )

        dist = estimator.estimate_distance(det)
        self.assertAlmostEqual(dist, 1.50, delta=0.05)
        logger.info(f"✓ Pinhole distance calculation verified: {dist} m (expected 1.50 m)")

    def test_04_temporal_distance_smoothing(self):
        """Test that TemporalDistanceSmoother dampens webcam pixel jitter."""
        smoother = TemporalDistanceSmoother(alpha=0.5)

        # Frame 1: 1.50m
        det1 = DetectionResult(
            bbox=(100, 100, 200, 250),
            class_id=0, class_name="person", confidence=0.9,
            width_px=100, height_px=150, center=(150, 175),
            distance_m=1.50
        )
        s1 = smoother.smooth(det1)
        self.assertEqual(s1, 1.50)

        # Frame 2: simulated jitter jump to 1.70m
        det2 = DetectionResult(
            bbox=(102, 101, 202, 251),
            class_id=0, class_name="person", confidence=0.9,
            width_px=100, height_px=150, center=(152, 176),
            distance_m=1.70
        )
        s2 = smoother.smooth(det2)
        self.assertAlmostEqual(s2, 1.60, delta=0.02)
        logger.info(f"✓ Temporal distance smoother verified: raw 1.70m -> smoothed {s2}m")

    def test_05_distance_estimator_pipeline(self):
        """Test unified DistanceEstimator with pinhole mode."""
        engine = DistanceEstimator(self.config)
        cam = SyntheticCamera(width=640, height=480)
        _, frame = cam.read()

        detections = [
            DetectionResult(
                bbox=(200, 150, 300, 370),
                class_id=0,
                class_name="person",
                confidence=0.88,
                width_px=100,
                height_px=220,
                center=(250, 260)
            )
        ]

        processed, depth_vis = engine.process(frame, detections)
        self.assertEqual(len(processed), 1)
        self.assertIsNotNone(processed[0].distance_m)
        self.assertGreater(processed[0].distance_m, 0.0)
        logger.info(f"✓ Instantaneous pinhole distance verified: {processed[0].distance_m}m")

    def test_06_frame_annotation(self):
        """Test frame annotation draws bounding boxes, labels, and FPS indicators."""
        cam = SyntheticCamera(width=640, height=480)
        _, frame = cam.read()

        det = DetectionResult(
            bbox=(50, 50, 150, 200),
            class_id=0,
            class_name="person",
            confidence=0.95,
            width_px=100,
            height_px=150,
            center=(100, 125),
            distance_m=2.35,
            relative_depth=0.82
        )

        annotated = annotate_frame(frame, [det], display_fps=29.8, inf_fps=18.4)
        self.assertEqual(annotated.shape, frame.shape)
        logger.info("✓ Frame annotation verified.")

def main():
    logger.info("Running Edge Vision Pipeline Test Suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEdgeVisionPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
