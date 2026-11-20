"""
Lightweight HTTP MJPEG video streamer and real-time dashboard for Raspberry Pi.
Allows viewing live detection and distance calculation from any browser on local network.
"""

import time
import json
import logging
import threading
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from flask import Flask, Response, render_template_string, jsonify

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raspberry Pi Edge Vision & Distance Dashboard</title>
    <style>
        :root {
            --bg-color: #121418;
            --card-bg: #1c2026;
            --accent: #00d26a;
            --accent-blue: #0099ff;
            --text-main: #f0f3f6;
            --text-dim: #9aa5b1;
            --border: #2e3642;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 20px;
        }
        h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
        .badge {
            background-color: #273142;
            color: var(--accent);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
        .video-card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 16px;
            border: 1px solid var(--border);
            text-align: center;
        }
        .video-card img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #000;
            border: 1px solid #333;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 10px;
            padding: 16px;
            border: 1px solid var(--border);
        }
        .card h2 {
            margin-top: 0;
            font-size: 1.1rem;
            color: var(--text-dim);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .stat-item {
            background: #15191e;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-val { font-size: 1.5rem; font-weight: bold; color: var(--accent); }
        .stat-label { font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 8px;
        }
        th, td {
            text-align: left;
            padding: 8px 6px;
            border-bottom: 1px solid var(--border);
        }
        th { color: var(--text-dim); font-weight: 500; }
        .dist-badge {
            font-weight: 700;
            color: #ffb703;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🍓 RPi-4 / RPi-5 Edge Vision & Distance Dashboard</h1>
        <div class="badge">● LIVE LOCAL INFERENCE</div>
    </div>

    <div class="container">
        <div class="video-card">
            <img src="/video_feed" alt="Live Camera Video Stream">
            <p style="color: var(--text-dim); font-size: 0.85rem; margin-top: 10px;">
                Direct V4L2 Hardware Stream with real-time detection boxes, classification, and metric distance estimation.
            </p>
        </div>

        <div class="sidebar">
            <div class="card">
                <h2>Performance Stats</h2>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-val" id="fps-val">--</div>
                        <div class="stat-label">Inference FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val" id="obj-count">0</div>
                        <div class="stat-label">Objects Detected</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Detected Objects & Distance</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Conf</th>
                            <th>Distance</th>
                            <th>Depth</th>
                        </tr>
                    </thead>
                    <tbody id="telemetry-table">
                        <tr><td colspan="4" style="text-align: center; color: var(--text-dim);">Awaiting frame detections...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (!res.ok) return;
                const data = await res.json();
                document.getElementById('fps-val').innerText = data.fps.toFixed(1);
                document.getElementById('obj-count').innerText = data.detections.length;

                const tbody = document.getElementById('telemetry-table');
                if (data.detections.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#9aa5b1;">No objects in view</td></tr>';
                    return;
                }

                let rows = '';
                data.detections.forEach(d => {
                    const distStr = d.distance_m ? `${d.distance_m.toFixed(2)} m` : '--';
                    const depthStr = d.relative_depth !== null ? `${d.relative_depth}` : '--';
                    rows += `<tr>
                        <td><strong>${d.class_name}</strong></td>
                        <td>${(d.confidence * 100).toFixed(0)}%</td>
                        <td class="dist-badge">${distStr}</td>
                        <td style="color:#0099ff;">${depthStr}</td>
                    </tr>`;
                });
                tbody.innerHTML = rows;
            } catch (err) {
                console.error("Telemetry fetch error:", err);
            }
        }
        setInterval(fetchTelemetry, 350);
    </script>
</body>
</html>
"""


class WebStreamer:
    """Handles HTTP MJPEG video feed and telemetry REST endpoints."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, quality: int = 80):
        self.host = host
        self.port = port
        self.quality = quality

        self.latest_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()
        self.telemetry_data: Dict[str, Any] = {
            "fps": 0.0,
            "timestamp": time.time(),
            "detections": []
        }

        self.app = Flask(__name__)
        self._setup_routes()
        self.server_thread = None

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_mjpeg(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.route("/api/telemetry")
        def api_telemetry():
            with self.lock:
                return jsonify(self.telemetry_data)

    def _generate_mjpeg(self):
        while True:
            with self.lock:
                frame_bytes = self.latest_jpeg
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)

    def update_frame(self, frame: np.ndarray, fps: float, detections: List[Any]):
        """Encodes annotated frame to JPEG and stores telemetry, safely converting numpy types."""
        if frame is None:
            return

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_param)
        if ret:
            with self.lock:
                self.latest_jpeg = buffer.tobytes()
                self.telemetry_data = {
                    "fps": float(round(fps, 1)),
                    "timestamp": float(time.time()),
                    "detections": [
                        {
                            "class_name": str(d.class_name),
                            "confidence": float(round(d.confidence, 2)),
                            "distance_m": float(d.distance_m) if d.distance_m is not None else None,
                            "relative_depth": float(d.relative_depth) if d.relative_depth is not None else None,
                            "bbox": [int(coord) for coord in d.bbox]
                        }
                        for d in detections
                    ]
                }

    def start(self):
        """Starts Flask server in background daemon thread."""
        logger.info(f"Starting Web Streamer on http://{self.host}:{self.port}")
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True),
            daemon=True,
            name="WebStreamerWorker"
        )
        self.server_thread.start()
