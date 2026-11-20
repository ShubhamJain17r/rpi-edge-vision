# 🍓 RPi Edge Vision: Object Detection, Transfer Learning & Distance Estimation

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20%7C%20Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)
[![Architecture](https://img.shields.io/badge/Arch-ARM64%20%2F%20aarch64-blue.svg)](https://en.wikipedia.org/wiki/AArch64)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An edge-optimized computer vision and distance calculation system packaged in a lightweight Docker image for **Raspberry Pi 4** (Cortex-A72) and **Raspberry Pi 5** (Cortex-A76). Runs **100% locally** (completely offline) with native webcam support, transfer learning, and headless browser telemetry.

---

## 🌟 Key Highlights

- **Edge-Optimized Detection & Classification**: Powered by Ultralytics YOLOv8/11 and ONNX Runtime with ARM NEON SIMD acceleration.
- **Transfer Learning Pipeline**: Single-command fine-tuning for custom objects and classes with backbone freezing (`--freeze 10`) to prevent edge OOM and reduce CPU load.
- **Dual Distance Calculation Engine**:
  - **Optical Pinhole Model**: Calculates real-world metric distance (meters/centimeters) using perspective camera geometry and known object dimensions.
  - **Monocular AI Depth Model (MiDaS v2.1 Small)**: Per-pixel inverse disparity mapping via multi-threaded ONNX Runtime CPU.
- **Webcam-Specific Enhancements**:
  - **V4L2 Single-Buffer Clamping (`buffer_size: 1`)**: Completely removes OpenCV's 4-frame queue lag for true zero-latency live video.
  - **Hardware MJPEG Decoding**: Leverages webcam hardware decompression, drastically reducing USB bus utilization.
  - **Temporal Distance Smoothing (EMA)**: Dampens camera sensor pixel jitter to provide stable, readable distance telemetry.
  - **Decoupled Depth Cadence (`depth_interval: 3`)**: Sub-samples the heavier AI depth model while running detection and metric distance at full 30 FPS.
- **INT8 Dynamic Quantization**: Tool to quantize models to 8-bit integers, reducing model size by ~60% and speeding up CPU inference by 30–50%.
- **Zero-Display Headless Web Dashboard**: Built-in HTTP MJPEG video stream and telemetry dashboard at `http://<rpi-ip>:8080`.
- **Fault-Tolerant Simulation Fallback**: Automatically engages a synthetic test camera if no physical webcam is plugged in.

---

## 📊 Hardware Benchmarks & Resource Profile

Tested on **Raspberry Pi OS 64-bit (Bookworm)**:

| Metric | Raspberry Pi 4 (4GB, Cortex-A72 @ 1.5 GHz) | Raspberry Pi 5 (4GB, Cortex-A76 @ 2.4 GHz) |
|---|---|---|
| **RAM Footprint (Runtime)** | **~650 – 800 MB** *(~18% of 4GB)* | **~650 – 800 MB** *(~18% of 4GB)* |
| **YOLOv8n + Optical Pinhole** *(640x640)* | **8 – 11 FPS** | **20 – 25 FPS** *(Real-Time)* |
| **YOLOv8n + Optical Pinhole** *(416x416)* | **14 – 18 FPS** | **30 – 35 FPS** |
| **YOLOv8n + MiDaS (Decoupled N=3)** | **12 – 15 FPS** | **22 – 26 FPS** |
| **Image Classification (`yolov8n-cls`)** | **22 – 28 FPS** | **50+ FPS** |
| **Docker Image Size** | **~1.9 GB** | **~1.9 GB** |

---

## 🛠️ Docker Installation & Permissions on Raspberry Pi (No `curl`, No Reboot)

If Docker is not yet installed on your Raspberry Pi, install it using the native `apt` package manager without downloading external scripts via `curl` and without restarting your system:

### 1. Install Docker & Docker Compose via `apt`
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
```

### 2. Enable & Start the Docker Service
```bash
sudo systemctl enable --now docker
```

### 3. Grant Non-Root Docker & Webcam Access
Add your current user (`$USER` or `pi`) to both the `docker` and `video` groups:
```bash
sudo usermod -aG docker $USER
sudo usermod -aG video $USER
```

### 4. Activate Permissions Immediately (No Reboot Required!)
Normally, group membership changes require logging out or rebooting. You can bypass this and activate the permissions in your current terminal session instantly with:
```bash
newgrp docker
```

### 5. Verify Installation (Without `sudo`)
```bash
docker run --rm hello-world
```
*(If this prints "Hello from Docker!", Docker is ready and fully functional without requiring `sudo` or a system reboot).*

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ShubhamJain17r/rpi-edge-vision.git
cd rpi-edge-vision
```

### 2. Build the Docker Image
```bash
chmod +x build.sh run.sh
./build.sh
```
*(Base weights `yolov8n.pt`, `yolov8n-cls.pt`, and `midas_v21_small.onnx` are downloaded during build for 100% offline edge deployment).*

### 3. Run the Container
```bash
./run.sh
```
Or with Docker Compose:
```bash
docker compose up -d rpi-vision
```

### 4. Open the Web Dashboard
Open any browser on your laptop or mobile phone connected to the same local network:
```text
http://<your-raspberry-pi-ip>:8080
```
- Real-time video with color-coded bounding boxes and metric distances.
- Picture-in-picture depth map preview.
- Live FPS, latency profiler, and detected objects telemetry table.
- REST API endpoint: `http://<your-raspberry-pi-ip>:8080/api/telemetry`

---

## 📷 Webcam Configuration & Permissions

### Standard USB Webcams
Standard USB webcams attach as `/dev/video0`. The container accesses this directly via Docker device mapping:
```yaml
devices:
  - "/dev/video0:/dev/video0"
privileged: true
group_add:
  - video
```

### Raspberry Pi Camera Modules (CSI / Ribbon Cable)
On Raspberry Pi OS **Bookworm**, CSI cameras use `libcamera`. To expose the camera as a standard V4L2 `/dev/video0` node:
```bash
# Enable V4L2 driver module
sudo modprobe bcm2835-v4l2

# Verify device node
v4l2-ctl --list-devices
```

---

## 📏 Distance Calculation & Calibration

### Mathematical Principle
$$d = \frac{W_{\text{real}} \times f}{w_{\text{pixel}}}$$

Pre-configured object dimensions in `config/default.yaml`:
- `person`: $1.70\text{ m} \times 0.50\text{ m}$
- `car`: $1.80\text{ m}$
- `bottle`: $0.25\text{ m}$
- `laptop`: $0.35\text{ m}$
- `cell phone`: $0.16\text{ m}$

### Calibrating Your Specific Webcam Focal Length ($f$)
To calibrate the exact focal length ($f$) in pixels for your webcam lens:
```bash
# Example: An A4 paper (0.21m width) at 0.50m distance measures 252 pixels:
python3 scripts/calibrate_camera.py --real-width 0.21 --distance 0.50 --pixel-width 252
```
Or launch interactive calibration with live ROI bounding box selection:
```bash
python3 scripts/calibrate_camera.py --interactive --real-width 0.21 --distance 0.50 --source 0
```
Then update `focal_length` in `config/default.yaml`.

---

## 🧠 Transfer Learning Guide

### 1. Object Detection Transfer Learning
Fine-tune the detector on your own annotated objects (YOLO format: `images/` and `labels/`):

```bash
docker compose run --rm train-detector \
  --data data/sample_dataset.yaml \
  --base-model weights/yolov8n.pt \
  --epochs 20 \
  --batch 8 \
  --freeze 10
```

- `--freeze 10`: Freezes the backbone layers and trains only the detection heads. This saves ~50% of CPU and RAM on the Pi.
- Automatically exports the fine-tuned model to `models/custom_detector.onnx` for accelerated edge inference.

### 2. Image Classification Transfer Learning
Fine-tune an image classifier on folders organized by class name:

```bash
docker compose run --rm train-classifier \
  --data data/my_classes/ \
  --base-model weights/yolov8n-cls.pt \
  --epochs 15 \
  --batch 16
```

---

## ⚡ Model Acceleration & INT8 Quantization

To convert your trained ONNX model to an 8-bit dynamically quantized model:
```bash
python3 scripts/quantize_model.py \
  --input weights/yolov8n.onnx \
  --output weights/yolov8n_int8.onnx
```
- Reduces model disk size from ~6.3 MB to ~2.6 MB.
- Lowers memory bandwidth by 50% and improves ARM NEON CPU inference speed by 30–50%.

---

## 📂 Project Architecture

```text
.
├── Dockerfile                   # Multi-arch ARM64/AMD64 Dockerfile
├── docker-compose.yml           # Multi-service container orchestration
├── requirements.txt             # Pinned edge dependencies
├── build.sh                     # Automated build script
├── run.sh                       # Quick launcher with camera detection
├── config/
│   └── default.yaml             # Webcam, model, calibration, and distance configs
├── src/
│   ├── __init__.py
│   ├── camera.py                # Zero-latency V4L2 reader + synthetic camera fallback
│   ├── detector.py              # Dual PyTorch / Native ONNX Runtime detection engine
│   ├── distance.py              # Optical pinhole + MiDaS depth + temporal EMA smoother
│   ├── web_streamer.py          # HTTP MJPEG video feed & real-time telemetry dashboard
│   ├── pipeline.py              # Master pipeline with decoupled depth cadence
│   ├── train_detector.py        # Object detection transfer learning script
│   └── train_classifier.py      # Classification transfer learning script
├── scripts/
│   ├── download_weights.py      # Pre-bakes weights during docker build
│   ├── calibrate_camera.py      # Camera focal length calibration utility
│   ├── quantize_model.py        # INT8 dynamic quantization tool
│   └── test_pipeline.py         # Automated self-test suite (7 tests)
├── data/                        # Datasets volume mount
├── models/                      # Trained custom models volume mount
└── weights/                     # Pre-trained base model weights
```

---

## 🧪 Testing & Verification

Run the comprehensive 7-test suite inside the container:
```bash
docker run --rm --entrypoint python3 rpi-edge-vision:latest scripts/test_pipeline.py
```

Expected output:
```text
Ran 7 tests in 0.571s

OK
✓ Synthetic camera verified.
✓ Camera stream fallback verified (with buffer_size=1).
✓ Pinhole distance calculation verified (1.50 m ground truth).
✓ Temporal distance smoother verified (1.70m jitter -> 1.60m smoothed).
✓ Decoupled cadence pipeline verified (distance=4.64m).
✓ Frame annotation verified with live latency profiling.
✓ Web streamer frame encoding and telemetry verified.
```

---

## 📄 License
Released under the [MIT License](LICENSE).
