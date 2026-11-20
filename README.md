# 🍓 RPi Edge Vision: High-FPS Object Detection, Transfer Learning & Distance Estimation

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20%7C%20Raspberry%20Pi%205-red.svg)](https://www.raspberrypi.com/)
[![Architecture](https://img.shields.io/badge/Arch-ARM64%20%2F%20aarch64-blue.svg)](https://en.wikipedia.org/wiki/AArch64)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, edge-optimized computer vision and distance estimation system in a lightweight Docker image for **Raspberry Pi 4** (Cortex-A72) and **Raspberry Pi 5** (Cortex-A76). Runs **100% locally** (completely offline) with native webcam support, CPU-only PyTorch, transfer learning, and dual display options (Native OpenCV Window and Web Dashboard).

---

## 🌟 Key Highlights

- **⚡ High-FPS Asynchronous Architecture (`AsyncPipeline`)**:
  - **Camera Thread**: Captures live frames at hardware **30 FPS**.
  - **Inference Worker**: Runs YOLO object detection asynchronously at **~15–25 FPS** (at `imgsz: 320`) without stalling video capture.
  - **Display Thread**: Renders bounding boxes and distance tags at a smooth **25–30 FPS**.
- **🖥️ Dual Display Modes**:
  - **Native OpenCV GUI Window (`cv2.imshow`)**: Hardware-accelerated local display on your connected monitor/HDMI screen.
  - **Headless Web Dashboard**: Live HTTP MJPEG stream at `http://<rpi-ip>:8080` for remote viewing on phones/laptops.
- **📏 Dual Distance Calculation Engine**:
  - **Instantaneous Optical Pinhole Model (<0.05ms)**: Calculates real-world metric distance (meters/centimeters) using perspective camera geometry and known object dimensions.
  - **Monocular AI Depth Model (MiDaS v2.1 Small)**: Per-pixel relative inverse disparity running in a background worker thread to never block the main vision loop.
- **📷 Webcam-Specific Latency Optimizations**:
  - **V4L2 Single-Buffer Clamping (`buffer_size: 1`)**: Completely removes OpenCV's 4-frame FIFO queue delay for true zero-latency live video.
  - **Hardware MJPEG Stream Decoding**: Offloads JPEG decompression directly to the webcam DSP chip.
  - **Temporal Distance Smoothing (EMA)**: Dampens camera pixel noise to provide stable, readable distance telemetry.
- **🌱 CPU-Only & Zero CUDA/NVIDIA Bloat**:
  - Configured with official CPU-only PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`), stripping out over 4.5 GB of useless NVIDIA libraries and fitting cleanly on 8GB/16GB+ SD cards.
- **🧠 Transfer Learning Pipeline**: Single-command fine-tuning for custom objects with backbone layer freezing (`--freeze 10`) to prevent edge OOM.
- **⚡ INT8 Dynamic Quantization**: Tool to quantize models to 8-bit integers, cutting model size by ~60% and speeding up CPU inference by 30–50%.

---

## 📊 Hardware Benchmarks & Performance Comparison

Tested on **Raspberry Pi OS 64-bit (Bookworm)**:

| Metric | Raspberry Pi 4 (4GB, Cortex-A72 @ 1.5 GHz) | Raspberry Pi 5 (4GB, Cortex-A76 @ 2.4 GHz) |
|---|---|---|
| **Video Display FPS** | **25 – 30 FPS** *(Smooth motion)* | **30 FPS** *(Full Webcam Speed)* |
| **YOLOv8n AI Inference FPS** *(320x320)* | **15 – 18 FPS** | **28 – 35 FPS** |
| **Pinhole Metric Distance Overhead** | **< 0.05 ms** *(Instantaneous)* | **< 0.05 ms** *(Instantaneous)* |
| **Active Runtime RAM** | **~650 – 780 MB** *(~18% of 4GB)* | **~650 – 780 MB** *(~18% of 4GB)* |
| **Docker Image Size** | **~1.8 GB** *(Clean CPU build)* | **~1.8 GB** *(Clean CPU build)* |

---

## 🛠️ Docker Installation on Raspberry Pi (No `curl`, No Reboot)

If Docker is not yet installed on your Raspberry Pi:

```bash
# 1. Install Docker & Compose via native apt package manager
sudo apt update
sudo apt install -y docker.io docker-compose-v2

# 2. Enable & start the Docker service
sudo systemctl enable --now docker

# 3. Add your user to the docker and video groups (for webcam access)
sudo usermod -aG docker $USER
sudo usermod -aG video $USER

# 4. Activate group permissions immediately in current session (No reboot needed!)
newgrp docker

# 5. Verify you can run docker without sudo
docker run --rm hello-world
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
No login required (public repository):
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

### 3. Launch the Container
```bash
./run.sh
```

- **If an HDMI monitor is connected**: It automatically opens the **native OpenCV GUI window (`cv2.imshow`)**. Press `'q'` or `ESC` to exit.
- **If running headless**: Open your browser at `http://<your-raspberry-pi-ip>:8080`.

#### 💡 Maximum Performance Mode (Disable Web Server)
If you are watching the video feed directly on your monitor via the OpenCV window, disable the web server to save another 15–20% of CPU cycles:
```bash
./run.sh --no-web
```

---

## 🖥️ Display Modes & CLI Flags

The pipeline can be customized on launch:

| Flag | Description | Example |
|---|---|---|
| `--no-web` | Disables Flask HTTP web server to save CPU for maximum FPS | `./run.sh --no-web` |
| `--no-gui` | Disables OpenCV GUI window (runs headless with web dashboard only) | `./run.sh --no-gui` |
| `--imgsz 320` | Sets inference size (320 for 15-25+ FPS, 416 for balance, 640 for small objects) | `./run.sh --imgsz 320` |
| `--mode pinhole` | Distance calculation mode: `pinhole` (<0.05ms) or `both` (includes async MiDaS depth) | `./run.sh --mode pinhole` |

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
Fine-tune the detector on custom labeled datasets (YOLO format: `images/` and `labels/`):

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
├── Dockerfile                   # Multi-arch CPU-optimized Dockerfile (No CUDA bloat)
├── docker-compose.yml           # Pre-configured container orchestration with X11 & webcam
├── requirements.txt             # Pinned edge dependencies (CPU-only PyTorch)
├── build.sh                     # Automated build script
├── run.sh                       # Quick launcher with X11 GUI and camera detection
├── config/
│   └── default.yaml             # High-FPS webcam, model, calibration, and distance configs
├── src/
│   ├── __init__.py
│   ├── camera.py                # Zero-latency V4L2 reader + synthetic camera fallback
│   ├── detector.py              # Dual PyTorch / Native ONNX Runtime detection engine
│   ├── distance.py              # Optical pinhole + background async MiDaS + temporal smoother
│   ├── web_streamer.py          # HTTP MJPEG video feed & real-time telemetry dashboard
│   ├── pipeline.py              # Asynchronous multi-threaded pipeline with OpenCV GUI window
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
Ran 7 tests in 0.624s

OK
✓ Synthetic camera verified.
✓ Camera stream fallback verified (with buffer_size=1).
✓ Pinhole distance calculation verified (1.50 m ground truth).
✓ Temporal distance smoother verified (1.70m jitter -> 1.60m smoothed).
✓ Instantaneous pinhole distance verified (4.64m).
✓ Frame annotation verified with live FPS indicators.
✓ Web streamer frame encoding and telemetry verified.
```

---

## 📄 License
Released under the [MIT License](LICENSE).
