# syntax=docker/dockerfile:1
# Multi-arch Edge AI Dockerfile for Raspberry Pi 4 & Raspberry Pi 5 (ARM64 / linux/arm64)
# Also compatible with linux/amd64 for development and testing.

FROM python:3.11-slim-bookworm

LABEL maintainer="Edge AI Lab"
LABEL description="Edge AI Vision & Distance Calculation container for Raspberry Pi 4 & 5 (CPU-Optimized)"

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Install hardware video drivers, OpenMP runtime, Qt/X11 libraries, and essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libice6 \
    libxext6 \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch & Torchvision first, completely preventing CUDA/NVIDIA bloat
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy weights, configs, source code and scripts
COPY weights/ ./weights/
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/

# Verify or download base weights for 100% offline local edge operation
RUN python3 scripts/download_weights.py

# Expose web video feed and telemetry dashboard port

# Health check verifies the pipeline process is actively running
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD pgrep -f "python3 -m src.pipeline" || exit 1

# Default execution starts the edge vision and distance estimation pipeline
ENTRYPOINT ["python3", "-m", "src.pipeline"]
CMD ["--config", "config/default.yaml"]
