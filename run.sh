#!/usr/bin/env bash
set -e

CONTAINER_NAME="rpi_vision_app"
IMAGE_TAG="rpi-edge-vision:latest"

echo "=========================================================="
echo " Launching Full-Screen Edge Vision & Distance"
echo " Image: ${IMAGE_TAG}"
echo "=========================================================="

mkdir -p data models config weights

# 1. Webcam detection
DEVICE_FLAG=""
if [ -e "/dev/video0" ]; then
    echo "✓ Found hardware webcam at /dev/video0"
    DEVICE_FLAG="--device /dev/video0:/dev/video0"
else
    echo "! No /dev/video0 detected (Synthetic camera fallback active)."
fi

# 2. X11 Display detection for native full-screen OpenCV window (cv2.imshow)
GUI_FLAG=""
if [ -n "$DISPLAY" ]; then
    echo "✓ Found X11 Display ($DISPLAY)"
    xhost +local:root >/dev/null 2>&1 || true
    GUI_FLAG="-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:ro"
else
    echo "! No X11 DISPLAY detected (Running in headless terminal mode)."
fi

# 3. Clean up existing container if running
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "=========================================================="
echo " Opening Full-Screen Video Window on screen..."
echo " Hotkeys:"
echo "   [ESC] or [q] : Cleanly exit and close application"
echo "   [f]          : Toggle between Full-Screen and Windowed"
echo "=========================================================="

exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    --privileged \
    ${DEVICE_FLAG} \
    ${GUI_FLAG} \
    -v "$(pwd)/src:/app/src:ro" \
    -v "$(pwd)/config:/app/config:ro" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    "${IMAGE_TAG}" "$@"
