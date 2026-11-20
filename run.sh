#!/usr/bin/env bash
set -e

CONTAINER_NAME="rpi_vision_app"
IMAGE_TAG="rpi-edge-vision:latest"

echo "=========================================================="
echo " Launching High-FPS Edge Vision & Distance Container"
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

# 2. X11 Display detection for native OpenCV window (cv2.imshow)
GUI_FLAG=""
if [ -n "$DISPLAY" ]; then
    echo "✓ Found X11 Display ($DISPLAY)"
    xhost +local:root >/dev/null 2>&1 || true
    GUI_FLAG="-e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix:ro"
else
    echo "! No X11 DISPLAY detected (Running in headless mode)."
fi

# 3. Clean up existing container
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

# Check arguments
NO_WEB=false
HAS_GUI=false
for arg in "$@"; do
    if [ "$arg" == "--no-web" ]; then
        NO_WEB=true
    fi
    if [ "$arg" == "--gui" ]; then
        HAS_GUI=true
    fi
done

echo "Starting container ${CONTAINER_NAME}..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --privileged \
    -p 8080:8080 \
    ${DEVICE_FLAG} \
    ${GUI_FLAG} \
    -v "$(pwd)/src:/app/src:ro" \
    -v "$(pwd)/config:/app/config:ro" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    "${IMAGE_TAG}" "$@"

echo "=========================================================="
echo " Container launched successfully!"
if [ "$HAS_GUI" = true ]; then
    echo " Native GUI Window:    ACTIVE on desktop ('RPi Edge Vision & Distance')"
    echo "                       (Press 'q' or ESC in the window to quit)"
fi
if [ "$NO_WEB" = true ]; then
    echo " Mode:                 Maximum Performance (--no-web)"
    echo " Web Dashboard:        Disabled"
    if [ "$HAS_GUI" = false ]; then
        echo " Note:                 Running in terminal mode (no visual display)."
        echo "                       -> To view live desktop window: ./run.sh --no-web --gui"
        echo "                       -> To view in web browser:     ./run.sh"
    fi
    echo " Showing live telemetry output below (Press Ctrl+C to detach):"
    echo "=========================================================="
    docker logs -f "${CONTAINER_NAME}"
else
    echo " Web Dashboard:        http://localhost:8080 (or http://${IP_ADDR}:8080)"
    echo "                       👉 Open in any browser to watch live annotated video!"
    if [ "$HAS_GUI" = false ]; then
        echo " Native GUI Window:    To also pop up desktop window, pass --gui: ./run.sh --gui"
    fi
    echo " View live logs:       docker logs -f ${CONTAINER_NAME}"
    echo " Stop container:       docker stop ${CONTAINER_NAME}"
    echo "=========================================================="
fi
