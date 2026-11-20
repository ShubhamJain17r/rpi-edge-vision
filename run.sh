#!/usr/bin/env bash
set -e

IMAGE_TAG="${1:-rpi-edge-vision:latest}"
CONTAINER_NAME="rpi_vision_app"

echo "=========================================================="
echo " Launching Edge AI Vision & Distance Container"
echo " Target Image: ${IMAGE_TAG}"
echo "=========================================================="

# Ensure local directories exist
mkdir -p data models config weights

# Check if hardware camera is available
DEVICE_FLAG=""
if [ -e "/dev/video0" ]; then
    echo "✓ Found hardware camera at /dev/video0"
    DEVICE_FLAG="--device /dev/video0:/dev/video0"
else
    echo "! No /dev/video0 detected."
    echo "! Pipeline will automatically engage Synthetic Test Camera for simulation."
fi

# Stop existing container if running
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping existing container ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Determine IP address for web stream
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo "Starting container ${CONTAINER_NAME}..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --privileged \
    -p 8080:8080 \
    ${DEVICE_FLAG} \
    -v "$(pwd)/config:/app/config:ro" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    "${IMAGE_TAG}"

echo "=========================================================="
echo " Container launched successfully!"
echo " Web Stream Dashboard: http://${IP_ADDR}:8080"
echo " Live Telemetry API:  http://${IP_ADDR}:8080/api/telemetry"
echo " View logs:           docker logs -f ${CONTAINER_NAME}"
echo " Stop container:      docker stop ${CONTAINER_NAME}"
echo "=========================================================="
