#!/usr/bin/env bash
set -e

IMAGE_TAG="${1:-rpi-edge-vision:latest}"
ARCH=$(uname -m)

echo "=========================================================="
echo " Building Edge AI Vision Image for RPi 4 & RPi 5"
echo " Host Architecture: ${ARCH}"
echo " Image Tag:         ${IMAGE_TAG}"
echo "=========================================================="

if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    echo "-> Detected native ARM64 host (Raspberry Pi 4 / 5)."
    echo "-> Running native Docker build..."
    docker build -t "${IMAGE_TAG}" .
else
    echo "-> Running on x86_64 host."
    echo "-> To build for local development/testing on this PC:"
    echo "   docker build -t ${IMAGE_TAG} ."
    echo "-> To cross-build for deployment to RPi 4 / RPi 5 (ARM64):"
    echo "   docker buildx build --platform linux/arm64 -t ${IMAGE_TAG} --load ."
    echo ""
    read -p "Build for current machine [1] or cross-build for RPi ARM64 [2]? (default: 1): " choice
    if [[ "$choice" == "2" ]]; then
        docker buildx build --platform linux/arm64 -t "${IMAGE_TAG}" --load .
    else
        docker build -t "${IMAGE_TAG}" .
    fi
fi

echo "=========================================================="
echo " Build completed successfully: ${IMAGE_TAG}"
echo "=========================================================="
