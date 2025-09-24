#!/bin/bash

# ESP32 CDE Test Script
# This script builds and runs the ESP32 CDE project with QEMU emulator

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"

# Check if ESP-IDF is available
if [ -z "$IDF_PATH" ]; then
    echo "ESP-IDF not found. Attempting to source from common locations..."
    
    # Try common ESP-IDF installation paths
    IDF_PATHS=(
        "/opt/esp/idf"
        "/usr/local/esp/esp-idf"
        "$HOME/esp/esp-idf"
        "/tmp/esp-idf"
    )
    
    for path in "${IDF_PATHS[@]}"; do
        if [ -f "$path/export.sh" ]; then
            echo "Found ESP-IDF at: $path"
            export IDF_PATH="$path"
            source "$path/export.sh"
            break
        fi
    done
    
    if [ -z "$IDF_PATH" ]; then
        echo "Error: ESP-IDF not found. Please install ESP-IDF and source export.sh"
        echo "Download from: https://github.com/espressif/esp-idf"
        exit 1
    fi
fi

echo "Using ESP-IDF from: $IDF_PATH"

# Set target to ESP32
idf.py set-target esp32

# Clean and build the project
echo "Building ESP32 CDE project..."
idf.py clean
idf.py build

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi

# Check if QEMU is available
QEMU_ESP32="qemu-system-xtensa"
if ! command -v $QEMU_ESP32 &> /dev/null; then
    echo "Warning: QEMU ESP32 emulator not found. Attempting to use ESP-IDF QEMU..."
    QEMU_ESP32="$IDF_PATH/tools/qemu/esp32/bin/qemu-system-xtensa"
    
    if [ ! -f "$QEMU_ESP32" ]; then
        echo "Error: QEMU ESP32 emulator not available."
        echo "Please install QEMU with ESP32 support or use ESP-IDF's QEMU tools."
        echo "For now, we'll show you how to flash to a real device:"
        echo ""
        echo "To flash to a real ESP32 device:"
        echo "  idf.py flash monitor"
        echo ""
        echo "To use with QEMU (when available):"
        echo "  idf.py qemu monitor"
        exit 1
    fi
fi

# Run with QEMU
echo "Starting QEMU ESP32 emulator..."
echo "Note: Press Ctrl+C to exit QEMU"

# QEMU command for ESP32
$QEMU_ESP32 \
    -nographic \
    -machine esp32 \
    -drive file=build/flashimage.bin,if=mtd,format=raw \
    -serial mon:stdio \
    -S -s

# Alternative: Use ESP-IDF's built-in QEMU support if available
# idf.py qemu monitor