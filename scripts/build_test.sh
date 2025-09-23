#!/bin/bash

# ESP32 CDE Quick Build Test
# This script tests the build process without requiring a full ESP-IDF installation

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Create a minimal sdkconfig for testing
cat > sdkconfig.defaults << EOF
# ESP32 CDE Configuration
CONFIG_ESPTOOLPY_FLASHSIZE_4MB=y
CONFIG_PARTITION_TABLE_SINGLE_APP=y
CONFIG_FREERTOS_HZ=1000

# CDE specific configuration
CONFIG_CDE_MAX_FEATURES=32
CONFIG_CDE_SAMPLE_BUFFER_SIZE=256
CONFIG_CDE_USE_FLOAT_PRECISION=y

# Logging configuration
CONFIG_LOG_DEFAULT_LEVEL_INFO=y
CONFIG_LOG_DEFAULT_LEVEL=3

# Memory configuration
CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192
EOF

echo "Created sdkconfig.defaults"

# Validate CMakeLists.txt files
echo "Validating CMake configuration..."

if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Root CMakeLists.txt not found"
    exit 1
fi

if [ ! -f "main/CMakeLists.txt" ]; then
    echo "Error: main/CMakeLists.txt not found"
    exit 1
fi

if [ ! -f "components/esp32_cde/CMakeLists.txt" ]; then
    echo "Error: components/esp32_cde/CMakeLists.txt not found"
    exit 1
fi

echo "CMake files validated successfully"

# Check source files
echo "Checking source files..."

REQUIRED_FILES=(
    "main/esp32_cde.c"
    "components/esp32_cde/include/esp32_cde.h"
    "components/esp32_cde/include/cde_internal.h"
    "components/esp32_cde/src/cde_core.c"
    "components/esp32_cde/src/cde_math.c"
    "components/esp32_cde/src/cde_buffer.c"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found"
        exit 1
    fi
    echo "✓ $file"
done

echo "All source files present"

# Create a test compilation script for validation
cat > test_compile.sh << 'EOF'
#!/bin/bash
# Simple compilation test (requires gcc)

echo "Testing compilation of source files..."

# Test compile the core files
gcc -c -I components/esp32_cde/include \
    -I test/mock_headers \
    -I /usr/include \
    -D ESP_PLATFORM \
    -D CONFIG_CDE_MAX_FEATURES=32 \
    -D CONFIG_CDE_SAMPLE_BUFFER_SIZE=256 \
    -D CONFIG_CDE_USE_FLOAT_PRECISION=1 \
    components/esp32_cde/src/cde_math.c \
    -o /tmp/cde_math.o

if [ $? -eq 0 ]; then
    echo "✓ Math module compiles successfully"
else
    echo "✗ Math module compilation failed"
    exit 1
fi

gcc -c -I components/esp32_cde/include \
    -I test/mock_headers \
    -I /usr/include \
    -D ESP_PLATFORM \
    -D CONFIG_CDE_MAX_FEATURES=32 \
    -D CONFIG_CDE_SAMPLE_BUFFER_SIZE=256 \
    -D CONFIG_CDE_USE_FLOAT_PRECISION=1 \
    components/esp32_cde/src/cde_buffer.c \
    -o /tmp/cde_buffer.o

if [ $? -eq 0 ]; then
    echo "✓ Buffer module compiles successfully"
else
    echo "✗ Buffer module compilation failed"
    exit 1
fi

gcc -c -I components/esp32_cde/include \
    -I test/mock_headers \
    -I /usr/include \
    -D ESP_PLATFORM \
    -D CONFIG_CDE_MAX_FEATURES=32 \
    -D CONFIG_CDE_SAMPLE_BUFFER_SIZE=256 \
    -D CONFIG_CDE_USE_FLOAT_PRECISION=1 \
    components/esp32_cde/src/cde_core.c \
    -o /tmp/cde_core.o

if [ $? -eq 0 ]; then
    echo "✓ Core module compiles successfully"
else
    echo "✗ Core module compilation failed"
    exit 1
fi

echo "Basic compilation test passed!"
EOF

chmod +x test_compile.sh

echo ""
echo "================================================================"
echo "ESP32 CDE Project Structure Created Successfully!"
echo "================================================================"
echo ""
echo "Project structure:"
echo "├── CMakeLists.txt                    (Root CMake file)"
echo "├── Kconfig.projbuild                 (Configuration options)"
echo "├── sdkconfig.defaults                (Default configuration)"
echo "├── main/"
echo "│   ├── CMakeLists.txt"
echo "│   └── esp32_cde.c                   (Main application)"
echo "├── components/esp32_cde/"
echo "│   ├── CMakeLists.txt"
echo "│   ├── include/"
echo "│   │   ├── esp32_cde.h               (Public API)"
echo "│   │   └── cde_internal.h            (Internal definitions)"
echo "│   └── src/"
echo "│       ├── cde_core.c                (Core CDE implementation)"
echo "│       ├── cde_math.c                (Mathematical utilities)"
echo "│       └── cde_buffer.c              (Buffer management)"
echo "├── scripts/"
echo "│   └── test_qemu.sh                  (QEMU test script)"
echo "└── test_compile.sh                   (Basic compilation test)"
echo ""
echo "To build with ESP-IDF:"
echo "  1. Install ESP-IDF: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/"
echo "  2. Source the environment: source \$IDF_PATH/export.sh"
echo "  3. Build: idf.py build"
echo "  4. Flash: idf.py flash monitor"
echo "  5. Test with QEMU: ./scripts/test_qemu.sh"
echo ""
echo "For basic validation (without ESP-IDF):"
echo "  ./test_compile.sh"
echo ""