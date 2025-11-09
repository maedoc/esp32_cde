#!/bin/bash
# Component validation script

set -e

echo "=== CAN NVS Component Validation ==="
echo ""

# Check required files
echo "Checking required files..."
required_files=(
    "CMakeLists.txt"
    "README.md"
    "include/can_nvs.h"
    "src/can_nvs.c"
    "test/main/test_can_nvs.c"
    "test/CMakeLists.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        exit 1
    fi
done

echo ""
echo "Checking code structure..."

# Check for header guards
if grep -q "#ifndef CAN_NVS_H" include/can_nvs.h && \
   grep -q "#define CAN_NVS_H" include/can_nvs.h && \
   grep -q "#endif" include/can_nvs.h; then
    echo "  ✓ Header guards present"
else
    echo "  ✗ Header guards missing or incorrect"
    exit 1
fi

# Check for required API functions
api_functions=(
    "can_nvs_init"
    "can_nvs_deinit"
    "can_nvs_store_sequence"
    "can_nvs_load_sequence"
    "can_nvs_delete_sequence"
    "can_nvs_free_sequence"
)

for func in "${api_functions[@]}"; do
    if grep -q "$func" include/can_nvs.h && grep -q "$func" src/can_nvs.c; then
        echo "  ✓ Function $func declared and defined"
    else
        echo "  ✗ Function $func missing"
        exit 1
    fi
done

echo ""
echo "Checking test coverage..."

# Check for test functions
test_functions=(
    "test_can_nvs_init_deinit"
    "test_can_nvs_store_load_single_frame"
    "test_can_nvs_store_load_multiple_frames"
    "test_can_nvs_extended_id"
    "test_can_nvs_delete_sequence"
)

for func in "${test_functions[@]}"; do
    if grep -q "$func" test/main/test_can_nvs.c; then
        echo "  ✓ Test $func present"
    else
        echo "  ✗ Test $func missing"
        exit 1
    fi
done

echo ""
echo "Checking documentation..."

# Check README sections
readme_sections=(
    "Features"
    "Installation"
    "Quick Start"
    "API Reference"
    "Testing"
)

for section in "${readme_sections[@]}"; do
    if grep -q "## $section" README.md; then
        echo "  ✓ README section: $section"
    else
        echo "  ✗ README section missing: $section"
        exit 1
    fi
done

echo ""
echo "=== All checks passed! ==="
echo ""
echo "Component statistics:"
echo "  Header lines:   $(wc -l < include/can_nvs.h)"
echo "  Source lines:   $(wc -l < src/can_nvs.c)"
echo "  Test lines:     $(wc -l < test/main/test_can_nvs.c)"
echo "  README lines:   $(wc -l < README.md)"
echo ""
echo "The component is ready for use!"
