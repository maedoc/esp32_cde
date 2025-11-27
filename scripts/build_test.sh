#!/bin/bash

# ESP32 CDE Quick Build Test (Updated for MAF)
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Validate CMakeLists.txt files
echo "Validating CMake configuration..."
if [ ! -f "CMakeLists.txt" ]; then echo "Error: Root CMakeLists.txt not found"; exit 1; fi
if [ ! -f "main/CMakeLists.txt" ]; then echo "Error: main/CMakeLists.txt not found"; exit 1; fi
if [ ! -f "components/esp32_cde/CMakeLists.txt" ]; then echo "Error: components/esp32_cde/CMakeLists.txt not found"; exit 1; fi
echo "CMake files validated successfully"

# Check source files
echo "Checking source files..."
REQUIRED_FILES=(
    "main/esp32_cde.c"
    "components/esp32_cde/include/maf.h"
    "components/esp32_cde/src/maf.c"
    "examples/maf_test_gradients.c"
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
set -e

echo "Testing compilation of source files..."

# Compile MAF library (Standalone)
gcc -c -I components/esp32_cde/include \
    components/esp32_cde/src/maf.c \
    -o /tmp/maf.o -lm

if [ $? -eq 0 ]; then
    echo "✓ MAF module compiles successfully"
else
    echo "✗ MAF module compilation failed"
    exit 1
fi

# Compile Gradient Test
echo "Compiling and running gradient test..."
gcc -I components/esp32_cde/include -o test_grad examples/maf_test_gradients.c components/esp32_cde/src/maf.c -lm
./test_grad
rm test_grad

echo "Basic compilation tests passed!"
EOF

chmod +x test_compile.sh

echo "Build test script completed."
