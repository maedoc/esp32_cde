# ESP32 Conditional Density Estimation Component

This project provides a conditional density estimation (CDE) library for ESP32 microcontrollers, designed for probabilistic modeling and machine learning applications on embedded systems.

## Features

- **Kernel Density Estimation**: Implements Gaussian kernel-based conditional density estimation
- **Memory Efficient**: Circular buffer management for limited memory environments  
- **Configurable**: Adjustable parameters via ESP-IDF's configuration system
- **Real-time Capable**: Optimized for real-time density estimation
- **Test Coverage**: Includes QEMU emulator support for testing

## Project Structure

```
esp32_cde/
├── CMakeLists.txt                    # Root CMake configuration
├── Kconfig.projbuild                 # ESP-IDF configuration options
├── sdkconfig.defaults                # Default configuration values
├── main/                             # Main application
│   ├── CMakeLists.txt
│   └── esp32_cde.c                   # Demo application
├── components/esp32_cde/             # CDE component library
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── esp32_cde.h               # Public API header
│   │   └── cde_internal.h            # Internal definitions
│   └── src/
│       ├── cde_core.c                # Core CDE implementation
│       ├── cde_math.c                # Mathematical utilities
│       └── cde_buffer.c              # Buffer management
├── scripts/
│   ├── test_qemu.sh                  # QEMU emulator test script
│   └── build_test.sh                 # Build validation script
├── python/                           # Python training scripts
│   ├── cde_training.py               # MDN and MAF implementations
│   ├── test_cde_training.py          # Training script tests
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # Python scripts documentation
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD pipeline
└── test/
    └── mock_headers/                 # Mock headers for testing
```

## Quick Start

### Prerequisites

1. **ESP-IDF**: Install ESP-IDF v5.0 or later
   ```bash
   git clone --recursive https://github.com/espressif/esp-idf.git
   cd esp-idf
   ./install.sh esp32
   source export.sh
   ```

2. **QEMU** (optional, for emulation): Install QEMU with ESP32 support

### Building

1. Clone this repository:
   ```bash
   git clone https://github.com/maedoc/esp32_cde.git
   cd esp32_cde
   ```

2. Configure the project:
   ```bash
   idf.py menuconfig
   ```
   Navigate to "ESP32 CDE Configuration" to adjust parameters.

3. Build the project:
   ```bash
   idf.py build
   ```

### Running

#### On Real Hardware
```bash
idf.py flash monitor
```

#### With QEMU Emulator
```bash
./scripts/test_qemu.sh
```

### Basic Validation (without ESP-IDF)
```bash
./scripts/build_test.sh
./test_compile.sh
```

### Python Training Scripts
For advanced conditional density estimation training and validation:
```bash
cd python
pip install -r requirements.txt
python test_cde_training.py     # Run tests
python cde_training.py --test   # Quick integration test
python cde_training.py          # Full training with visualization
```

## API Reference

### Initialization
```c
#include "esp32_cde.h"

cde_config_t config = {
    .max_features = 16,
    .buffer_size = 128,
    .learning_rate = 0.01f,
    .use_float_precision = true
};

cde_handle_t cde = cde_init(&config);
```

### Adding Training Data
```c
float features[] = {1.0f, 2.0f, 3.0f};
float target = 5.0f;
esp_err_t ret = cde_add_sample(cde, features, 3, target);
```

### Density Estimation
```c
float density = cde_get_density(cde, features, 3, target_value);
printf("Density estimate: %.6f\n", density);
```

### Cleanup
```c
cde_deinit(cde);
```

## Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `CDE_MAX_FEATURES` | Maximum number of input features | 32 | 1-256 |
| `CDE_SAMPLE_BUFFER_SIZE` | Circular buffer size for samples | 256 | 64-4096 |
| `CDE_USE_FLOAT_PRECISION` | Use single precision floats | true | true/false |

## Memory Usage

The component's memory usage scales with configuration:
- Base overhead: ~100 bytes
- Per sample: `(max_features * 4 + 16)` bytes  
- Total: `buffer_size * (max_features * 4 + 16) + 100` bytes

Example: 128 samples × 16 features = ~8.5 KB RAM

## Algorithm

The implementation uses kernel density estimation with Gaussian kernels:

```
p(y|x) ≈ (1/n) ∑ᵢ K((x-xᵢ)/h) K((y-yᵢ)/h)
```

Where:
- `K(·)` is the Gaussian kernel
- `h` is the bandwidth parameter
- `n` is the number of samples
- `(xᵢ, yᵢ)` are training samples

## Testing with QEMU

The project includes QEMU emulator support for testing without physical hardware:

1. Ensure QEMU with ESP32 support is installed
2. Build the project: `idf.py build`
3. Run the test script: `./scripts/test_qemu.sh`

The QEMU script will:
- Launch the ESP32 emulator
- Load the firmware
- Display serial output
- Allow interactive debugging

## Continuous Integration

The project includes GitHub Actions workflows that automatically:
- Run Python-C validation (Python training tests and C compilation tests)
- Build with ESP-IDF Docker image
- Test with QEMU emulation
- Run comprehensive test suites

See `.github/workflows/ci.yml` for the complete CI/CD pipeline.

## Contributing

1. Follow ESP-IDF coding standards
2. Add tests for new features
3. Update documentation
4. Ensure QEMU compatibility

## License

This project is open source. Please check the repository for license details.

## See Also

- [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/)
- [ESP32 QEMU Documentation](https://github.com/espressif/qemu)
- [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
- [Python Training Scripts](python/README.md) - Advanced MDN/MAF implementations
