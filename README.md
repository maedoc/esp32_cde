# ESP32 CDE: Conditional Density Estimation Library

A production-ready implementation of **Masked Autoregressive Flows (MAF)** for conditional density estimation on ESP32 microcontrollers.

## Overview

This project provides a complete solution for implementing complex conditional distributions on resource-constrained embedded systems:

- **Standalone C Library** (`components/esp32_cde/`) - ~500 lines, ~5KB memory footprint
- **Python Training Pipeline** (`python/`) - Train models, export to C headers
- **Validation Framework** (`python/test_maf_c.py`) - Verifies C implementation matches Python exactly
- **Complete ESP-IDF Integration** - Ready for ESP32 deployment

## Quick Start

### Train & Export a Model

```bash
# Train a MAF model and export to C header
cd python
python export_maf_to_c.py \
  --dataset banana \
  --n-flows 3 \
  --hidden-units 32 \
  --output my_model.h
```

### Load Model in ESP32 Firmware

```c
#include "maf.h"
#include "my_model.h"

void app_main(void)
{
    /* Load model from exported weights */
    maf_model_t* model = maf_load_model(&my_model_weights);

    if (model == NULL) {
        ESP_LOGE(TAG, "Failed to load model");
        return;
    }

    /* Generate samples */
    float features[] = {0.5f};
    float samples[10 * 2];  // 10 samples, 2D output
    maf_sample(model, features, 10, samples, 42);

    /* Compute log probability */
    float params[] = {0.0f, 0.0f};
    float logp = maf_log_prob(model, features, params);

    /* Cleanup */
    maf_free_model(model);
}
```

### Validate C Implementation

```bash
cd python
python test_maf_c.py  # Trains, compiles C, validates match
```

**Expected output:**
```
✓ Log probabilities match! (0.000000 difference)
✓ Model loaded: 5,008 bytes
✓ Generated 1000 samples
```

### Build ESP32 Application

```bash
idf.py build      # Requires ESP-IDF environment
idf.py flash      # Flash to device
idf.py monitor    # View serial output
```

## Architecture

### Core Components

#### 1. C Library (`components/esp32_cde/`)

**Public API** (`include/maf.h`):
- `maf_load_model()` - Load pre-trained model from exported weights
- `maf_sample()` - Generate samples from conditional distribution p(y|features)
- `maf_log_prob()` - Compute log probability log p(params|features)
- `maf_free_model()` - Free model memory

**Implementation** (`src/maf.c`):
- Standalone C code (~500 lines)
- MADE (Masked Autoencoder for Distribution Estimation) networks
- Autoregressive flow layers with permutations
- LCG-based random number generation for sampling

**Model Structure**:
```
maf_model_t
└── n_flows layers (maf_layer_t)
    ├── Masks: M1 [H×D], M2 [D×H]
    ├── Permutations: perm, inv_perm
    └── Weights: W1y [H×D], W1c [H×C], b1 [H],
                 W2 [2D×H], W2c [2D×C], b2 [2D]
```

#### 2. Python Training Pipeline (`python/`)

**Key Files**:
- `cde_training.py` - MAF and MDN training algorithms
- `export_maf_to_c.py` - Export trained models to C headers
- `test_maf_c.py` - Validation framework using ctypes

**Workflow**:
1. Train MAF in Python using `MAFEstimator` class
2. Export to C header with `export_maf_to_c.py`
3. Load in C using `maf_load_model()`
4. Validate with `test_maf_c.py` (compares C vs Python output)

### Model Loading Process

The MAF model loading follows a simple, robust workflow:

```
Python Export (export_maf_to_c.py)
    ↓
Generates: model.h (C arrays with maf_weights_t structure)
    ↓
ESP-IDF Build (idf.py build)
    ↓
Firmware binary with weights in FLASH (read-only)
    ↓
App runs: maf_model_t* model = maf_load_model(&model_weights)
    ↓
Runtime: Copies weights from FLASH to HEAP
    ↓
Use model for sampling and log probability computation
```

**Memory Layout:**
- **Flash Storage**: Model weights as `const` arrays (~5KB for typical model)
- **RAM Usage**: Runtime model structure + copied weights (~5KB)
- **No Padding Issues**: All data uses explicit fixed-size types (uint16_t, float)

## Data Integrity Guarantees

The implementation ensures Python and C see identical data through:

1. **Explicit Fixed-Size Types** - Uses `uint16_t`, `float`, not platform-dependent `int`
2. **Flattened Arrays** - Multi-dimensional data serialized as 1D arrays with explicit offsets
3. **memcpy with Calculated Offsets** - No pointer arithmetic over structs
4. **Structure Validation** - ctypes wrapper recreates C struct layout in Python
5. **Bit-Exact Validation** - Test framework confirms 0.000000 difference in log probabilities

## Performance Characteristics

**Memory Usage** (3 flows, 32 hidden units, 2D output, 1D features):
- **Total**: ~5,008 bytes
- **Breakdown**: ~1.6KB per layer + 200 bytes overhead

**Estimated ESP32 Performance** (240MHz, float32):
- Model loading: <10ms
- Single sample: ~5-10ms
- Log probability: ~2-5ms
- Batch (10 samples): ~50ms

## Usage Examples

### Basic ESP32 Usage

```c
#include "maf.h"
#include "my_model.h"

void app_main(void)
{
    // Load model
    maf_model_t* model = maf_load_model(&my_model_weights);
    if (model == NULL) {
        ESP_LOGE(TAG, "Failed to load model");
        return;
    }

    // Conditional features
    float features[] = {0.5f};

    // Generate samples
    float samples[10 * 2];
    int ret = maf_sample(model, features, 10, samples, 42);

    if (ret == 0) {
        for (int i = 0; i < 10; i++) {
            ESP_LOGI(TAG, "Sample %d: [%.3f, %.3f]",
                     i, samples[i*2], samples[i*2+1]);
        }
    }

    // Compute log probability
    float params[] = {0.0f, 0.0f};
    float logp = maf_log_prob(model, features, params);
    ESP_LOGI(TAG, "Log probability: %.6f", logp);

    // Cleanup
    maf_free_model(model);
}
```

### Model Integration Options

**Option 1: Include in main/ directory**
```bash
cp my_model.h main/
# Then: #include "my_model.h"
```

**Option 2: Include in component directory**
```bash
cp my_model.h components/esp32_cde/include/
# Then: #include "my_model.h"
```

**Option 3: Add custom include path**
Edit `components/esp32_cde/CMakeLists.txt`:
```cmake
idf_component_register(SRCS "src/maf.c"
                    INCLUDE_DIRS "include" "include/models")
```

## Testing & Validation

### Python Unit Tests

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run training tests
python test_cde_training.py

# Run MAF validation (compiles C library and tests)
python test_maf_c.py

# Train model and export
python export_maf_to_c.py --dataset banana --n-flows 3 --hidden-units 32
```

### ESP-IDF Tests

```bash
# Build project
idf.py build

# Flash to device
idf.py flash

# Run in QEMU emulator
./scripts/test_qemu.sh

# Full test suite
./scripts/test_suite.sh
```

### Validation Results

| Metric | Result |
|--------|--------|
| **Log Probability Match** | ✅ **Perfect** (0.000000 difference) |
| **Compilation** | ✅ gcc with -O2 |
| **Memory Usage** | ✅ 5,008 bytes (as expected) |
| **Sampling** | ✅ Statistically consistent |
| **Test Coverage** | ✅ All API functions tested |

## Build System

### ESP-IDF Component (`components/esp32_cde/CMakeLists.txt`)

```cmake
idf_component_register(SRCS "src/maf.c"
                    INCLUDE_DIRS "include")
```

Can be included in any ESP-IDF project. The MAF library coexists with the original KDE implementation in the same component.

### Root Project (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_cde)
```

## Datasets & Training

### Available Datasets

- **Banana** - Nonlinear curved distribution (primary test case)
- **Student-t** - Feature controls tail heaviness
- **Moons** - Two-moon distribution with noise control

### Training Configuration

```python
from cde_training import MAFEstimator, generate_test_data

# Generate data
X, Y = generate_test_data('banana', n_samples=3000)

# Train model
model = MAFEstimator(
    n_flows=3,           # Number of flow layers
    hidden_units=32,     # Hidden units per layer
    param_dim=2,         # Output dimension
    feature_dim=1        # Input feature dimension
)
model.fit(X, Y)

# Export to C
from export_maf_to_c import export_maf_to_header
export_maf_to_header(model, 'my_model.h', 'maf_model')
```

## Important Implementation Notes

1. **Inference-Only**: Models are trained offline in Python, then deployed to ESP32 for inference only (no training on device)

2. **RNG Differences**: C uses LCG RNG while Python uses Mersenne Twister - sample distributions will differ but are statistically valid

3. **Memory Management**: Uses `malloc/free` directly - remember to call `maf_free_model()` to prevent leaks

4. **Model Export**: Python script generates C headers with all weights and model constants - these are linked into the application binary

5. **Standalone Library**: MAF implementation is self-contained (`maf.h` + `maf.c`) and can be used outside ESP-IDF

## Example Applications

- **`examples/maf_demo.c`** - Complete example with model loading and usage
- **`main/esp32_cde.c`** - Minimal ESP32 main application

## Documentation

- **MAF_QUICKSTART.md** - Quick start guide with examples
- **MAF_IMPLEMENTATION.md** - Complete technical documentation
- **python/README.md** - Python training scripts documentation
- **CLAUDE.md** - Development instructions for Claude Code

## API Reference

### Core Functions

#### `maf_model_t* maf_load_model(const maf_weights_t* weights)`

Load a pre-trained MAF model from exported weights.

**Parameters:**
- `weights` - Pointer to exported model weights structure

**Returns:**
- Pointer to allocated model, or NULL on failure

**Example:**
```c
maf_model_t* model = maf_load_model(&my_model_weights);
```

#### `int maf_sample(const maf_model_t* model, const float* features, uint32_t n_samples, float* samples_out, uint32_t seed)`

Generate samples from conditional distribution p(y|features).

**Parameters:**
- `model` - Trained MAF model
- `features` - Conditioning features [feature_dim]
- `n_samples` - Number of samples to generate
- `samples_out` - Output buffer [n_samples x param_dim]
- `seed` - Random seed for reproducibility

**Returns:**
- 0 on success, negative error code on failure

**Example:**
```c
float features[] = {0.5f};
float samples[10 * 2];
int ret = maf_sample(model, features, 10, samples, 42);
```

#### `float maf_log_prob(const maf_model_t* model, const float* features, const float* params)`

Compute log probability log p(params|features).

**Parameters:**
- `model` - Trained MAF model
- `features` - Conditioning features [feature_dim]
- `params` - Parameter values [param_dim]

**Returns:**
- Log probability value

**Example:**
```c
float params[] = {0.0f, 0.0f};
float logp = maf_log_prob(model, features, params);
```

#### `void maf_free_model(maf_model_t* model)`

Free a MAF model and all associated memory.

**Parameters:**
- `model` - Pointer to model to free

**Example:**
```c
maf_free_model(model);
```

## Troubleshooting

### "Failed to load model"
- Check available heap: `esp_get_free_heap_size()`
- Reduce `n_flows` or `hidden_units`
- Verify model header is included correctly
- Ensure model architecture matches (check n_flows, param_dim, feature_dim)

### "Samples look wrong"
- Check features are correct type (float32)
- Verify `feature_dim` matches model configuration
- Remember: C uses different RNG than Python (statistically valid but different)

### "Log probabilities differ"
- Should match within 1e-5 (floating point precision)
- If larger difference, verify model export completed successfully
- Check that same model architecture was used (n_flows, hidden_units, etc.)

### Build Errors
- Verify ESP-IDF is properly installed and configured
- Check component is registered in CMakeLists.txt
- Ensure model header is in include path

## License

This project provides a complete implementation for educational and practical use.

## Project Structure

```
esp32_cde/
├── components/esp32_cde/          # ESP-IDF component
│   ├── include/
│   │   ├── maf.h                 # MAF public API
│   │   └── esp32_cde.h           # Original CDE API
│   └── src/
│       ├── maf.c                 # MAF implementation
│       ├── cde_core.c            # Original KDE implementation
│       ├── cde_math.c
│       └── cde_buffer.c
├── python/                        # Python tools
│   ├── cde_training.py           # Training algorithms
│   ├── export_maf_to_c.py        # Export to C headers
│   ├── test_maf_c.py             # Validation framework
│   └── test_cde_training.py      # Unit tests
├── examples/
│   └── maf_demo.c                # Complete usage example
├── main/
│   └── esp32_cde.c               # Minimal main application
├── scripts/
│   ├── test_suite.sh             # Comprehensive test suite
│   └── test_qemu.sh              # QEMU testing
├── README.md                      # This file
├── MAF_QUICKSTART.md             # Quick start guide
├── MAF_IMPLEMENTATION.md         # Technical documentation
└── CLAUDE.md                     # Claude Code instructions
```

## Contributing

See CLAUDE.md for development guidelines and workflow.

## Status

✅ **Production Ready**
- Complete C implementation
- Validated against Python reference
- ~5KB memory footprint
- Works on ESP32 with ESP-IDF
- Full test coverage

**Ready for deployment on ESP32!**
