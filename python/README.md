# ESP32 CDE Python Training Scripts

This directory contains Python training scripts for conditional density estimation using Mixture Density Networks (MDNs) and Masked Autoregressive Flows (MAFs).

## Files

- `cde_training.py` - Main training module with MDN and MAF implementations
- `test_cde_training.py` - Unit tests for the training algorithms
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
# Run unit tests
python test_cde_training.py

# Run quick integration tests
python cde_training.py --test --no-plot

# Run full training with visualization (requires display)
python cde_training.py
```

### Command Line Options

The main training script supports:
- `--test` - Run quick tests only (suitable for CI)
- `--no-plot` - Disable matplotlib plotting (for headless environments)

## Algorithms Implemented

### Mixture Density Networks (MDN)

MDNs model conditional distributions using a mixture of Gaussians:
- Neural network maps input features to mixture parameters
- Full covariance matrices using Cholesky decomposition
- Proper negative log-likelihood training
- Gumbel-max sampling for mixture component selection

### Masked Autoregressive Flow (MAF)

MAFs use autoregressive transformations:
- MADE (Masked Autoencoder for Distribution Estimation) blocks
- Proper masking for autoregressive property
- Multiple flow layers with random permutations
- Invertible transformations for exact likelihood computation

## Test Datasets

The module includes three challenging conditional datasets:

1. **Banana**: Feature controls curvature of banana-shaped distribution
2. **Student-t**: Feature controls degrees of freedom (tail heaviness)
3. **Moons**: Feature controls noise level in two-moon distribution

## Integration with ESP32

These Python scripts serve as training/validation tools for the ESP32 CDE library. The trained models are exported to C headers and loaded into ESP32 firmware.

### Export to ESP32

The `export_maf_to_c.py` script trains a MAF model and exports it to a C header file:

```bash
# Train and export model
python export_maf_to_c.py \
  --dataset banana \
  --n-flows 3 \
  --hidden-units 32 \
  --output ../my_model.h

# This generates my_model.h with:
# - const maf_weights_t my_model_weights structure
# - All model weights as static const arrays
# - Ready to include in ESP32 firmware
```

### Validation Testing

Use `test_maf_c.py` to validate the C implementation:

```bash
# Runs complete validation:
# 1. Trains model in Python
# 2. Exports to C header
# 3. Compiles C library
# 4. Tests via ctypes
# 5. Compares Python vs C outputs
python test_maf_c.py

# Expected output:
# ✓ Log probabilities match! (0.000000 difference)
# ✓ Model loaded: 5,008 bytes
```

### Loading in ESP32

Once exported, load in ESP32 firmware:

```c
#include "maf.h"
#include "my_model.h"  // From export_maf_to_c.py

void app_main(void) {
    // Load model
    maf_model_t* model = maf_load_model(&my_model_weights);

    // Use model
    float features[] = {0.5f};
    float samples[10 * 2];
    maf_sample(model, features, 10, samples, 42);

    // Cleanup
    maf_free_model(model);
}
```

**Complete Workflow:**
1. **Python Training**: Train model and export to C header
2. **ESP-IDF Build**: Include header in project, compile to flash
3. **Runtime**: Load model from flash into RAM, use for inference
4. **Validation**: Test ensures C and Python outputs match exactly

## Mathematical Details

### MDN Forward Pass

```
h = MLP(features)
α = softmax(W_α h + b_α)          # Mixture weights
μ = reshape(W_μ h + b_μ)           # Component means
L = construct_precision_matrix(h)  # Precision matrices
```

### MAF Forward Pass

```
For each flow layer k:
  u_k = permute(u_{k-1})
  μ_k, α_k = MADE(u_k, features)
  u_k = (u_k - μ_k) * exp(-α_k)
```

## Performance Notes

- Training uses Adam optimizer with automatic gradient computation
- Models support batch processing for efficiency
- All implementations use single precision (float32) for ESP32 compatibility
- Proper numerical stability measures (gradient clipping, finite checks)