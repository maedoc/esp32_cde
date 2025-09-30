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

These Python scripts serve as training/validation tools for the ESP32 CDE library. The trained models can be used to:

1. Generate reference datasets for testing the ESP32 implementation
2. Validate density estimation accuracy
3. Develop optimal hyperparameters for the embedded system

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