"""
Rigorous test for MAF sampling: verify the inverse transformation is correct.

Instead of relying on RNG matching, this test:
1. Generates base noise z in Python
2. Passes same z to both Python and C implementations
3. Verifies the transformed outputs match exactly
"""

import numpy as np
import ctypes
from pathlib import Path
from test_maf_c import compile_maf_library, MAF_C_Wrapper
from cde_training import MAFEstimator, generate_test_data


def test_inverse_transformation_matches():
    """
    Test that Python and C produce the same output given the same base noise.
    This validates the transformation logic independent of RNG.
    """
    print("=" * 70)
    print("MAF Sampling Validation: Inverse Transformation Test")
    print("=" * 70)

    # Train a simple model
    print("\n1. Training MAF model...")
    params, features = generate_test_data("banana", 2000, seed=42)

    model = MAFEstimator(param_dim=2, feature_dim=1, n_flows=2, hidden_units=16)
    model.train(params, features, n_iter=500, seed=42, use_tqdm=True)
    print(f"   Training loss: {model.loss_history[-1]:.4f}")

    # Compile C library
    print("\n2. Compiling C library...")
    lib_path = compile_maf_library()
    wrapper = MAF_C_Wrapper(str(lib_path))
    model_ptr = wrapper.load_model(model)
    print(f"   Model loaded at {hex(model_ptr)}")

    # Test features
    test_features = np.array([[0.5]], dtype=np.float32)

    # Generate base noise in Python
    print("\n3. Generating base noise z ~ N(0, I)...")
    n_test_samples = 100
    rng = np.random.RandomState(12345)
    base_noise = rng.randn(n_test_samples, model.param_dim).astype(np.float32)
    print(f"   Generated {n_test_samples} noise vectors")
    print(f"   Noise mean: {base_noise.mean(axis=0)}")
    print(f"   Noise std: {base_noise.std(axis=0)}")

    # Transform in Python
    print("\n4. Applying inverse transformation in Python...")
    samples_py = manual_maf_inverse_python(model, test_features, base_noise)
    print(f"   Python samples shape: {samples_py.shape}")
    print(f"   Python samples mean: {samples_py.mean(axis=0)}")
    print(f"   Python samples std: {samples_py.std(axis=0)}")

    # Transform in C
    print("\n5. Applying inverse transformation in C...")
    samples_c = manual_maf_inverse_c(wrapper, model_ptr, test_features, base_noise)
    print(f"   C samples shape: {samples_c.shape}")
    print(f"   C samples mean: {samples_c.mean(axis=0)}")
    print(f"   C samples std: {samples_c.std(axis=0)}")

    # Compare outputs
    print("\n6. Comparing outputs...")
    max_diff = np.max(np.abs(samples_py - samples_c))
    mean_diff = np.mean(np.abs(samples_py - samples_c))

    print(f"   Max absolute difference: {max_diff:.6f}")
    print(f"   Mean absolute difference: {mean_diff:.6f}")

    # Check per-sample differences
    sample_diffs = np.abs(samples_py - samples_c)
    print(f"   Per-dimension max diff: {sample_diffs.max(axis=0)}")
    print(f"   Per-dimension mean diff: {sample_diffs.mean(axis=0)}")

    # Success criteria
    tolerance = 1e-4  # Allow small numerical differences
    if max_diff < tolerance:
        print(f"\n   ✓ SUCCESS: Transformations match (max diff {max_diff:.2e} < {tolerance})")
        success = True
    else:
        print(f"\n   ✗ FAILURE: Transformations differ (max diff {max_diff:.2e} >= {tolerance})")
        print("\n   First 5 samples comparison:")
        for i in range(min(5, n_test_samples)):
            print(f"   Sample {i}:")
            print(f"     Python: {samples_py[i]}")
            print(f"     C:      {samples_c[i]}")
            print(f"     Diff:   {samples_py[i] - samples_c[i]}")
        success = False

    # Cleanup
    wrapper.free_model(model_ptr)

    print("\n" + "=" * 70)
    return success


def manual_maf_inverse_python(model: MAFEstimator, features: np.ndarray,
                               base_noise: np.ndarray) -> np.ndarray:
    """
    Manually apply MAF inverse transformation using Python implementation.

    This bypasses the RNG and uses provided base noise.
    """
    import autograd.numpy as anp

    features_expanded = anp.repeat(features, base_noise.shape[0], axis=0)
    x = base_noise.copy()

    # Invert flow stack (reverse order)
    for k in reversed(range(model.n_flows)):
        layer_const = model.model_constants['layers'][k]

        # Apply permutation
        y_perm = x[:, layer_const['perm']]

        # Autoregressive inversion
        u = anp.zeros_like(y_perm)
        for i in range(model.param_dim):
            # Forward pass to get mu and alpha at dimension i
            mu, alpha = manual_made_forward(
                model, k, u, features_expanded
            )
            # Invert: u[i] = y[i] * exp(alpha[i]) + mu[i]
            u[:, i] = y_perm[:, i] * anp.exp(alpha[:, i]) + mu[:, i]

        # Apply inverse permutation
        x = u[:, layer_const['inv_perm']]

    return np.array(x).astype(np.float32)


def manual_made_forward(model: MAFEstimator, layer_idx: int,
                       y: np.ndarray, context: np.ndarray):
    """Manual MADE forward pass."""
    import autograd.numpy as anp

    k = layer_idx
    layer_const = model.model_constants['layers'][k]

    W1y = model.weights[f'W1y_{k}']
    W1c = model.weights[f'W1c_{k}']
    b1 = model.weights[f'b1_{k}']
    W2 = model.weights[f'W2_{k}']
    W2c = model.weights[f'W2c_{k}']
    b2 = model.weights[f'b2_{k}']
    M1 = layer_const['M1']
    M2 = layer_const['M2']

    # Hidden layer
    y_h = anp.dot(y, (W1y * M1).T)
    c_h = anp.dot(context, W1c.T) if model.feature_dim > 0 else 0.0
    h = anp.tanh(y_h + c_h + b1)

    # Output layer
    M2_tiled = anp.concatenate([M2, M2], axis=0)
    out = anp.dot(h, (W2 * M2_tiled).T)
    if model.feature_dim > 0:
        out = out + anp.dot(context, W2c.T)
    out = out + b2

    # Split into mu and alpha
    mu = out[:, :model.param_dim]
    alpha = anp.clip(out[:, model.param_dim:], -7.0, 7.0)

    return mu, alpha


def manual_maf_inverse_c(wrapper: MAF_C_Wrapper, model_ptr: int,
                         features: np.ndarray, base_noise: np.ndarray) -> np.ndarray:
    """
    Call C implementation with pre-generated base noise.
    """
    # Now we can pass base noise directly!
    samples = wrapper.sample_from_noise(model_ptr, features[0], base_noise)
    return samples


if __name__ == "__main__":
    success = test_inverse_transformation_matches()

    if success:
        print("\n✓ All tests passed!")
        print("✓ C and Python implementations produce identical results!")
        exit(0)
    else:
        print("\n✗ Tests failed!")
        print("✗ C and Python implementations differ beyond numerical precision")
        exit(1)
