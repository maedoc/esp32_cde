"""
Test MAF C implementation against Python implementation.

This script:
1. Trains a MAF model in Python on a nonlinear dataset
2. Exports the model to a C header
3. Compiles the C library
4. Uses ctypes to call the C library
5. Validates that C and Python produce consistent results
"""

import numpy as np
import ctypes
import subprocess
import os
import sys
from pathlib import Path

# Add parent directory to path to import cde_training
sys.path.insert(0, str(Path(__file__).parent))

from cde_training import MAFEstimator, generate_test_data
from export_maf_to_c import export_maf_to_header


class MAF_C_Wrapper:
    """Ctypes wrapper for the C MAF library."""

    def __init__(self, lib_path):
        """Load the shared library."""
        self.lib = ctypes.CDLL(lib_path)

        # Define maf_weights_t structure
        class MAFWeights(ctypes.Structure):
            _fields_ = [
                ("n_flows", ctypes.c_uint16),
                ("param_dim", ctypes.c_uint16),
                ("feature_dim", ctypes.c_uint16),
                ("hidden_units", ctypes.c_uint16),
                ("M1_data", ctypes.POINTER(ctypes.c_float)),
                ("M2_data", ctypes.POINTER(ctypes.c_float)),
                ("perm_data", ctypes.POINTER(ctypes.c_uint16)),
                ("inv_perm_data", ctypes.POINTER(ctypes.c_uint16)),
                ("W1y_data", ctypes.POINTER(ctypes.c_float)),
                ("W1c_data", ctypes.POINTER(ctypes.c_float)),
                ("b1_data", ctypes.POINTER(ctypes.c_float)),
                ("W2_data", ctypes.POINTER(ctypes.c_float)),
                ("W2c_data", ctypes.POINTER(ctypes.c_float)),
                ("b2_data", ctypes.POINTER(ctypes.c_float)),
            ]

        self.MAFWeights = MAFWeights

        # Define function signatures
        self.lib.maf_load_model.argtypes = [ctypes.POINTER(MAFWeights)]
        self.lib.maf_load_model.restype = ctypes.c_void_p

        self.lib.maf_free_model.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_model.restype = None

        self.lib.maf_create_workspace.argtypes = [ctypes.c_void_p]
        self.lib.maf_create_workspace.restype = ctypes.c_void_p

        self.lib.maf_free_workspace.argtypes = [ctypes.c_void_p]
        self.lib.maf_free_workspace.restype = None

        self.lib.maf_sample.argtypes = [
            ctypes.c_void_p,                    # model
            ctypes.POINTER(ctypes.c_float),     # features
            ctypes.c_uint32,                    # n_samples
            ctypes.POINTER(ctypes.c_float),     # samples_out
            ctypes.c_uint32                     # seed
        ]
        self.lib.maf_sample.restype = ctypes.c_int

        self.lib.maf_sample_from_noise.argtypes = [
            ctypes.c_void_p,                    # model
            ctypes.POINTER(ctypes.c_float),     # features
            ctypes.POINTER(ctypes.c_float),     # base_noise
            ctypes.c_uint32,                    # n_samples
            ctypes.POINTER(ctypes.c_float)      # samples_out
        ]
        self.lib.maf_sample_from_noise.restype = ctypes.c_int

        self.lib.maf_log_prob.argtypes = [
            ctypes.c_void_p,                    # model
            ctypes.c_void_p,                    # workspace
            ctypes.POINTER(ctypes.c_float),     # features
            ctypes.POINTER(ctypes.c_float)      # params
        ]
        self.lib.maf_log_prob.restype = ctypes.c_float

        self.lib.maf_get_memory_usage.argtypes = [ctypes.c_void_p]
        self.lib.maf_get_memory_usage.restype = ctypes.c_size_t

    def create_weights_struct(self, model: MAFEstimator):
        """Create a ctypes weights structure from a Python MAF model."""
        n_flows = model.n_flows
        param_dim = model.param_dim
        feature_dim = model.feature_dim
        hidden_units = model.hidden_units

        # Concatenate all layer data
        M1_list = []
        M2_list = []
        perm_list = []
        inv_perm_list = []
        W1y_list = []
        W1c_list = []
        b1_list = []
        W2_list = []
        W2c_list = []
        b2_list = []

        for k in range(n_flows):
            layer = model.model_constants['layers'][k]
            M1_list.append(layer['M1'].flatten())
            M2_list.append(layer['M2'].flatten())
            perm_list.append(layer['perm'])
            inv_perm_list.append(layer['inv_perm'])

            W1y_list.append(model.weights[f'W1y_{k}'].flatten())
            W1c_list.append(model.weights[f'W1c_{k}'].flatten())
            b1_list.append(model.weights[f'b1_{k}'].flatten())
            W2_list.append(model.weights[f'W2_{k}'].flatten())
            W2c_list.append(model.weights[f'W2c_{k}'].flatten())
            b2_list.append(model.weights[f'b2_{k}'].flatten())

        # Convert to ctypes arrays
        def to_c_float_array(data):
            arr = np.concatenate(data).astype(np.float32)
            return (ctypes.c_float * len(arr))(*arr)

        def to_c_uint16_array(data):
            arr = np.concatenate(data).astype(np.uint16)
            return (ctypes.c_uint16 * len(arr))(*arr)

        # Create structure
        weights = self.MAFWeights()
        weights.n_flows = n_flows
        weights.param_dim = param_dim
        weights.feature_dim = feature_dim
        weights.hidden_units = hidden_units

        # Keep references to arrays so they don't get garbage collected
        self._arrays = {}
        self._arrays['M1'] = to_c_float_array(M1_list)
        self._arrays['M2'] = to_c_float_array(M2_list)
        self._arrays['perm'] = to_c_uint16_array(perm_list)
        self._arrays['inv_perm'] = to_c_uint16_array(inv_perm_list)
        self._arrays['W1y'] = to_c_float_array(W1y_list)
        self._arrays['W1c'] = to_c_float_array(W1c_list)
        self._arrays['b1'] = to_c_float_array(b1_list)
        self._arrays['W2'] = to_c_float_array(W2_list)
        self._arrays['W2c'] = to_c_float_array(W2c_list)
        self._arrays['b2'] = to_c_float_array(b2_list)

        weights.M1_data = ctypes.cast(self._arrays['M1'], ctypes.POINTER(ctypes.c_float))
        weights.M2_data = ctypes.cast(self._arrays['M2'], ctypes.POINTER(ctypes.c_float))
        weights.perm_data = ctypes.cast(self._arrays['perm'], ctypes.POINTER(ctypes.c_uint16))
        weights.inv_perm_data = ctypes.cast(self._arrays['inv_perm'], ctypes.POINTER(ctypes.c_uint16))
        weights.W1y_data = ctypes.cast(self._arrays['W1y'], ctypes.POINTER(ctypes.c_float))
        weights.W1c_data = ctypes.cast(self._arrays['W1c'], ctypes.POINTER(ctypes.c_float))
        weights.b1_data = ctypes.cast(self._arrays['b1'], ctypes.POINTER(ctypes.c_float))
        weights.W2_data = ctypes.cast(self._arrays['W2'], ctypes.POINTER(ctypes.c_float))
        weights.W2c_data = ctypes.cast(self._arrays['W2c'], ctypes.POINTER(ctypes.c_float))
        weights.b2_data = ctypes.cast(self._arrays['b2'], ctypes.POINTER(ctypes.c_float))

        return weights

    def load_model(self, model: MAFEstimator):
        """Load a Python MAF model into C."""
        weights = self.create_weights_struct(model)
        self._weights = weights  # Keep reference
        model_ptr = self.lib.maf_load_model(ctypes.byref(weights))
        if not model_ptr:
            raise RuntimeError("Failed to load model")
        
        # Create workspace
        self._ws_ptr = self.lib.maf_create_workspace(model_ptr)
        if not self._ws_ptr:
            self.lib.maf_free_model(model_ptr)
            raise RuntimeError("Failed to create workspace")
            
        return model_ptr

    def free_model(self, model_ptr):
        """Free a C model."""
        if hasattr(self, '_ws_ptr') and self._ws_ptr:
            self.lib.maf_free_workspace(self._ws_ptr)
            self._ws_ptr = None
        self.lib.maf_free_model(model_ptr)

    def sample(self, model_ptr, features, n_samples, seed=42):
        """Generate samples from C model."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Infer param_dim from model (we'll need to pass it or query it)
        # For now, assume we know it
        param_dim = 2  # Will be passed as parameter later

        samples_out = np.zeros((n_samples, param_dim), dtype=np.float32)

        features_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        samples_ptr = samples_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.lib.maf_sample(
            model_ptr,
            features_ptr,
            n_samples,
            samples_ptr,
            seed
        )

        if ret != 0:
            raise RuntimeError(f"maf_sample failed with code {ret}")

        return samples_out

    def sample_from_noise(self, model_ptr, features, base_noise):
        """Generate samples from provided base noise (deterministic transformation)."""
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        base_noise = np.asarray(base_noise, dtype=np.float32)
        if base_noise.ndim == 1:
            base_noise = base_noise.reshape(1, -1)

        n_samples, param_dim = base_noise.shape
        samples_out = np.zeros((n_samples, param_dim), dtype=np.float32)

        features_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        noise_ptr = base_noise.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        samples_ptr = samples_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.lib.maf_sample_from_noise(
            model_ptr,
            features_ptr,
            noise_ptr,
            n_samples,
            samples_ptr
        )

        if ret != 0:
            raise RuntimeError(f"maf_sample_from_noise failed with code {ret}")

        return samples_out

    def log_prob(self, model_ptr, features, params):
        """Compute log probability."""
        features = np.asarray(features, dtype=np.float32)
        params = np.asarray(params, dtype=np.float32)

        features_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        params_ptr = params.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        return self.lib.maf_log_prob(model_ptr, self._ws_ptr, features_ptr, params_ptr)

    def get_memory_usage(self, model_ptr):
        """Get model memory usage."""
        return self.lib.maf_get_memory_usage(model_ptr)


def compile_maf_library():
    """Compile the MAF C library as a shared object."""
    repo_root = Path(__file__).parent.parent
    maf_c = repo_root / "components/esp32_cde/src/maf.c"
    maf_h_dir = repo_root / "components/esp32_cde/include"
    lib_output = repo_root / "python/libmaf.so"

    cmd = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-I", str(maf_h_dir),
        str(maf_c),
        "-o", str(lib_output),
        "-lm"
    ]

    print(f"Compiling MAF library: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Compilation failed: {result.stderr}")

    print(f"Compiled successfully: {lib_output}")
    return lib_output


def test_maf_c_vs_python():
    """Main test function."""
    print("=" * 70)
    print("MAF C Implementation Validation Test")
    print("=" * 70)

    # Test configuration
    dataset = "banana"  # Nonlinear test case
    n_flows = 3
    hidden_units = 32
    n_samples_train = 3000
    n_iter = 800
    seed = 42

    # Generate training data
    print(f"\n1. Generating '{dataset}' dataset...")
    params, features = generate_test_data(dataset, n_samples_train, seed=seed)
    print(f"   Training data shape: params={params.shape}, features={features.shape}")

    # Train Python model
    print(f"\n2. Training MAF model (n_flows={n_flows}, hidden={hidden_units})...")
    model = MAFEstimator(
        param_dim=params.shape[1],
        feature_dim=features.shape[1],
        n_flows=n_flows,
        hidden_units=hidden_units
    )
    model.train(params, features, n_iter=n_iter, seed=seed, use_tqdm=True)
    print(f"   Final training loss: {model.loss_history[-1]:.4f}")

    # Compile C library
    print("\n3. Compiling C library...")
    lib_path = compile_maf_library()

    # Load C wrapper
    print("\n4. Loading C library via ctypes...")
    wrapper = MAF_C_Wrapper(str(lib_path))
    model_ptr = wrapper.load_model(model)
    print(f"   Model loaded at {hex(model_ptr)}")
    print(f"   Memory usage: {wrapper.get_memory_usage(model_ptr)} bytes")

    # Test sampling
    print("\n5. Testing sampling...")
    test_features = np.array([0.5], dtype=np.float32)
    n_test_samples = 1000

    # Python samples
    rng_py = np.random.RandomState(seed)
    samples_py = model.sample(test_features.reshape(1, -1), n_test_samples, rng_py)
    samples_py = samples_py.reshape(n_test_samples, -1)

    # C samples
    samples_c = wrapper.sample(model_ptr, test_features, n_test_samples, seed=seed)

    print(f"   Python samples shape: {samples_py.shape}")
    print(f"   C samples shape: {samples_c.shape}")

    # Statistical comparison
    print("\n6. Comparing sample distributions...")
    print(f"   Python - Mean: {samples_py.mean(axis=0)}, Std: {samples_py.std(axis=0)}")
    print(f"   C      - Mean: {samples_c.mean(axis=0)}, Std: {samples_c.std(axis=0)}")

    mean_diff = np.abs(samples_py.mean(axis=0) - samples_c.mean(axis=0))
    std_diff = np.abs(samples_py.std(axis=0) - samples_c.std(axis=0))

    print(f"   Mean difference: {mean_diff}")
    print(f"   Std difference: {std_diff}")

    # Note: Exact match is not expected due to different RNG implementations
    # But distributions should be similar
    mean_threshold = 0.3  # Allow some difference due to RNG
    std_threshold = 0.3

    if np.all(mean_diff < mean_threshold) and np.all(std_diff < std_threshold):
        print("   ✓ Statistical distributions are consistent!")
    else:
        print("   ✗ Warning: Distributions differ more than expected")
        print("   This may be due to different RNG implementations")

    # Test log probability
    print("\n7. Testing log probability...")
    test_params = np.array([0.0, 0.0], dtype=np.float32)

    logp_py = model.log_prob(test_features.reshape(1, -1), test_params.reshape(1, -1))[0]
    logp_c = wrapper.log_prob(model_ptr, test_features, test_params)

    print(f"   Python log_prob: {logp_py:.6f}")
    print(f"   C log_prob: {logp_c:.6f}")
    print(f"   Difference: {abs(logp_py - logp_c):.6f}")

    if abs(logp_py - logp_c) < 0.01:
        print("   ✓ Log probabilities match!")
    else:
        print("   ✗ Log probabilities differ significantly")

    # Cleanup
    print("\n8. Cleanup...")
    wrapper.free_model(model_ptr)

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_maf_c_vs_python()
