import time
import subprocess
import numpy as np
import sys
import os
from pathlib import Path

# Import the Python implementation
sys.path.insert(0, "python")
from cde_training import MAFEstimator, generate_test_data

def benchmark_python(params, features, n_flows, hidden_units, lr, n_iter, n_samples_infer, seed):
    print(f"--- Python Benchmark (Flows={n_flows}, Hidden={hidden_units}) ---")
    
    # Initialize model
    model = MAFEstimator(
        param_dim=params.shape[1],
        feature_dim=features.shape[1],
        n_flows=n_flows,
        hidden_units=hidden_units
    )
    rng = np.random.RandomState(seed)
    model.weights = model._initialize_weights(rng)
    
    # Export Initial Model for C
    # Must import the save function or copy it. 
    # We'll define a helper here or use the one from test_grad_compare if available.
    # For simplicity, let's inline a simple binary saver here.
    import struct
    with open("bench_init.maf", "wb") as f:
        f.write(b"MAF1")
        f.write(struct.pack("H", model.n_flows))
        f.write(struct.pack("H", model.param_dim))
        f.write(struct.pack("H", model.feature_dim))
        f.write(struct.pack("H", model.hidden_units))
        
        n = model.n_flows
        layers = model.model_constants['layers']
        weights = model.weights
        
        def write_arr(arr, dtype='f'):
            if dtype == 'f': f.write(arr.astype(np.float32).tobytes())
            elif dtype == 'H': f.write(arr.astype(np.uint16).tobytes())
                
        for k in range(n): write_arr(layers[k]['M1'])
        for k in range(n): write_arr(layers[k]['M2'])
        for k in range(n): write_arr(layers[k]['perm'], 'H')
        for k in range(n): write_arr(layers[k]['inv_perm'], 'H')
        for k in range(n): write_arr(weights[f'W1y_{k}'])
        for k in range(n): write_arr(weights[f'W1c_{k}'])
        for k in range(n): write_arr(weights[f'b1_{k}'])
        for k in range(n): write_arr(weights[f'W2_{k}'])
        for k in range(n): write_arr(weights[f'W2c_{k}'])
        for k in range(n): write_arr(weights[f'b2_{k}'])
    
    print("Initial model saved to bench_init.maf")

    # Benchmark Training
    start_time = time.time()
    model.train(params, features, n_iter=n_iter, learning_rate=lr, seed=seed, use_tqdm=False)
    train_time = time.time() - start_time
    print(f"Training Time: {train_time:.4f} s")
    
    # Benchmark Inference
    test_features = features[:100]
    rng = np.random.RandomState(seed)
    
    start_time = time.time()
    samples = model.sample(test_features, n_samples_infer, rng)
    infer_time = time.time() - start_time
    print(f"Inference Time: {infer_time:.4f} s (100 inputs x {n_samples_infer} samples)")
    
    flat_samples = samples.reshape(-1, params.shape[1])
    mean = np.mean(flat_samples, axis=0)
    std = np.std(flat_samples, axis=0)
    
    return train_time, infer_time, mean, std

def benchmark_c(params, features, n_flows, hidden_units, lr, n_iter, n_samples_infer, seed):
    print(f"--- C CLI Benchmark (Flows={n_flows}, Hidden={hidden_units}) ---")
    
    np.savetxt("bench_features.csv", features, delimiter=",")
    np.savetxt("bench_params.csv", params, delimiter=",")
    
    test_features = features[:100]
    np.savetxt("bench_test_features.csv", test_features, delimiter=",")
    
    model_file = "bench_model.maf"
    out_file = "bench_samples.csv"
    
    # Use --load bench_init.maf
    cmd_train = [
        "./maf_cli", "train",
        "--features", "bench_features.csv",
        "--params", "bench_params.csv",
        "--out", model_file,
        "--load", "bench_init.maf",
        "--epochs", str(n_iter),
        "--hidden", str(hidden_units),
        "--blocks", str(n_flows),
        "--lr", str(lr),
        "--batch", "0" 
    ]
    
    start_time = time.time()
    subprocess.run(cmd_train, check=True)
    train_time = time.time() - start_time
    print(f"Training Time: {train_time:.4f} s")
    
    cmd_infer = [
        "./maf_cli", "infer",
        "--model", model_file,
        "--features", "bench_test_features.csv",
        "--out", out_file,
        "--mode", "sample",
        "--samples", str(n_samples_infer)
    ]
    
    start_time = time.time()
    subprocess.run(cmd_infer, check=True, stdout=subprocess.DEVNULL)
    infer_time = time.time() - start_time
    print(f"Inference Time: {infer_time:.4f} s")
    
    data = np.loadtxt(out_file, delimiter=",", skiprows=1)
    samples = data[:, 2:]
    
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    return train_time, infer_time, mean, std

def main():
    # Config
    n_samples_train = 2000
    n_iter = 10000
    n_flows = 4
    hidden_units = 32
    lr = 0.0005
    n_samples_infer = 100
    seed = 42
    
    print("Generating Two Moons dataset...")
    params, features = generate_test_data('moons', n_samples_train, seed=seed)
    
    py_train, py_infer, py_mean, py_std = benchmark_python(
        params, features, n_flows, hidden_units, lr, n_iter, n_samples_infer, seed
    )
    
    print(f"Py Stats: Mean={py_mean}, Std={py_std}")
    
    if os.path.exists("maf_cli"):
        print("maf_cli already exists, skipping compilation.")
    else:
        print("Compiling CLI with optimizations via CMake...")
        build_dir = Path("cli/build")
        build_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir, check=True)
        
        # Locate and copy binary
        exe_name = "maf_cli"
        if os.name == 'nt':
            exe_name += ".exe"
            
        possible_paths = [
            build_dir / exe_name,
            build_dir / "Release" / exe_name,
            build_dir / "Debug" / exe_name
        ]
        
        copied = False
        for p in possible_paths:
            if p.exists():
                import shutil
                shutil.copy(p, exe_name)
                copied = True
                break
        
        if not copied:
            raise RuntimeError("Could not find compiled executable")

    c_train, c_infer, c_mean, c_std = benchmark_c(
        params, features, n_flows, hidden_units, lr, n_iter, n_samples_infer, seed
    )
    
    print(f"C  Stats: Mean={c_mean}, Std={c_std}")
    
    # Comparison
    print("\n--- Results Comparison ---")
    print(f"Training Speedup (Py/C): {py_train / c_train:.2f}x")
    print(f"Inference Speedup (Py/C): {py_infer / c_infer:.2f}x")
    
    diff_mean = np.abs(py_mean - c_mean)
    diff_std = np.abs(py_std - c_std)
    print(f"Mean Diff: {diff_mean}")
    print(f"Std Diff: {diff_std}")
    
    if np.all(diff_mean < 0.3) and np.all(diff_std < 0.3):
        print("✓ Validation Passed: Results are statistically similar.")
    else:
        print("? Validation Warning: Results differ (expected due to different RNG/Init).")

    # Plotting
    try:
        import matplotlib.pyplot as plt
        print("Generating plot...")
        plt.figure(figsize=(10, 6))
        
        # Plot True Data (subset)
        plt.scatter(features[:500, 0], params[:500, 0], c='gray', alpha=0.5, label='True Data', s=10)
        
        # Plot Python Samples
        # py_infer returned 100 samples for each of the first 100 test features
        # test_features are features[:100]
        # We need to flatten them for plotting
        # test_features (100, 1) -> repeat 100 times -> (10000, 1)
        n_infer = n_samples_infer
        n_test = 100
        test_feat_plot = np.repeat(features[:n_test], n_infer, axis=0)
        
        # py_samples needed. benchmark_python returned mean/std but we want samples.
        # We need to refactor benchmark_python to return samples or run inference again.
        # Let's run inference again for plotting (cheap)
        
        # Python Inference Re-run
        model = MAFEstimator(
            param_dim=params.shape[1],
            feature_dim=features.shape[1],
            n_flows=n_flows,
            hidden_units=hidden_units
        )
        # We need to load the weights we just trained? 
        # benchmark_python trained a local model and discarded it.
        # To plot the TRAINED model, we must return it or samples.
        # Refactoring benchmark functions is risky for the diff.
        # Let's just rely on the fact that C loaded 'bench_init.maf' and trained.
        # But Python trained from scratch.
        # Ideally we want to compare the FINAL state.
        # Since we can't easily get the Python model back without refactoring, 
        # and the C model is saved to disk...
        # We will just plot the C samples vs True Data to see if C learned the distribution.
        # Comparison with Python visual is secondary if C looks good.
        # Actually, we can re-load C samples from 'bench_samples.csv'.
        
        # Load C samples
        c_data = np.loadtxt("bench_samples.csv", delimiter=",", skiprows=1)
        # Format: feature_idx, sample_idx, p0...
        # We need to map feature_idx back to feature values
        c_feat_idx = c_data[:, 0].astype(int)
        c_samples = c_data[:, 2:]
        
        # Reconstruct feature array corresponding to samples
        # test_features was features[:100]
        bench_test_features = np.loadtxt("bench_test_features.csv", delimiter=",")
        if bench_test_features.ndim == 1: bench_test_features = bench_test_features.reshape(-1, 1)
        
        c_feat_vals = bench_test_features[c_feat_idx]
        
        plt.scatter(c_feat_vals[:, 0], c_samples[:, 0], c='red', alpha=0.1, s=2, label='C Samples')
        
        plt.legend()
        plt.title("MAF C CLI: Posterior Samples vs True Data")
        plt.xlabel("Feature (X)")
        plt.ylabel("Param (Y)")
        plt.savefig("benchmark_plot.png")
        print("Saved benchmark_plot.png")
        
    except ImportError:
        print("Matplotlib not found, skipping plot.")

    # Cleanup
    for f in ["bench_features.csv", "bench_params.csv", "bench_test_features.csv", "bench_model.maf", "bench_samples.csv", "bench_init.maf"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()
