import numpy as np
import subprocess
import os
import sys

# Add python dir to path to import cde_training if needed, 
# but we mainly rely on maf_cli for training to ensure binary compatibility.
# However, we need to generate the INIT model first.
sys.path.insert(0, "python")
from cde_training import MAFEstimator, generate_test_data

def generate_and_train():
    print("Generating data...")
    # Seed 42 for reproducibility
    params, features = generate_test_data('moons', 2000, seed=42)
    
    np.savetxt("test_features.csv", features, delimiter=",")
    np.savetxt("test_params.csv", params, delimiter=",")
    
    print("Initializing model...")
    # 4 Flows, 32 Hidden Units
    model = MAFEstimator(param_dim=2, feature_dim=1, n_flows=4, hidden_units=32)
    rng = np.random.RandomState(42)
    model.weights = model._initialize_weights(rng)
    
    # Save init binary manually (copying logic from benchmark_c_vs_py.py)
    import struct
    with open("test_init.maf", "wb") as f:
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

    print("Training via maf_cli...")
    # Train for 5000 epochs to be safe
    cmd = [
        "./maf_cli", "train",
        "--features", "test_features.csv",
        "--params", "test_params.csv",
        "--out", "test_model.maf",
        "--load", "test_init.maf",
        "--epochs", "5000",
        "--hidden", "32",
        "--blocks", "4",
        "--lr", "0.0005",
        "--batch", "0"
    ]
    subprocess.run(cmd, check=True)
    
    print("Running inference to get baseline stats...")
    cmd_infer = [
        "./maf_cli", "infer",
        "--model", "test_model.maf",
        "--features", "test_features.csv", # Use training features for stats check
        "--out", "test_samples.csv",
        "--mode", "sample",
        "--samples", "100"
    ]
    subprocess.run(cmd_infer, check=True, stdout=subprocess.DEVNULL)
    
    data = np.loadtxt("test_samples.csv", delimiter=",", skiprows=1)
    samples = data[:, 2:]
    
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    print(f"BASELINE MEAN: {mean[0]}, {mean[1]}")
    print(f"BASELINE STD: {std[0]}, {std[1]}")

    # Write these to a header file for the test to verify against?
    # Or I can just parse this output.
    
    # Clean up intermediate files
    for f in ["test_features.csv", "test_params.csv", "test_init.maf", "test_samples.csv"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    generate_and_train()
