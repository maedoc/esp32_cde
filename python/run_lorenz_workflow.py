
import numpy as np
import ctypes
import subprocess
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cde_training import MAFEstimator
from gen_lorenz_csv import generate_lorenz_data
from test_maf_c import MAF_C_Wrapper, compile_maf_library

def run_lorenz_workflow():
    print("=" * 80)
    print("Lorenz MAF Workflow: Generate -> Train -> Test (C vs Python)")
    print("=" * 80)

    # 1. Generate Data
    # We'll use a smaller sample size for speed in this interactive session, 
    # or the default if it's fast enough. 1000 samples should be fine.
    features, params = generate_lorenz_data(n_samples=1000)
    
    print(f"\nData Shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Params:   {params.shape}")

    # Split train/test
    n_train = int(0.9 * len(features))
    X_train, Y_train = features[:n_train], params[:n_train]
    X_test, Y_test = features[n_train:], params[n_train:]

    # 2. Train MAF
    print("\nTraining MAF Model...")
    # Adjust architecture for the higher dimensional data (74 dims)
    # Lorenz params are 3 dims.
    # Using 5 flows, 64 hidden units
    model = MAFEstimator(
        param_dim=Y_train.shape[1], 
        feature_dim=X_train.shape[1],
        n_flows=5,
        hidden_units=64
    )
    
    model.train(Y_train, X_train, n_iter=1000, learning_rate=1e-3, use_tqdm=True)
    print(f"Final Training Loss: {model.loss_history[-1]:.4f}")

    # 3. Compile C Library
    print("\nCompiling C Library...")
    lib_path = compile_maf_library()

    # 4. Load Model into C
    print("\nLoading Model into C...")
    wrapper = MAF_C_Wrapper(str(lib_path))
    try:
        model_ptr = wrapper.load_model(model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Model loaded. Memory usage: {wrapper.get_memory_usage(model_ptr)} bytes")

    # 5. Evaluate Performance
    print("\nEvaluating Performance on Test Set (First 5 samples)...")
    
    # We'll compare log probabilities and sample means
    
    # Select a few test cases
    indices = range(5)
    
    for i in indices:
        feat = X_test[i:i+1]
        true_param = Y_test[i:i+1]
        
        # Log Prob
        lp_py = model.log_prob(feat, true_param)[0]
        lp_c = wrapper.log_prob(model_ptr, feat, true_param)
        
        print(f"\nSample {i}:")
        print(f"  True Params: {true_param[0]}")
        print(f"  LogProb (Py): {lp_py:.4f}")
        print(f"  LogProb (C):  {lp_c:.4f}")
        print(f"  Diff:         {abs(lp_py - lp_c):.6f}")
        
        if abs(lp_py - lp_c) > 0.01:
            print("  WARNING: LogProb mismatch!")

        # Sampling
        n_samples_eval = 1000
        # Python sampling
        rng = np.random.RandomState(42 + i)
        s_py = model.sample(feat, n_samples_eval, rng)[0]
        
        # C sampling
        s_c = wrapper.sample(model_ptr, feat, n_samples_eval, seed=42 + i)
        
        mean_py = s_py.mean(axis=0)
        mean_c = s_c.mean(axis=0)
        std_py = s_py.std(axis=0)
        std_c = s_c.std(axis=0)
        
        print(f"  Mean (Py): {mean_py}")
        print(f"  Mean (C):  {mean_c}")
        print(f"  Std (Py):  {std_py}")
        print(f"  Std (C):   {std_c}")

    # Clean up
    wrapper.free_model(model_ptr)
    print("\nWorkflow Completed.")

if __name__ == "__main__":
    run_lorenz_workflow()
