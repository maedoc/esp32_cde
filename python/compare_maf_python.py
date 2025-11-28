import numpy as np
import autograd.numpy as anp
from autograd import grad
import matplotlib.pyplot as plt
import pandas as pd
from cde_training import MAFEstimator
import sys
import os
from tqdm import tqdm

def main():
    # 1. Load Data
    print("Loading data...")
    X_train = np.loadtxt("lorenz_features_train.csv", delimiter=",")
    Y_train = np.loadtxt("lorenz_params_train.csv", delimiter=",")
    X_test = np.loadtxt("lorenz_features_test.csv", delimiter=",")
    Y_test = np.loadtxt("lorenz_params_test.csv", delimiter=",")

    # Handle 1D case if necessary
    if X_train.ndim == 1: X_train = X_train[:, None]
    if Y_train.ndim == 1: Y_train = Y_train[:, None]
    if X_test.ndim == 1: X_test = X_test[:, None]
    if Y_test.ndim == 1: Y_test = Y_test[:, None]

    print(f"Train shapes: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test shapes:  X={X_test.shape},  Y={Y_test.shape}")

    # 2. Check Ranges
    print("\n--- Parameter Ranges ---")
    param_names = ['sigma', 'rho', 'beta']
    for i, name in enumerate(param_names):
        t_min, t_max = Y_train[:, i].min(), Y_train[:, i].max()
        test_min, test_max = Y_test[:, i].min(), Y_test[:, i].max()
        print(f"{name:>5}: Train [{t_min:6.2f}, {t_max:6.2f}] | Test [{test_min:6.2f}, {test_max:6.2f}]")
        
        # Check if test is inside
        if test_min < t_min or test_max > t_max:
            print(f"       WARNING: Test data outside training range for {name}!")
        else:
            print(f"       OK.")

    # 3. Setup Model
    n_flows = 5
    hidden_units = 64
    batch_size = 64
    epochs = 2000
    lr = 0.001
    
    print(f"\n--- Training Python MAF ---")
    print(f"Arch: {n_flows} flows, {hidden_units} hidden")
    print(f"Loop: {epochs} epochs, batch size {batch_size}, lr {lr}")
    
    model = MAFEstimator(
        param_dim=Y_train.shape[1],
        feature_dim=X_train.shape[1],
        n_flows=n_flows,
        hidden_units=hidden_units
    )
    
    # Initialize weights
    rng = anp.random.RandomState(42)
    model.weights = model._initialize_weights(rng)
    
    # Adam state
    m = {key: anp.zeros_like(val) for key, val in model.weights.items()}
    v = {key: anp.zeros_like(val) for key, val in model.weights.items()}
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    
    # Training Loop (Minibatch SGD)
    gradient_func = grad(model._loss_function)
    
    N = X_train.shape[0]
    n_batches = (N + batch_size - 1) // batch_size
    
    # Convert to autograd numpy
    X_train_anp = anp.array(X_train)
    Y_train_anp = anp.array(Y_train)
    
    pbar = tqdm(range(epochs), desc="Training")
    loss_history = []
    
    for epoch in pbar:
        # Shuffle
        perm = np.random.permutation(N)
        X_shuff = X_train_anp[perm]
        Y_shuff = Y_train_anp[perm]
        
        epoch_loss = 0
        
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            
            x_batch = X_shuff[start:end]
            y_batch = Y_shuff[start:end]
            
            # Gradients
            g = gradient_func(model.weights, x_batch, y_batch)
            loss = model._loss_function(model.weights, x_batch, y_batch)
            epoch_loss += loss
            
            # Adam Update
            t = (epoch * n_batches) + b + 1
            for key in model.weights:
                g_k = g[key]
                m[key] = beta1 * m[key] + (1 - beta1) * g_k
                v[key] = beta2 * v[key] + (1 - beta2) * (g_k**2)
                
                m_hat = m[key] / (1 - beta1**t)
                v_hat = v[key] / (1 - beta2**t)
                
                model.weights[key] -= lr * m_hat / (anp.sqrt(v_hat) + epsilon)
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if epoch % 100 == 0:
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            
    print(f"Final Training Loss: {loss_history[-1]:.4f}")

    # 4. Inference & Evaluation
    print("\n--- Evaluation ---")
    n_samples = 200
    
    # We evaluate on the test set
    z_scores_all = []
    
    # Prepare plot
    idx_to_plot = 0
    samples_to_plot = None
    
    print(f"Generating {n_samples} samples for each of {len(X_test)} test items...")
    
    for i in range(len(X_test)):
        # Condition
        x_cond = X_test[i:i+1] # (1, feature_dim)
        y_true = Y_test[i]
        
        # Sample
        # model.sample returns (n_cond, n_samples, param_dim)
        s = model.sample(x_cond, n_samples, rng) 
        s_flat = s[0] # (n_samples, param_dim)
        
        if i == idx_to_plot:
            samples_to_plot = s_flat
            true_to_plot = y_true
            
        # Stats
        mu = np.mean(s_flat, axis=0)
        sigma = np.std(s_flat, axis=0) + 1e-9
        
        # Z-score
        z = (y_true - mu) / sigma
        z_scores_all.append(z)
        
        print(f"Test {i}: True={y_true} | Mean={mu} | Std={sigma} | Z={z}")

    z_scores_all = np.array(z_scores_all)
    mean_abs_z = np.mean(np.abs(z_scores_all), axis=0)
    print(f"\nMean Absolute Z-score per parameter: {mean_abs_z}")
    if np.all(mean_abs_z < 1.0):
        print("SUCCESS: Mean Z-score < 1.0 indicates good fit (true value within 1 std dev).")
    else:
        print("WARNING: High Z-scores. Model might be biased or underconfident.")

    # 5. Plot
    if samples_to_plot is not None:
        print(f"\nPlotting posterior for Test #{idx_to_plot}...")
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(samples_to_plot[:, i], bins=30, density=True, alpha=0.6, color='green')
                    ax.axvline(true_to_plot[i], color='red', linestyle='--', linewidth=2, label='True')
                    ax.set_title(param_names[i])
                else:
                    ax.scatter(samples_to_plot[:, j], samples_to_plot[:, i], s=5, alpha=0.3, color='green')
                    ax.scatter(true_to_plot[j], true_to_plot[i], color='red', marker='x', s=100, linewidth=3, label='True')
                    if i == 2: ax.set_xlabel(param_names[j])
                    if j == 0: ax.set_ylabel(param_names[i])
                
                if i < 2: ax.set_xticklabels([])
                if j > 0: ax.set_yticklabels([])
        
        plt.suptitle(f"Lorenz Posterior (Python Model) - Test #{idx_to_plot}\nMAZ={np.mean(np.abs(z_scores_all[idx_to_plot])):.2f}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("lorenz_posterior_python.png")
        print("Saved lorenz_posterior_python.png")

if __name__ == "__main__":
    main()
