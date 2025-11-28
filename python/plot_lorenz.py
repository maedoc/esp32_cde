import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def main():
    try:
        # Load True Params
        y_true_all = np.loadtxt("lorenz_params_test.csv", delimiter=",")
        if y_true_all.ndim == 1: y_true_all = y_true_all.reshape(1, -1)
            
        # Load Samples
        df = pd.read_csv("lorenz_samples.csv")
        
        # Calculate Metrics for all test items
        print("\n--- C Model Evaluation ---")
        param_names = ['sigma', 'rho', 'beta']
        
        unique_feats = df['feature_idx'].unique()
        
        all_z_scores = []
        
        for idx in unique_feats:
            samples = df[df['feature_idx'] == idx][['p0', 'p1', 'p2']].values
            true_val = y_true_all[int(idx)]
            
            mu = np.mean(samples, axis=0)
            sigma = np.std(samples, axis=0) + 1e-9
            z = (true_val - mu) / sigma
            cv = sigma / np.abs(mu)
            
            all_z_scores.append(np.abs(z))
            
            print(f"Test {int(idx)}: True={true_val} | Mean={mu} | Std={sigma} | Z={z} | CV={cv}")

        mean_abs_z = np.mean(all_z_scores, axis=0)
        print(f"\nMean Absolute Z-score per parameter: {mean_abs_z}")
        if np.all(mean_abs_z < 1.0):
            print("SUCCESS: Mean Z-score < 1.0 indicates good fit.")
        else:
            print("WARNING: High Z-scores.")
            
        # Plot for first index
        idx = 0
        samples = df[df['feature_idx'] == idx][['p0', 'p1', 'p2']].values
        true_val = y_true_all[idx]
        
        print(f"\nPlotting for test index {idx}")
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                if i == j:
                    ax.hist(samples[:, i], bins=30, density=True, alpha=0.6, color='blue')
                    ax.axvline(true_val[i], color='red', linestyle='--', linewidth=2, label='True')
                    ax.set_title(param_names[i])
                else:
                    ax.scatter(samples[:, j], samples[:, i], s=5, alpha=0.3, color='blue')
                    ax.scatter(true_val[j], true_val[i], color='red', marker='x', s=100, linewidth=3, label='True')
                    if i == 2: ax.set_xlabel(param_names[j])
                    if j == 0: ax.set_ylabel(param_names[i])
                if i < 2: ax.set_xticklabels([])
                if j > 0: ax.set_yticklabels([])
        
        plt.suptitle(f"Lorenz Posterior (C Model) - Test #{idx}\nMAZ={np.mean(np.abs(all_z_scores[0])):.2f}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        out_file = "lorenz_posterior.png"
        plt.savefig(out_file)
        print(f"Saved plot to {out_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()