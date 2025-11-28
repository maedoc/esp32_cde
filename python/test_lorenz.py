import numpy as np
from scipy.integrate import odeint
from scipy.signal import welch
from cde_training import MAFEstimator
import os
import autograd.numpy as anp

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(n_samples=200):
    print(f"Generating {n_samples} Lorenz system samples...")
    # Parameter ranges
    # sigma: [5, 15]
    # rho: [20, 40]
    # beta: [1, 5]
    
    params = []
    features = []
    
    # Simulation settings
    t = np.linspace(0, 5, 500) # 5 seconds, 100Hz
    fs = 100
    
    for i in range(n_samples):
        sigma = np.random.uniform(5, 15)
        rho = np.random.uniform(20, 40)
        beta = np.random.uniform(1, 5)
        
        state0 = [1.0, 1.0, 1.0]
        try:
            states = odeint(lorenz, state0, t, args=(sigma, rho, beta))
        except Exception as e:
            print(f"Integration failed for {sigma, rho, beta}: {e}")
            continue
            
        # Take x component
        x_ts = states[:, 0]
        
        # Compute PSD
        f, Pxx = welch(x_ts, fs=fs, nperseg=128)
        # Pxx length = 128/2 + 1 = 65
        
        # Log PSD is usually better behaved
        log_Pxx = np.log10(Pxx + 1e-10)
        
        params.append([sigma, rho, beta])
        features.append(log_Pxx)
        
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples}")
            
    return np.array(features), np.array(params)

def main():
    features, params = generate_lorenz_data(n_samples=300) 
    
    print("Features shape:", features.shape)
    print("Params shape:", params.shape)
    
    # Split train/test
    n_train = int(0.8 * len(features))
    X_train, Y_train = features[:n_train], params[:n_train]
    X_test, Y_test = features[n_train:], params[:n_train]
    
    print("Training MAF...")
    # High dimensionality in features, low in params.
    model = MAFEstimator(n_flows=5, hidden_units=64, param_dim=3, feature_dim=features.shape[1])
    
    # train takes (params, features)
    model.train(Y_train, X_train, n_iter=2000, learning_rate=1e-3, use_tqdm=True)
    
    print("Evaluating...")
    rng = anp.random.RandomState(42)
    
    # Evaluate on a few test samples
    n_eval = 5
    print(f"\nEvaluating on {n_eval} test samples:")
    print("Idx | True Params (s, r, b) | Est Mean (s, r, b)")
    print("-" * 60)
    
    for i in range(n_eval):
        x_cond = X_test[i:i+1]
        y_true = Y_test[i]
        
        # Sample
        samples = model.sample(x_cond, 500, rng)
        # samples shape (1, 500, 3)
        s_flat = samples[0]
        
        mean_est = np.mean(s_flat, axis=0)
        
        print(f"{i:3d} | {y_true[0]:5.2f} {y_true[1]:5.2f} {y_true[2]:5.2f} | {mean_est[0]:5.2f} {mean_est[1]:5.2f} {mean_est[2]:5.2f}")
    
    # Save full samples for first test item
    s_flat = model.sample(X_test[0:1], 1000, rng)[0]
    np.savetxt("lorenz_samples.csv", s_flat, delimiter=",", header="sigma,rho,beta", comments="")
    print("\nSaved samples for first test item to lorenz_samples.csv")

if __name__ == "__main__":
    main()
