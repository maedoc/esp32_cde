import numpy as np
from scipy.integrate import odeint
from scipy.signal import welch
from scipy.stats import kurtosis
import os

def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def compute_acf(x, lags=100):
    # Compute Autocorrelation
    n = len(x)
    # Centered
    x = x - np.mean(x)
    # Correlate
    r = np.correlate(x, x, mode='full')[-n:]
    # Normalize
    r = r / np.max(r)
    return r[:lags]

def generate_lorenz_data(n_samples=1000):
    print(f"Generating {n_samples} Lorenz system samples with PSD + Moments...")
    
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
            
        x_ts = states[:, 0]
        y_ts = states[:, 1]
        z_ts = states[:, 2]
        
        # 1. PSD of x
        f, Pxx = welch(x_ts, fs=fs, nperseg=128)
        log_Pxx = np.log10(Pxx + 1e-10) # ~65 dims
        
        # 2. Moments for x, y, z
        moments = []
        for series in [x_ts, y_ts, z_ts]:
            moments.append(np.mean(series))
            moments.append(np.var(series))
            moments.append(kurtosis(series))
            
        # Combine
        feat_vec = np.concatenate([log_Pxx, moments])
        
        params.append([sigma, rho, beta])
        features.append(feat_vec)
        
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{n_samples}")
            
    return np.array(features), np.array(params)

def main():
    features, params = generate_lorenz_data(n_samples=1000)
    
    # Split train/test
    n_train = int(0.9 * len(features))
    
    X_train = features[:n_train]
    Y_train = params[:n_train]
    X_test = features[n_train:]
    Y_test = params[n_train:]
    
    # Save CSVs
    print("Saving CSV files...")
    np.savetxt("lorenz_features_train.csv", X_train, delimiter=",")
    np.savetxt("lorenz_params_train.csv", Y_train, delimiter=",")
    
    # Save test set
    np.savetxt("lorenz_features_test.csv", X_test[:5], delimiter=",")
    np.savetxt("lorenz_params_test.csv", Y_test[:5], delimiter=",")
    
    print(f"Saved: train={len(X_train)}, test={len(Y_test[:5])}, feature_dim={X_train.shape[1]}")

if __name__ == "__main__":
    main()
