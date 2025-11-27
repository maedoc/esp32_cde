import numpy as np
import sklearn.datasets

n_samples = 500
noise = 0.1
X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)
X = X.astype(np.float32)

P = X[:, 1:2] 
F = X[:, 0:1] 

# Save with headers
np.savetxt("features_h.csv", F, delimiter=",", header="feature_x", comments="")
np.savetxt("params_h.csv", P, delimiter=",", header="param_y", comments="")

# Test set with header
x_test = np.linspace(-1.5, 2.5, 10).reshape(-1, 1).astype(np.float32)
np.savetxt("test_features_h.csv", x_test, delimiter=",", header="feature_x", comments="")
