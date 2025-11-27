#!/bin/bash
set -e

echo "Generating Toy Data with Headers (Python)..."
cat > gen_toy_data.py << 'EOF'
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
EOF

python3 gen_toy_data.py

echo "Compiling CLI..."
gcc -o maf_cli main.c components/esp32_cde/src/maf.c -I components/esp32_cde/include -lm

echo "----------------------------------------------------------------"
echo "Test 1: Training with Headers (--skip-header)"
echo "----------------------------------------------------------------"
./maf_cli train \
    --features features_h.csv \
    --params params_h.csv \
    --out moons_h.maf \
    --epochs 10 \
    --hidden 16 \
    --blocks 3 \
    --lr 0.001 \
    --batch 50 \
    --skip-header

if [ -f "moons_h.maf" ]; then
    echo "✓ Model trained and saved."
else
    echo "✗ Model file not created."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Test 2: Inference Sampling with Headers (--skip-header)"
echo "----------------------------------------------------------------"
./maf_cli infer \
    --model moons_h.maf \
    --features test_features_h.csv \
    --out samples_h.csv \
    --mode sample \
    --samples 5 \
    --skip-header

if [ -f "samples_h.csv" ]; then
    echo "✓ Samples generated."
    # Check line count: 10 input rows * 5 samples + 1 header = 51 lines
    LC=$(wc -l < samples_h.csv)
    if [ "$LC" -eq 51 ]; then echo "✓ Line count correct (51)."; else echo "✗ Line count wrong: $LC"; exit 1; fi
else
    echo "✗ Samples file not created."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Test 3: Inference Stats (Mean/Std)"
echo "----------------------------------------------------------------"
./maf_cli infer \
    --model moons_h.maf \
    --features test_features_h.csv \
    --out stats_h.csv \
    --mode stats \
    --samples 100 \
    --skip-header

if [ -f "stats_h.csv" ]; then
    echo "✓ Stats generated."
    # Check structure: 10 input rows * 2 stats (mean/std) + 1 header = 21 lines
    LC=$(wc -l < stats_h.csv)
    if [ "$LC" -eq 21 ]; then echo "✓ Line count correct (21)."; else echo "✗ Line count wrong: $LC"; exit 1; fi
else
    echo "✗ Stats file not created."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Test 4: Inference Custom Quantiles (--quantiles-list)"
echo "----------------------------------------------------------------"
./maf_cli infer \
    --model moons_h.maf \
    --features test_features_h.csv \
    --out quantiles_h.csv \
    --mode quantiles \
    --samples 100 \
    --quantiles-list 0.25,0.75 \
    --skip-header

if [ -f "quantiles_h.csv" ]; then
    echo "✓ Quantiles generated."
    # Check structure: 10 input rows * 2 quantiles + 1 header = 21 lines
    LC=$(wc -l < quantiles_h.csv)
    if [ "$LC" -eq 21 ]; then echo "✓ Line count correct (21)."; else echo "✗ Line count wrong: $LC"; exit 1; fi
    
    # Verify quantiles values in CSV
    HEAD=$(head -n 1 quantiles_h.csv)
    if [[ "$HEAD" == *"quantile"* ]]; then echo "✓ Header correct."; else echo "✗ Header wrong."; exit 1; fi
else
    echo "✗ Quantiles file not created."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "All tests passed successfully!"
echo "----------------------------------------------------------------"