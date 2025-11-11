"""
Export trained MAF models to C header files.

This script trains a MAF model and exports all weights and constants
to a C header file that can be loaded by the maf.c library.
"""

import numpy as np
from cde_training import MAFEstimator, generate_test_data


def export_maf_to_header(model: MAFEstimator, output_path: str, model_name: str = "maf_model"):
    """
    Export a trained MAF model to a C header file.

    Parameters
    ----------
    model : MAFEstimator
        Trained MAF model
    output_path : str
        Path to output .h file
    model_name : str
        Name prefix for C variables
    """
    if model.weights is None:
        raise ValueError("Model must be trained before export")

    n_flows = model.n_flows
    param_dim = model.param_dim
    feature_dim = model.feature_dim
    hidden_units = model.hidden_units

    layers = model.model_constants['layers']

    with open(output_path, 'w') as f:
        # Header guard
        guard = f"{model_name.upper()}_H"
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include \"maf.h\"\n\n")

        # Model metadata
        f.write(f"/* MAF Model: {model_name} */\n")
        f.write(f"/* n_flows={n_flows}, param_dim={param_dim}, ")
        f.write(f"feature_dim={feature_dim}, hidden_units={hidden_units} */\n\n")

        # Export masks and permutations
        f.write("/* Masks and Permutations */\n")

        # M1 (all layers concatenated)
        f.write(f"static const float {model_name}_M1_data[] = {{\n")
        for k in range(n_flows):
            M1 = layers[k]['M1']  # Shape: (hidden_units, param_dim)
            f.write(f"    /* Layer {k} */\n    ")
            f.write(", ".join(f"{v:.6f}f" for v in M1.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # M2 (all layers concatenated)
        f.write(f"static const float {model_name}_M2_data[] = {{\n")
        for k in range(n_flows):
            M2 = layers[k]['M2']  # Shape: (param_dim, hidden_units)
            f.write(f"    /* Layer {k} */\n    ")
            f.write(", ".join(f"{v:.6f}f" for v in M2.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # Permutations
        f.write(f"static const uint16_t {model_name}_perm_data[] = {{\n")
        for k in range(n_flows):
            perm = layers[k]['perm']
            f.write(f"    /* Layer {k} */ ")
            f.write(", ".join(str(v) for v in perm))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # Inverse permutations
        f.write(f"static const uint16_t {model_name}_inv_perm_data[] = {{\n")
        for k in range(n_flows):
            inv_perm = layers[k]['inv_perm']
            f.write(f"    /* Layer {k} */ ")
            f.write(", ".join(str(v) for v in inv_perm))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # Export weights
        f.write("/* Layer Weights */\n")

        # W1y
        f.write(f"static const float {model_name}_W1y_data[] = {{\n")
        for k in range(n_flows):
            W1y = model.weights[f'W1y_{k}']
            f.write(f"    /* Layer {k} */\n    ")
            f.write(", ".join(f"{v:.6f}f" for v in W1y.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # W1c
        f.write(f"static const float {model_name}_W1c_data[] = {{\n")
        for k in range(n_flows):
            W1c = model.weights[f'W1c_{k}']
            if W1c.size > 0:
                f.write(f"    /* Layer {k} */\n    ")
                f.write(", ".join(f"{v:.6f}f" for v in W1c.flatten()))
            else:
                f.write(f"    /* Layer {k} */ 0.0f")
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # b1
        f.write(f"static const float {model_name}_b1_data[] = {{\n")
        for k in range(n_flows):
            b1 = model.weights[f'b1_{k}']
            f.write(f"    /* Layer {k} */ ")
            f.write(", ".join(f"{v:.6f}f" for v in b1.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # W2
        f.write(f"static const float {model_name}_W2_data[] = {{\n")
        for k in range(n_flows):
            W2 = model.weights[f'W2_{k}']
            f.write(f"    /* Layer {k} */\n    ")
            f.write(", ".join(f"{v:.6f}f" for v in W2.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # W2c
        f.write(f"static const float {model_name}_W2c_data[] = {{\n")
        for k in range(n_flows):
            W2c = model.weights[f'W2c_{k}']
            if W2c.size > 0:
                f.write(f"    /* Layer {k} */\n    ")
                f.write(", ".join(f"{v:.6f}f" for v in W2c.flatten()))
            else:
                f.write(f"    /* Layer {k} */ 0.0f")
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # b2
        f.write(f"static const float {model_name}_b2_data[] = {{\n")
        for k in range(n_flows):
            b2 = model.weights[f'b2_{k}']
            f.write(f"    /* Layer {k} */ ")
            f.write(", ".join(f"{v:.6f}f" for v in b2.flatten()))
            if k < n_flows - 1:
                f.write(",\n")
        f.write("\n};\n\n")

        # Create weights structure
        f.write(f"static const maf_weights_t {model_name}_weights = {{\n")
        f.write(f"    .n_flows = {n_flows},\n")
        f.write(f"    .param_dim = {param_dim},\n")
        f.write(f"    .feature_dim = {feature_dim},\n")
        f.write(f"    .hidden_units = {hidden_units},\n")
        f.write(f"    .M1_data = {model_name}_M1_data,\n")
        f.write(f"    .M2_data = {model_name}_M2_data,\n")
        f.write(f"    .perm_data = {model_name}_perm_data,\n")
        f.write(f"    .inv_perm_data = {model_name}_inv_perm_data,\n")
        f.write(f"    .W1y_data = {model_name}_W1y_data,\n")
        f.write(f"    .W1c_data = {model_name}_W1c_data,\n")
        f.write(f"    .b1_data = {model_name}_b1_data,\n")
        f.write(f"    .W2_data = {model_name}_W2_data,\n")
        f.write(f"    .W2c_data = {model_name}_W2c_data,\n")
        f.write(f"    .b2_data = {model_name}_b2_data\n")
        f.write("};\n\n")

        f.write(f"#endif /* {guard} */\n")

    print(f"Exported MAF model to {output_path}")
    print(f"  n_flows: {n_flows}")
    print(f"  param_dim: {param_dim}")
    print(f"  feature_dim: {feature_dim}")
    print(f"  hidden_units: {hidden_units}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and export MAF model to C header")
    parser.add_argument("--dataset", choices=["banana", "student_t", "moons"],
                        default="banana", help="Test dataset to use")
    parser.add_argument("--n-flows", type=int, default=3, help="Number of flow layers")
    parser.add_argument("--hidden-units", type=int, default=32, help="Hidden units per layer")
    parser.add_argument("--n-samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--n-iter", type=int, default=1000, help="Training iterations")
    parser.add_argument("--output", type=str, default="maf_model.h", help="Output header file")
    parser.add_argument("--name", type=str, default="maf_model", help="Model name in C")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Training MAF model on '{args.dataset}' dataset...")
    print(f"  n_flows={args.n_flows}, hidden_units={args.hidden_units}")

    # Generate training data
    params, features = generate_test_data(args.dataset, args.n_samples, seed=args.seed)

    # Create and train model
    model = MAFEstimator(
        param_dim=params.shape[1],
        feature_dim=features.shape[1],
        n_flows=args.n_flows,
        hidden_units=args.hidden_units
    )

    print(f"Training for {args.n_iter} iterations...")
    model.train(params, features, n_iter=args.n_iter, seed=args.seed)

    print(f"Final loss: {model.loss_history[-1]:.4f}")

    # Export to C header
    export_maf_to_header(model, args.output, args.name)

    # Test sampling
    print("\nTesting sampling in Python...")
    rng = np.random.RandomState(args.seed)
    test_features = features[:3]
    samples = model.sample(test_features, n_samples=10, rng=rng)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {samples.mean(axis=(0, 1))}")
    print(f"Sample std: {samples.std(axis=(0, 1))}")
