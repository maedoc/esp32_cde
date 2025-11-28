import sys
import json
import logging
import traceback
import argparse
import numpy as np
from cde_training import MAFEstimator
import os

# Configure logging to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("maf_mcp")

def handle_initialize(params):
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {
                "listChanged": False
            }
        },
        "serverInfo": {
            "name": "maf-mcp-server",
            "version": "0.1.0"
        }
    }

def handle_tools_list(params):
    return {
        "tools": [
            {
                "name": "train_maf",
                "description": "Train a Masked Autoregressive Flow (MAF) model on provided feature and parameter data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "features_file": {"type": "string", "description": "Path to CSV file containing feature data (X)."},
                        "params_file": {"type": "string", "description": "Path to CSV file containing target parameter data (Y)."},
                        "output_model_file": {"type": "string", "description": "Path to save the trained model (Python pickle)."},
                        "n_flows": {"type": "integer", "default": 3, "description": "Number of flow layers."},
                        "hidden_units": {"type": "integer", "default": 16, "description": "Number of hidden units per layer."},
                        "iterations": {"type": "integer", "default": 2000, "description": "Number of training iterations."}
                    },
                    "required": ["features_file", "params_file"]
                }
            },
            {
                "name": "sample_maf",
                "description": "Generate samples from a trained MAF model given features.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_file": {"type": "string", "description": "Path to the trained model file."}, 
                        "features_file": {"type": "string", "description": "Path to CSV file containing features to condition on."}, 
                        "output_file": {"type": "string", "description": "Path to save generated samples CSV."}, 
                        "n_samples": {"type": "integer", "default": 10, "description": "Number of samples per feature."}
                    },
                    "required": ["model_file", "features_file", "output_file"]
                }
            }
        ]
    }

def train_maf(features_file, params_file, output_model_file="model.pkl", n_flows=3, hidden_units=16, iterations=2000):
    logger.info(f"Training MAF: features={features_file}, params={params_file}, flows={n_flows}, hidden={hidden_units}")
    
    try:
        X = np.loadtxt(features_file, delimiter=",", skiprows=1)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        Y = np.loadtxt(params_file, delimiter=",", skiprows=1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)
        
        if len(X) != len(Y):
            return f"Error: Data length mismatch. Features: {len(X)}, Params: {len(Y)}"

        feature_dim = X.shape[1]
        param_dim = Y.shape[1]

        model = MAFEstimator(n_flows=n_flows, hidden_units=hidden_units, param_dim=param_dim, feature_dim=feature_dim)
        
        # Note: train takes (params, features)
        model.train(Y, X, n_iter=iterations, use_tqdm=False)
        
        import pickle
        with open(output_model_file, 'wb') as f:
            pickle.dump(model, f)
            
        final_loss = model.loss_history[-1] if model.loss_history else 'N/A'
        return f"Model trained and saved to {output_model_file}. Final loss: {final_loss}"
        
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Training failed: {str(e)}"

def sample_maf(model_file, features_file, output_file, n_samples=10):
    logger.info(f"Sampling MAF: model={model_file}, features={features_file}")
    try:
        import pickle
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            
        features = np.loadtxt(features_file, delimiter=",", skiprows=1)
        if features.ndim == 1: features = features.reshape(-1, 1)
        
        import autograd.numpy as anp
        rng = anp.random.RandomState(42)
        
        samples = model.sample(features, n_samples, rng)
        # samples shape: (N_features, n_samples, param_dim)
        
        N = samples.shape[0]
        S = samples.shape[1]
        D = samples.shape[2]
        
        with open(output_file, 'w') as f:
            header = "feature_idx,sample_idx," + ",".join([f"p{d}" for d in range(D)]) + "\n"
            f.write(header)
            for i in range(N):
                for s in range(S):
                    row = [str(i), str(s)] + [str(x) for x in samples[i,s,:]]
                    f.write(",".join(row) + "\n")
                    
        return f"Generated {N*S} samples to {output_file}"

    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Sampling failed: {str(e)}"

def handle_tools_call(params):
    name = params.get("name")
    args = params.get("arguments", {})
    
    if name == "train_maf":
        result = train_maf(**args)
        return {"content": [{"type": "text", "text": str(result)}]}
    elif name == "sample_maf":
        result = sample_maf(**args)
        return {"content": [{"type": "text", "text": str(result)}]}
    else:
        raise ValueError(f"Unknown tool: {name}")

def main():
    logger.info("Starting MAF MCP Server...")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            
            response = {"jsonrpc": "2.0", "id": req_id}
            
            try:
                if method == "initialize":
                    response["result"] = handle_initialize(params)
                elif method == "tools/list":
                    response["result"] = handle_tools_list(params)
                elif method == "tools/call":
                    response["result"] = handle_tools_call(params)
                elif method == "notifications/initialized":
                     continue 
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue 
                    
            except Exception as e:
                logger.error(f"Error handling {method}: {e}")
                traceback.print_exc()
                response["error"] = {"code": -32000, "message": str(e)}
            
            if req_id is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            continue
        except Exception as e:
            logger.error(f"Fatal loop error: {e}")
            break

if __name__ == "__main__":
    main()