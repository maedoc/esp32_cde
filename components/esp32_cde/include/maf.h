/**
 * @file maf.h
 * @brief Masked Autoregressive Flow (MAF) Inference Library
 *
 * Standalone C implementation for MAF conditional density estimation inference.
 * Supports loading pre-trained models and generating samples.
 *
 * Architecture:
 * - Multiple flow layers (MADE blocks) with permutations
 * - Each layer transforms through: y_i = (x_i - mu_i) * exp(-alpha_i)
 * - Sampling inverts: x_i = y_i * exp(alpha_i) + mu_i
 * - Context/features condition the transformation parameters
 *
 * Usage:
 *   1. Train model in Python
 *   2. Export weights to C header using provided script
 *   3. Load model: maf_model_t* model = maf_load_model(&exported_weights);
 *   4. Sample: maf_sample(model, features, n_samples, samples_out);
 *   5. Cleanup: maf_free_model(model);
 */

#ifndef MAF_H
#define MAF_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * Data Structures
 * ========================================================================== */

/**
 * @brief MAF layer parameters (one MADE block)
 */
typedef struct {
    uint16_t param_dim;        ///< Dimensionality of parameters (D)
    uint16_t feature_dim;      ///< Dimensionality of features/context (C)
    uint16_t hidden_units;     ///< Number of hidden units (H)

    /* Model constants (masks and permutations) */
    float* M1;                 ///< Input-to-hidden mask [H x D]
    float* M2;                 ///< Hidden-to-output mask [D x H]
    uint16_t* perm;            ///< Output permutation [D]
    uint16_t* inv_perm;        ///< Inverse permutation [D]

    /* Trainable weights */
    float* W1y;                ///< Input-to-hidden weights [H x D]
    float* W1c;                ///< Context-to-hidden weights [H x C]
    float* b1;                 ///< Hidden bias [H]
    float* W2;                 ///< Hidden-to-output weights [2*D x H]
    float* W2c;                ///< Context-to-output weights [2*D x C]
    float* b2;                 ///< Output bias [2*D]
} maf_layer_t;

/**
 * @brief Complete MAF model
 */
typedef struct {
    uint16_t n_flows;          ///< Number of flow layers
    uint16_t param_dim;        ///< Dimensionality of parameters
    uint16_t feature_dim;      ///< Dimensionality of features
    maf_layer_t* layers;       ///< Array of flow layers
} maf_model_t;

/**
 * @brief Workspace for intermediate calculations (avoids dynamic allocation)
 */
typedef struct {
    float* h;          ///< Hidden units buffer [max_hidden_units]
    float* out;        ///< Output buffer [2 * param_dim]
    float* u;          ///< Input/latent buffer [param_dim]
    float* u_perm;     ///< Permuted buffer [param_dim]
    float* mu;         ///< Mean buffer [param_dim]
    float* alpha;      ///< Log-scale buffer [param_dim]
    float* x;          ///< Sampling buffer [param_dim]
    float* y_perm;     ///< Sampling permuted buffer [param_dim]
} maf_workspace_t;

/**
 * @brief Layer cache for backpropagation
 */
typedef struct {
    float* input;   ///< Layer input (permuted) [D]
    float* h;       ///< Hidden state (post-tanh) [H]
    float* mu;      ///< Output mu [D]
    float* alpha;   ///< Output alpha [D]
} maf_layer_cache_t;

/**
 * @brief Model cache for backpropagation
 */
typedef struct {
    uint16_t n_flows;
    maf_layer_cache_t* layers;
} maf_cache_t;

/**
 * @brief Gradients for one layer
 */
typedef struct {
    float* dW1y;    ///< [H x D]
    float* dW1c;    ///< [H x C]
    float* db1;     ///< [H]
    float* dW2;     ///< [2*D x H]
    float* dW2c;    ///< [2*D x C]
    float* db2;     ///< [2*D]
} maf_layer_grad_t;

/**
 * @brief Gradients for the full model
 */
typedef struct {
    uint16_t n_flows;
    maf_layer_grad_t* layers;
} maf_grad_t;

/**
 * @brief Adam optimizer state for one layer
 */
typedef struct {
    /* First moments (m) */
    float* mW1y; float* mW1c; float* mb1;
    float* mW2;  float* mW2c; float* mb2;
    
    /* Second moments (v) */
    float* vW1y; float* vW1c; float* vb1;
    float* vW2;  float* vW2c; float* vb2;
} maf_layer_adam_t;

/**
 * @brief Adam optimizer state for full model
 */
typedef struct {
    uint16_t n_flows;
    uint32_t t;         ///< Time step
    float beta1;        ///< Decay rate for first moment (e.g., 0.9)
    float beta2;        ///< Decay rate for second moment (e.g., 0.999)
    float epsilon;      ///< Stability term (e.g., 1e-8)
    float lr;           ///< Learning rate
    maf_layer_adam_t* layers;
} maf_adam_t;

/**
 * @brief Model weights for initialization (Python export format)
 */
typedef struct {
    uint16_t n_flows;
    uint16_t param_dim;
    uint16_t feature_dim;
    uint16_t hidden_units;

    /* Pointers to layer data (flattened arrays) */
    const float* M1_data;      ///< All M1 masks concatenated
    const float* M2_data;      ///< All M2 masks concatenated
    const uint16_t* perm_data; ///< All permutations concatenated
    const uint16_t* inv_perm_data; ///< All inverse permutations

    const float* W1y_data;     ///< All W1y weights concatenated
    const float* W1c_data;     ///< All W1c weights concatenated
    const float* b1_data;      ///< All b1 biases concatenated
    const float* W2_data;      ///< All W2 weights concatenated
    const float* W2c_data;     ///< All W2c weights concatenated
    const float* b2_data;      ///< All b2 biases concatenated
} maf_weights_t;

/* =============================================================================
 * Core API
 * ========================================================================== */

/**
 * @brief Load a MAF model from exported weights
 *
 * @param weights Pointer to exported model weights structure
 * @return Pointer to allocated model, or NULL on failure
 */
maf_model_t* maf_load_model(const maf_weights_t* weights);

/**
 * @brief Free a MAF model and all associated memory
 *
 * @param model Pointer to model to free
 */
void maf_free_model(maf_model_t* model);

/**
 * @brief Create a workspace for the model
 *
 * Allocates buffers based on the maximum dimensions of the model layers.
 * Thread-safety: Each thread should have its own workspace.
 *
 * @param model The model to create a workspace for
 * @return Pointer to allocated workspace, or NULL on failure
 */
maf_workspace_t* maf_create_workspace(const maf_model_t* model);

/**
 * @brief Free a workspace
 *
 * @param ws Pointer to workspace to free
 */
void maf_free_workspace(maf_workspace_t* ws);

/* =============================================================================
 * Training API
 * ========================================================================== */

/**
 * @brief Create a cache for storing forward pass activations
 * 
 * @param model Model structure
 * @return Pointer to allocated cache, or NULL on failure
 */
maf_cache_t* maf_create_cache(const maf_model_t* model);

/**
 * @brief Free a cache
 * 
 * @param cache Pointer to cache to free
 */
void maf_free_cache(maf_cache_t* cache);

/**
 * @brief Create a gradient structure
 * 
 * @param model Model structure
 * @return Pointer to allocated gradient struct, or NULL on failure
 */
maf_grad_t* maf_create_grad(const maf_model_t* model);

/**
 * @brief Free a gradient structure
 * 
 * @param grad Pointer to gradient to free
 */
void maf_free_grad(maf_grad_t* grad);

/**
 * @brief Clear gradients (set to zero)
 * 
 * @param model Model structure (to know sizes)
 * @param grad Gradient structure to clear
 */
void maf_zero_grad(const maf_model_t* model, maf_grad_t* grad);

/**
 * @brief Create Adam optimizer state
 * 
 * @param model Model to optimize
 * @param lr Learning rate
 * @param beta1 Beta1 (default 0.9)
 * @param beta2 Beta2 (default 0.999)
 * @param epsilon Epsilon (default 1e-8)
 * @return Pointer to Adam state, or NULL on failure
 */
maf_adam_t* maf_create_adam(const maf_model_t* model, float lr, float beta1, float beta2, float epsilon);

/**
 * @brief Free Adam optimizer state
 * 
 * @param adam Pointer to Adam state
 */
void maf_free_adam(maf_adam_t* adam);

/**
 * @brief Forward pass for training (populates cache)
 *
 * @param model Trained MAF model
 * @param ws Workspace
 * @param cache Cache to populate
 * @param features Conditioning features
 * @param params Parameter values (input)
 * @return Log probability value
 */
float maf_forward_train(const maf_model_t* model,
                        maf_workspace_t* ws,
                        maf_cache_t* cache,
                        const float* features,
                        const float* params);

/**
 * @brief Backward pass (calculates gradients)
 *
 * @param model Trained MAF model
 * @param cache Populated cache from forward pass
 * @param grad Gradient structure to accumulate into
 * @param features Conditioning features
 * @param params Parameter values (original input)
 * @return 0 on success, error code otherwise
 */
int maf_backward(const maf_model_t* model,
                 const maf_cache_t* cache,
                 maf_grad_t* grad,
                 const float* features,
                 const float* params);

/**
 * @brief Update model weights using SGD
 * 
 * @param model Model to update
 * @param grad Gradients to use
 * @param lr Learning rate
 */
void maf_sgd_step(maf_model_t* model, const maf_grad_t* grad, float lr);

/**
 * @brief Update model weights using Adam
 * 
 * @param model Model to update
 * @param adam Adam optimizer state
 * @param grad Gradients to use
 */
void maf_adam_step(maf_model_t* model, maf_adam_t* adam, const maf_grad_t* grad);

/* =============================================================================
 * Inference API
 * ========================================================================== */

/**
 * @brief Generate samples from the conditional distribution p(y|features)
 *
 * @param model Trained MAF model
 * @param features Conditioning features [feature_dim]
 * @param n_samples Number of samples to generate
 * @param samples_out Output buffer [n_samples x param_dim]
 * @param seed Random seed for reproducibility
 * @return 0 on success, negative error code on failure
 */
int maf_sample(const maf_model_t* model,
               const float* features,
               uint32_t n_samples,
               float* samples_out,
               uint32_t seed);

/**
 * @brief Generate samples from provided base noise (for testing/validation)
 *
 * This function applies the MAF inverse transformation to provided base noise,
 * allowing deterministic testing independent of RNG implementation.
 *
 * @param model Trained MAF model
 * @param features Conditioning features [feature_dim]
 * @param base_noise Base noise samples [n_samples x param_dim], z ~ N(0, I)
 * @param n_samples Number of samples
 * @param samples_out Output buffer [n_samples x param_dim]
 * @return 0 on success, negative error code on failure
 */
int maf_sample_from_noise(const maf_model_t* model,
                          const float* features,
                          const float* base_noise,
                          uint32_t n_samples,
                          float* samples_out);

/**
 * @brief Compute log probability log p(params|features)
 *
 * @param model Trained MAF model
 * @param ws Workspace for temporary buffers
 * @param features Conditioning features [feature_dim]
 * @param params Parameter values [param_dim]
 * @return Log probability value
 */
float maf_log_prob(const maf_model_t* model,
                   maf_workspace_t* ws,
                   const float* features,
                   const float* params);

/**
 * @brief Get memory usage of a model in bytes
 *
 * @param model MAF model
 * @return Total memory usage in bytes
 */
size_t maf_get_memory_usage(const maf_model_t* model);

/* =============================================================================
 * Layer Operations (Internal but exposed for testing)
 * ========================================================================== */

/**
 * @brief MADE forward pass: compute mu and alpha given input and context
 *
 * @param layer Layer parameters
 * @param ws Workspace (uses h and out buffers)
 * @param y Input vector [param_dim]
 * @param context Feature/context vector [feature_dim]
 * @param mu_out Output mean [param_dim]
 * @param alpha_out Output log-scale [param_dim]
 */
void maf_made_forward(const maf_layer_t* layer,
                      maf_workspace_t* ws,
                      const float* y,
                      const float* context,
                      float* mu_out,
                      float* alpha_out);

/**
 * @brief Apply inverse transformation for one layer during sampling
 *
 * @param layer Layer parameters
 * @param ws Workspace
 * @param y_perm Permuted input from previous layer [param_dim]
 * @param context Feature vector [feature_dim]
 * @param x_out Output (unpermuted) [param_dim]
 */
void maf_inverse_layer(const maf_layer_t* layer,
                       maf_workspace_t* ws,
                       const float* y_perm,
                       const float* context,
                       float* x_out);

#ifdef __cplusplus
}
#endif

#endif /* MAF_H */
