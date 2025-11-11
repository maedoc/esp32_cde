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
 * @param features Conditioning features [feature_dim]
 * @param params Parameter values [param_dim]
 * @return Log probability value
 */
float maf_log_prob(const maf_model_t* model,
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
 * @param y Input vector [param_dim]
 * @param context Feature/context vector [feature_dim]
 * @param mu_out Output mean [param_dim]
 * @param alpha_out Output log-scale [param_dim]
 */
void maf_made_forward(const maf_layer_t* layer,
                      const float* y,
                      const float* context,
                      float* mu_out,
                      float* alpha_out);

/**
 * @brief Apply inverse transformation for one layer during sampling
 *
 * @param layer Layer parameters
 * @param y_perm Permuted input from previous layer [param_dim]
 * @param context Feature vector [feature_dim]
 * @param x_out Output (unpermuted) [param_dim]
 */
void maf_inverse_layer(const maf_layer_t* layer,
                       const float* y_perm,
                       const float* context,
                       float* x_out);

#ifdef __cplusplus
}
#endif

#endif /* MAF_H */
