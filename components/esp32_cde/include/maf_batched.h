#ifndef MAF_BATCHED_H
#define MAF_BATCHED_H

#include "maf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Batched API for optimized training.
 * 
 * NOTE: Data layout is Column-Major (Structure of Arrays) to facilitate SIMD.
 * Input arrays should be [Dim, BatchSize].
 * That is, all samples for dimension 0 are contiguous, followed by all samples for dimension 1.
 */

typedef struct maf_workspace_batched_t maf_workspace_batched_t;
typedef struct maf_cache_batched_t maf_cache_batched_t;

/**
 * @brief Create batched workspace.
 * @param model The model.
 * @param batch_size The fixed batch size (must match compile-time constant if hardcoded).
 */
maf_workspace_batched_t* maf_create_workspace_batched(const maf_model_t* model, int batch_size);
void maf_free_workspace_batched(maf_workspace_batched_t* ws);

maf_cache_batched_t* maf_create_cache_batched(const maf_model_t* model, int batch_size);
void maf_free_cache_batched(maf_cache_batched_t* cache);

/**
 * @brief Batched Forward Pass
 * @param features_batch Input features [FeatureDim, BatchSize]
 * @param params_batch Input parameters [ParamDim, BatchSize]
 * @param log_probs_out Output log probabilities [BatchSize]
 */
void maf_forward_train_batch(const maf_model_t* model,
                             maf_workspace_batched_t* ws,
                             maf_cache_batched_t* cache,
                             const float* features_batch,
                             const float* params_batch,
                             float* log_probs_out,
                             int batch_size);

/**
 * @brief Batched Backward Pass
 * Accumulates gradients into 'grad' (which is shared/summed).
 * @param features_batch Input features [FeatureDim, BatchSize]
 * @param params_batch Input parameters [ParamDim, BatchSize]
 */
int maf_backward_batch(const maf_model_t* model,
                       const maf_cache_batched_t* cache,
                       maf_grad_t* grad,
                       const float* features_batch,
                       const float* params_batch,
                       int batch_size);

/**
 * @brief Vectorized Adam Step
 * Performs Adam update on all parameters using vectorized kernels.
 * 'batch_size' argument is unused for the step itself but kept for consistency? 
 * Actually Adam step is over parameters, not batch.
 */
void maf_adam_step_vectorized(maf_model_t* model, 
                              maf_adam_t* adam, 
                              const maf_grad_t* grad);

#ifdef __cplusplus
}
#endif

#endif // MAF_BATCHED_H
