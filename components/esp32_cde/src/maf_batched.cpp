#include "maf_batched.h"
#include "kernels.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <iostream>

// We fix the batch size for the kernel template to ensure aggressive unrolling/SIMD
#define BATCH_SIZE_KERNEL 32

using namespace tvbk;

struct maf_workspace_batched_t {
    float* h;       // [max_H * B]
    float* out;     // [2*D * B]
    float* u;       // [D * B]
    float* u_perm;  // [D * B]
    float* mu;      // [D * B]
    float* alpha;   // [D * B]
    
    // Backward pass temporaries
    float* d_out;   // [2*D * B]
    float* d_h;     // [max_H * B]
    float* d_u_in;  // [D * B]
    float* delta;   // [D * B]
    float* prev_delta; // [D * B]

    int max_H;
    int D;
    int batch_size;
};

struct maf_layer_cache_batched_t {
    float* input;   // [D * B]
    float* h;       // [H * B]
    float* mu;      // [D * B]
    float* alpha;   // [D * B]
};

struct maf_cache_batched_t {
    int n_flows;
    int batch_size;
    maf_layer_cache_batched_t* layers;
};

extern "C" {

maf_workspace_batched_t* maf_create_workspace_batched(const maf_model_t* model, int batch_size) {
    if (!model) return nullptr;
    assert(batch_size == BATCH_SIZE_KERNEL); // Enforce fixed size for now

    maf_workspace_batched_t* ws = new maf_workspace_batched_t();
    ws->batch_size = batch_size;
    ws->D = model->param_dim;
    
    ws->max_H = 0;
    for (int k = 0; k < model->n_flows; k++) {
        if (model->layers[k].hidden_units > ws->max_H) {
            ws->max_H = model->layers[k].hidden_units;
        }
    }
    int max_H = ws->max_H;
    int D = ws->D;
    int B = batch_size;

    ws->h = new float[max_H * B];
    ws->out = new float[2 * D * B];
    ws->u = new float[D * B];
    ws->u_perm = new float[D * B];
    ws->mu = new float[D * B];
    ws->alpha = new float[D * B];

    ws->d_out = new float[2 * D * B];
    ws->d_h = new float[max_H * B];
    ws->d_u_in = new float[D * B];
    ws->delta = new float[D * B];
    ws->prev_delta = new float[D * B];

    return ws;
}

void maf_free_workspace_batched(maf_workspace_batched_t* ws) {
    if (!ws) return;
    delete[] ws->h;
    delete[] ws->out;
    delete[] ws->u;
    delete[] ws->u_perm;
    delete[] ws->mu;
    delete[] ws->alpha;
    delete[] ws->d_out;
    delete[] ws->d_h;
    delete[] ws->d_u_in;
    delete[] ws->delta;
    delete[] ws->prev_delta;
    delete ws;
}

maf_cache_batched_t* maf_create_cache_batched(const maf_model_t* model, int batch_size) {
    if (!model) return nullptr;
    
    maf_cache_batched_t* cache = new maf_cache_batched_t();
    cache->n_flows = model->n_flows;
    cache->batch_size = batch_size;
    cache->layers = new maf_layer_cache_batched_t[model->n_flows];

    for (int k = 0; k < model->n_flows; k++) {
        int D = model->layers[k].param_dim;
        int H = model->layers[k].hidden_units;
        int B = batch_size;
        
        cache->layers[k].input = new float[D * B];
        cache->layers[k].h = new float[H * B];
        cache->layers[k].mu = new float[D * B];
        cache->layers[k].alpha = new float[D * B];
    }
    return cache;
}

void maf_free_cache_batched(maf_cache_batched_t* cache) {
    if (!cache) return;
    for (int k = 0; k < cache->n_flows; k++) {
        delete[] cache->layers[k].input;
        delete[] cache->layers[k].h;
        delete[] cache->layers[k].mu;
        delete[] cache->layers[k].alpha;
    }
    delete[] cache->layers;
    delete cache;
}

// =============================================================================
// Forward
// =============================================================================

// Internal MADE forward
static void maf_made_forward_batch(const maf_layer_t* layer,
                                   maf_workspace_batched_t* ws,
                                   const float* y_batch,      // [D * B]
                                   const float* context_batch // [C * B]
                                   ) 
{
    const int B = BATCH_SIZE_KERNEL;
    int D = layer->param_dim;
    int C = layer->feature_dim;
    int H = layer->hidden_units;
    
    float* h_batch = ws->h;     // [H * B]
    float* out_batch = ws->out; // [2D * B]
    
    // h = tanh((y @ W1y.T) * M1 + (ctx @ W1c.T) + b1)
    
    for (int i = 0; i < H; i++) {
        float* h_row = &h_batch[i * B];
        
        // Initialize with bias
        fill<B>(h_row, layer->b1[i]);
        
        // y @ W1y.T * M1
        // For each input j, if M1[i,j] != 0, add W[i,j] * y[j]
        for (int j = 0; j < D; j++) {
            float m1 = layer->M1[i * D + j];
            if (m1 != 0.0f) {
                float w = layer->W1y[i * D + j] * m1;
                const float* y_row = &y_batch[j * B];
                inc<B>(h_row, (float*)y_row, w);
            }
        }
        
        // context @ W1c.T
        for (int j = 0; j < C; j++) {
            float w = layer->W1c[i * C + j];
            const float* c_row = &context_batch[j * B];
            inc<B>(h_row, (float*)c_row, w);
        }
        
        // Activation
        ktanhf<B>(h_row, h_row);
    }
    
    // out = (h @ W2.T) * M2 + (ctx @ W2c.T) + b2
    for (int i = 0; i < 2 * D; i++) {
        float* out_row = &out_batch[i * B];
        
        // Init with bias
        fill<B>(out_row, layer->b2[i]);
        
        // h @ W2.T * M2
        int d_idx = i % D;
        for (int j = 0; j < H; j++) {
            float m2 = layer->M2[d_idx * H + j];
            if (m2 != 0.0f) {
                float w = layer->W2[i * H + j] * m2;
                const float* h_row = &h_batch[j * B];
                inc<B>(out_row, (float*)h_row, w);
            }
        }
        
        // context @ W2c.T
        for (int j = 0; j < C; j++) {
            float w = layer->W2c[i * C + j];
            const float* c_row = &context_batch[j * B];
            inc<B>(out_row, (float*)c_row, w);
        }
    }
}

void maf_forward_train_batch(const maf_model_t* model,
                             maf_workspace_batched_t* ws,
                             maf_cache_batched_t* cache,
                             const float* features_batch,
                             const float* params_batch,
                             float* log_probs_out,
                             int batch_size) 
{
    assert(batch_size == BATCH_SIZE_KERNEL);
    const int B = BATCH_SIZE_KERNEL;
    int D = model->param_dim;
    
    // Init u with params
    std::memcpy(ws->u, params_batch, D * B * sizeof(float));
    
    // Init log_det accumulator to 0
    // We don't have a separate log_det buffer, we'll accumulate into log_probs_out directly
    // Or we can track log_det separately. Let's just use log_probs_out as accumulation.
    // We start with base_logp.
    
    // For intermediate u updates: u = (u - mu) * exp(-alpha)
    
    for (int k = 0; k < model->n_flows; k++) {
        const maf_layer_t* layer = &model->layers[k];
        maf_layer_cache_batched_t* lcache = &cache->layers[k];
        
        // Apply permutation: u_perm = u[perm]
        for (int i = 0; i < D; i++) {
            int src_idx = layer->perm[i];
            float* dest_row = &ws->u_perm[i * B];
            float* src_row = &ws->u[src_idx * B];
            load<B>(dest_row, src_row);
        }
        
        // Store input to cache
        std::memcpy(lcache->input, ws->u_perm, D * B * sizeof(float));
        
        // Forward pass
        maf_made_forward_batch(layer, ws, ws->u_perm, features_batch);
        
        // Split out into mu/alpha
        for (int i = 0; i < D; i++) {
            float* out_mu_row = &ws->out[i * B];
            float* out_alpha_row = &ws->out[(D + i) * B];
            
            float* mu_row = &ws->mu[i * B];
            float* alpha_row = &ws->alpha[i * B];
            
            // Copy/Process
            load<B>(mu_row, out_mu_row);
            
            // Clip alpha: alpha = min(max(out_alpha, -7), 7)
            // Not implemented in kernels, do manually loop or add kernel
            // For simplicity, assume standard loop for clipping or add kernel later
            // Let's use loop for safety now
            for(int b=0; b<B; ++b) {
                 float val = out_alpha_row[b];
                 if (val < -7.0f) val = -7.0f;
                 if (val > 7.0f) val = 7.0f;
                 alpha_row[b] = val;
            }
        }
        
        // Store activations
        std::memcpy(lcache->h, ws->h, layer->hidden_units * B * sizeof(float));
        std::memcpy(lcache->mu, ws->mu, D * B * sizeof(float));
        std::memcpy(lcache->alpha, ws->alpha, D * B * sizeof(float));
        
        // Update u and log_det
        // u = (u_perm - mu) * exp(-alpha)
        // log_det -= alpha
        for (int i = 0; i < D; i++) {
            float* u_row = &ws->u[i * B]; // In-place update? No, u buffer is reused.
            // Actually u is [D, B]. We can write back to it.
            
            float* u_perm_row = &ws->u_perm[i * B];
            float* mu_row = &ws->mu[i * B];
            float* alpha_row = &ws->alpha[i * B];
            
            for (int b = 0; b < B; b++) {
                 u_row[b] = (u_perm_row[b] - mu_row[b]) * expf(-alpha_row[b]);
            }
        }
    }
    
    // Compute Final Log Prob
    // Base dist: N(0, I)
    // logp = -0.5 * u^2 - 0.5 * log(2pi) + sum(log_det)
    // log_det = sum(-alpha)
    
    // Initialize log_probs_out with constant
    float const_term = -0.5f * D * logf(2.0f * M_PI_F);
    for (int b = 0; b < B; b++) log_probs_out[b] = const_term;
    
    // Add base logp (-0.5 * u^2)
    for (int i = 0; i < D; i++) {
        float* u_row = &ws->u[i * B];
        for (int b = 0; b < B; b++) {
            log_probs_out[b] -= 0.5f * u_row[b] * u_row[b];
        }
    }
    
    // Add log det terms from all layers
    for (int k = 0; k < model->n_flows; k++) {
        maf_layer_cache_batched_t* lcache = &cache->layers[k];
        for (int i = 0; i < D; i++) {
            float* alpha_row = &lcache->alpha[i * B];
            for (int b = 0; b < B; b++) {
                log_probs_out[b] -= alpha_row[b];
            }
        }
    }
}

// =============================================================================
// Backward
// =============================================================================

static void maf_layer_backward_batch(const maf_layer_t* layer,
                                     const maf_layer_cache_batched_t* lcache,
                                     maf_layer_grad_t* lgrad, // Shared accumulator
                                     maf_workspace_batched_t* ws,
                                     const float* features_batch,
                                     const float* grad_from_top, // [D, B]
                                     float* grad_to_bottom)      // [D, B]
{
    const int B = BATCH_SIZE_KERNEL;
    int D = layer->param_dim;
    int C = layer->feature_dim;
    int H = layer->hidden_units;
    
    // Clear temps
    float* d_out = ws->d_out; // [2D, B]
    float* d_h = ws->d_h;     // [H, B]
    float* d_u_in = ws->d_u_in; // [D, B]
    
    std::memset(d_out, 0, 2 * D * B * sizeof(float));
    std::memset(d_h, 0, H * B * sizeof(float));
    std::memset(d_u_in, 0, D * B * sizeof(float));
    
    // 1. Gradients wrt mu and alpha
    for (int i = 0; i < D; i++) {
        float* u_in_row = &lcache->input[i * B];
        float* mu_row = &lcache->mu[i * B];
        float* alpha_row = &lcache->alpha[i * B];
        const float* delta_row = &grad_from_top[i * B];
        
        float* d_mu_row = &d_out[i * B];
        float* d_alpha_row = &d_out[(D + i) * B];
        
        // d_mu = delta * (-exp(-alpha))
        // d_alpha = delta * (-(u_in - mu)*exp(-alpha)) + 1
        
        for (int b = 0; b < B; b++) {
            float exp_neg_alpha = expf(-alpha_row[b]);
            float u_out = (u_in_row[b] - mu_row[b]) * exp_neg_alpha;
            
            d_mu_row[b] = delta_row[b] * (-exp_neg_alpha);
            d_alpha_row[b] = delta_row[b] * (-u_out) + 1.0f;
        }
    }
    
    // 2. Backprop Output Layer
    // dW2, db2, dW2c
    for (int i = 0; i < 2 * D; i++) {
        float* delta_row = &d_out[i * B];
        
        // db2 (sum over batch)
        float db2_acc;
        // We just sum delta_row.
        // dot with ones? or manual sum.
        float sum = 0.0f;
        for (int b = 0; b < B; b++) sum += delta_row[b];
        lgrad->db2[i] += sum;
        
        // dW2c = delta * context.T -> sum_b(delta[b] * ctx[b])
        for (int j = 0; j < C; j++) {
            const float* ctx_row = &features_batch[j * B];
            float dot_val;
            dot<B>(&dot_val, delta_row, (float*)ctx_row);
            lgrad->dW2c[i * C + j] += dot_val;
        }
        
        // Backprop to h
        int d_idx = i % D;
        for (int j = 0; j < H; j++) {
            float m2 = layer->M2[d_idx * H + j];
            if (m2 != 0.0f) {
                float w = layer->W2[i * H + j];
                // d_h[j] += delta * w * m2
                float* d_h_row = &d_h[j * B];
                inc<B>(d_h_row, delta_row, w * m2);
                
                // dW2[i,j] = sum_b(delta[b] * h[b]) * m2
                const float* h_row = &lcache->h[j * B];
                float dot_val;
                dot<B>(&dot_val, delta_row, (float*)h_row);
                lgrad->dW2[i * H + j] += dot_val * m2;
            }
        }
    }
    
    // 3. Backprop Tanh
    // d_h = d_h * (1 - h^2)
    for (int i = 0; i < H; i++) {
        float* d_h_row = &d_h[i * B];
        float* h_row = &lcache->h[i * B];
        dtanhf<B>(d_h_row, h_row);
    }
    
    // 4. Backprop Input Layer
    for (int i = 0; i < H; i++) {
        float* delta_row = &d_h[i * B];
        
        // db1
        float sum = 0.0f;
        for(int b=0; b<B; b++) sum += delta_row[b];
        lgrad->db1[i] += sum;
        
        // dW1c
        for (int j = 0; j < C; j++) {
            const float* ctx_row = &features_batch[j * B];
            float dot_val;
            dot<B>(&dot_val, delta_row, (float*)ctx_row);
            lgrad->dW1c[i * C + j] += dot_val;
        }
        
        // dW1y and d_u_in
        for (int j = 0; j < D; j++) {
            float m1 = layer->M1[i * D + j];
            if (m1 != 0.0f) {
                float w = layer->W1y[i * D + j];
                // d_u_in[j] += delta * w * m1
                float* d_u_in_row = &d_u_in[j * B];
                inc<B>(d_u_in_row, delta_row, w * m1);
                
                // dW1y = sum(delta * u_in) * m1
                const float* u_in_row = &lcache->input[j * B];
                float dot_val;
                dot<B>(&dot_val, delta_row, (float*)u_in_row);
                lgrad->dW1y[i * D + j] += dot_val * m1;
            }
        }
    }
    
    // 5. Combine d_u_in (direct path)
    for (int i = 0; i < D; i++) {
        float* d_u_in_row = &d_u_in[i * B];
        const float* delta_row = &grad_from_top[i * B]; // from prev layer
        float* alpha_row = &lcache->alpha[i * B];
        
        // d_u_in += delta_from_top * exp(-alpha)
        for (int b = 0; b < B; b++) {
            d_u_in_row[b] += delta_row[b] * expf(-alpha_row[b]);
        }
    }
    
    // 6. Inverse Permutation
    for (int i = 0; i < D; i++) {
        int dest_idx = layer->perm[i];
        float* dest_row = &grad_to_bottom[dest_idx * B];
        float* src_row = &d_u_in[i * B];
        load<B>(dest_row, src_row);
    }
}

int maf_backward_batch(const maf_model_t* model,
                       const maf_cache_batched_t* cache,
                       maf_grad_t* grad,
                       const float* features_batch,
                       const float* params_batch,
                       int batch_size)
{
    assert(batch_size == BATCH_SIZE_KERNEL);
    const int B = BATCH_SIZE_KERNEL;
    int D = model->param_dim;
    
    maf_workspace_batched_t* ws = maf_create_workspace_batched(model, batch_size); // Inefficient to alloc here? 
    // The API passed ws for forward but not backward.
    // We should probably change API or use a thread-local / persistent workspace.
    // For now, allocate.
    
    // Initial gradient (wrt base dist u)
    // delta = (u - mu) * exp(-alpha)  [Matches forward computation]
    // This is effectively "z" in base distribution space.
    // Wait, forward computed: u = (u_in - mu) * exp(-alpha). This IS z.
    // d(logp)/dz = -z.
    // d(logp)/du = d(logp)/dz * dz/du + d(log_det)/du.
    // But our backward logic propagates d(logp)/du_out -> d(logp)/du_in.
    // At top layer, we have z.
    // J = -0.5 z^2 - 0.5 log 2pi.
    // dJ/dz = -z.
    
    // Reconstruct z from last layer
    const maf_layer_cache_batched_t* last = &cache->layers[model->n_flows - 1];
    for (int i = 0; i < D; i++) {
        float* delta_row = &ws->delta[i * B]; // delta is grad from top
        float* u_in_row = &last->input[i * B];
        float* mu_row = &last->mu[i * B];
        float* alpha_row = &last->alpha[i * B];
        
        for (int b = 0; b < B; b++) {
             float z = (u_in_row[b] - mu_row[b]) * expf(-alpha_row[b]);
             delta_row[b] = z; // Initial gradient is z (for NLL minimization)
        }
    }
    
    // Backprop
    for (int k = model->n_flows - 1; k >= 0; k--) {
        maf_layer_backward_batch(&model->layers[k],
                                 &cache->layers[k],
                                 &grad->layers[k],
                                 ws,
                                 features_batch,
                                 ws->delta,
                                 ws->prev_delta);
        
        // Swap delta buffers
        float* tmp = ws->delta;
        ws->delta = ws->prev_delta;
        ws->prev_delta = tmp;
    }
    
    maf_free_workspace_batched(ws);
    return 0;
}

// =============================================================================
// Adam
// =============================================================================

void maf_adam_step_vectorized(maf_model_t* model, 
                              maf_adam_t* adam, 
                              const maf_grad_t* grad) 
{
    adam->t++;
    float beta1_t = powf(adam->beta1, adam->t);
    float beta2_t = powf(adam->beta2, adam->t);
    float lr = adam->lr;
    float b1 = adam->beta1;
    float b2 = adam->beta2;
    float eps = adam->epsilon;

    // We can vectorise over parameters.
    // Define a lambda or macro to process arrays
    auto update = [&](float* p, float* g, float* m, float* v, int size) {
        // Loop with stride BATCH_SIZE_KERNEL
        for (int i = 0; i < size; i += BATCH_SIZE_KERNEL) {
            int remaining = size - i;
            if (remaining >= BATCH_SIZE_KERNEL) {
                // Can use vector ops?
                // Yes, treat p[i...i+31] as a "batch".
                // m[i] = b1 * m[i] + (1-b1) * g[i]
                // This is: m = m*b1 + g*(1-b1) -> muls, inc
                
                float* m_ptr = m + i;
                float* v_ptr = v + i;
                float* g_ptr = g + i;
                float* p_ptr = p + i;
                
                // m = m * b1
                muls<BATCH_SIZE_KERNEL>(m_ptr, b1);
                // m += g * (1-b1)
                inc<BATCH_SIZE_KERNEL>(m_ptr, g_ptr, 1.0f - b1);
                
                // v = v * b2
                muls<BATCH_SIZE_KERNEL>(v_ptr, b2);
                // v += g^2 * (1-b2). Need g^2.
                // We don't have 'inc_sq'. Do manually or add kernel.
                // Let's do loop for v update complex part, or:
                for (int k=0; k<BATCH_SIZE_KERNEL; ++k) {
                     float g_val = g_ptr[k];
                     v_ptr[k] += (1.0f - b2) * g_val * g_val;
                }
                
                // Bias correction and Update
                // p -= lr * m_hat / (sqrt(v_hat) + eps)
                for (int k=0; k<BATCH_SIZE_KERNEL; ++k) {
                    float m_hat = m_ptr[k] / (1.0f - beta1_t);
                    float v_hat = v_ptr[k] / (1.0f - beta2_t);
                    p_ptr[k] -= lr * m_hat / (sqrtf(v_hat) + eps);
                }
                
            } else {
                // Tail
                 for (int k=0; k<remaining; ++k) {
                    int idx = i + k;
                    float g_val = g[idx];
                    m[idx] = b1 * m[idx] + (1.0f - b1) * g_val;
                    v[idx] = b2 * v[idx] + (1.0f - b2) * g_val * g_val;
                    float m_hat = m[idx] / (1.0f - beta1_t);
                    float v_hat = v[idx] / (1.0f - beta2_t);
                    p[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
                }
            }
        }
    };

    for (int k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        maf_layer_adam_t* ladam = &adam->layers[k];
        const maf_layer_grad_t* lgrad = &grad->layers[k];
        
        int D = layer->param_dim;
        int C = layer->feature_dim;
        int H = layer->hidden_units;

        update(layer->W1y, lgrad->dW1y, ladam->mW1y, ladam->vW1y, H*D);
        update(layer->W1c, lgrad->dW1c, ladam->mW1c, ladam->vW1c, H*C);
        update(layer->b1,  lgrad->db1,  ladam->mb1,  ladam->vb1,  H);
        update(layer->W2,  lgrad->dW2,  ladam->mW2,  ladam->vW2,  2*D*H);
        update(layer->W2c, lgrad->dW2c, ladam->mW2c, ladam->vW2c, 2*D*C);
        update(layer->b2,  lgrad->db2,  ladam->mb2,  ladam->vb2,  2*D);
    }
}

} // extern "C"
