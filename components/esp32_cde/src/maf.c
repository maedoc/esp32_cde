/**
 * @file maf.c
 * @brief MAF inference implementation
 */

#include "maf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Simple random number generator for sampling (LCG) */
static uint32_t maf_rng_state = 12345;

static void maf_seed_rng(uint32_t seed) {
    maf_rng_state = seed;
}

static float maf_randn(void) {
    /* Box-Muller transform for Gaussian samples */
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    /* Generate two uniform random numbers */
    maf_rng_state = maf_rng_state * 1103515245 + 12345;
    float u1 = (float)(maf_rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    maf_rng_state = maf_rng_state * 1103515245 + 12345;
    float u2 = (float)(maf_rng_state & 0x7FFFFFFF) / (float)0x7FFFFFFF;

    /* Box-Muller */
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * M_PI * u2;

    spare = r * sinf(theta);
    has_spare = 1;

    return r * cosf(theta);
}

/* =============================================================================
 * Model Loading and Memory Management
 * ========================================================================== */

maf_model_t* maf_load_model(const maf_weights_t* weights) {
    if (weights == NULL) {
        return NULL;
    }

    maf_model_t* model = (maf_model_t*)malloc(sizeof(maf_model_t));
    if (model == NULL) {
        return NULL;
    }

    model->n_flows = weights->n_flows;
    model->param_dim = weights->param_dim;
    model->feature_dim = weights->feature_dim;

    /* Allocate layers array */
    model->layers = (maf_layer_t*)calloc(weights->n_flows, sizeof(maf_layer_t));
    if (model->layers == NULL) {
        free(model);
        return NULL;
    }

    uint16_t D = weights->param_dim;
    uint16_t C = weights->feature_dim;
    uint16_t H = weights->hidden_units;

    /* Offset into flattened arrays */
    size_t m1_offset = 0;
    size_t m2_offset = 0;
    size_t perm_offset = 0;
    size_t w1y_offset = 0;
    size_t w1c_offset = 0;
    size_t b1_offset = 0;
    size_t w2_offset = 0;
    size_t w2c_offset = 0;
    size_t b2_offset = 0;

    /* Load each layer */
    for (uint16_t k = 0; k < weights->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];

        layer->param_dim = D;
        layer->feature_dim = C;
        layer->hidden_units = H;

        /* Allocate and copy masks */
        layer->M1 = (float*)malloc(H * D * sizeof(float));
        layer->M2 = (float*)malloc(D * H * sizeof(float));
        layer->perm = (uint16_t*)malloc(D * sizeof(uint16_t));
        layer->inv_perm = (uint16_t*)malloc(D * sizeof(uint16_t));

        if (!layer->M1 || !layer->M2 || !layer->perm || !layer->inv_perm) {
            maf_free_model(model);
            return NULL;
        }

        memcpy(layer->M1, &weights->M1_data[m1_offset], H * D * sizeof(float));
        memcpy(layer->M2, &weights->M2_data[m2_offset], D * H * sizeof(float));
        memcpy(layer->perm, &weights->perm_data[perm_offset], D * sizeof(uint16_t));
        memcpy(layer->inv_perm, &weights->inv_perm_data[perm_offset], D * sizeof(uint16_t));

        m1_offset += H * D;
        m2_offset += D * H;
        perm_offset += D;

        /* Allocate and copy weights */
        layer->W1y = (float*)malloc(H * D * sizeof(float));
        layer->W1c = (float*)malloc(H * C * sizeof(float));
        layer->b1 = (float*)malloc(H * sizeof(float));
        layer->W2 = (float*)malloc(2 * D * H * sizeof(float));
        layer->W2c = (float*)malloc(2 * D * C * sizeof(float));
        layer->b2 = (float*)malloc(2 * D * sizeof(float));

        if (!layer->W1y || !layer->W1c || !layer->b1 ||
            !layer->W2 || !layer->W2c || !layer->b2) {
            maf_free_model(model);
            return NULL;
        }

        memcpy(layer->W1y, &weights->W1y_data[w1y_offset], H * D * sizeof(float));
        memcpy(layer->W1c, &weights->W1c_data[w1c_offset], H * C * sizeof(float));
        memcpy(layer->b1, &weights->b1_data[b1_offset], H * sizeof(float));
        memcpy(layer->W2, &weights->W2_data[w2_offset], 2 * D * H * sizeof(float));
        memcpy(layer->W2c, &weights->W2c_data[w2c_offset], 2 * D * C * sizeof(float));
        memcpy(layer->b2, &weights->b2_data[b2_offset], 2 * D * sizeof(float));

        w1y_offset += H * D;
        w1c_offset += H * C;
        b1_offset += H;
        w2_offset += 2 * D * H;
        w2c_offset += 2 * D * C;
        b2_offset += 2 * D;
    }

    return model;
}

void maf_free_model(maf_model_t* model) {
    if (model == NULL) {
        return;
    }

    if (model->layers != NULL) {
        for (uint16_t k = 0; k < model->n_flows; k++) {
            maf_layer_t* layer = &model->layers[k];
            free(layer->M1);
            free(layer->M2);
            free(layer->perm);
            free(layer->inv_perm);
            free(layer->W1y);
            free(layer->W1c);
            free(layer->b1);
            free(layer->W2);
            free(layer->W2c);
            free(layer->b2);
        }
        free(model->layers);
    }

    free(model);
}

maf_workspace_t* maf_create_workspace(const maf_model_t* model) {
    if (model == NULL) {
        return NULL;
    }

    maf_workspace_t* ws = (maf_workspace_t*)calloc(1, sizeof(maf_workspace_t));
    if (ws == NULL) {
        return NULL;
    }

    uint16_t max_H = 0;
    uint16_t D = model->param_dim;

    for (uint16_t k = 0; k < model->n_flows; k++) {
        if (model->layers[k].hidden_units > max_H) {
            max_H = model->layers[k].hidden_units;
        }
    }

    /* Allocate buffers */
    ws->h = (float*)malloc(max_H * sizeof(float));
    ws->out = (float*)malloc(2 * D * sizeof(float));
    ws->u = (float*)malloc(D * sizeof(float));
    ws->u_perm = (float*)malloc(D * sizeof(float));
    ws->mu = (float*)malloc(D * sizeof(float));
    ws->alpha = (float*)malloc(D * sizeof(float));
    ws->x = (float*)malloc(D * sizeof(float));
    ws->y_perm = (float*)malloc(D * sizeof(float));

    /* Check allocation success */
    if (!ws->h || !ws->out || !ws->u || !ws->u_perm || 
        !ws->mu || !ws->alpha || !ws->x || !ws->y_perm) {
        maf_free_workspace(ws);
        return NULL;
    }

    return ws;
}

void maf_free_workspace(maf_workspace_t* ws) {
    if (ws == NULL) {
        return;
    }

    free(ws->h);
    free(ws->out);
    free(ws->u);
    free(ws->u_perm);
    free(ws->mu);
    free(ws->alpha);
    free(ws->x);
    free(ws->y_perm);
    free(ws);
}

size_t maf_get_memory_usage(const maf_model_t* model) {
    if (model == NULL) {
        return 0;
    }

    size_t total = sizeof(maf_model_t);
    total += model->n_flows * sizeof(maf_layer_t);

    uint16_t D = model->param_dim;
    uint16_t C = model->feature_dim;

    for (uint16_t k = 0; k < model->n_flows; k++) {
        uint16_t H = model->layers[k].hidden_units;

        /* Masks and permutations */
        total += H * D * sizeof(float);        /* M1 */
        total += D * H * sizeof(float);        /* M2 */
        total += D * sizeof(uint16_t);         /* perm */
        total += D * sizeof(uint16_t);         /* inv_perm */

        /* Weights */
        total += H * D * sizeof(float);        /* W1y */
        total += H * C * sizeof(float);        /* W1c */
        total += H * sizeof(float);            /* b1 */
        total += 2 * D * H * sizeof(float);    /* W2 */
        total += 2 * D * C * sizeof(float);    /* W2c */
        total += 2 * D * sizeof(float);        /* b2 */
    }

    return total;
}

/* =============================================================================
 * MADE Forward Pass
 * ========================================================================== */

void maf_made_forward(const maf_layer_t* layer,
                      maf_workspace_t* ws,
                      const float* y,
                      const float* context,
                      float* mu_out,
                      float* alpha_out) {
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    /* Use workspace buffers */
    float* h = ws->h;
    float* out = ws->out;

    /* Hidden layer: h = tanh((y @ W1y.T) * M1 + (ctx @ W1c.T) + b1) */
    for (uint16_t i = 0; i < H; i++) {
        float sum = layer->b1[i];

        /* y @ W1y.T * M1 */
        for (uint16_t j = 0; j < D; j++) {
            sum += y[j] * layer->W1y[i * D + j] * layer->M1[i * D + j];
        }

        /* context @ W1c.T */
        for (uint16_t j = 0; j < C; j++) {
            sum += context[j] * layer->W1c[i * C + j];
        }

        h[i] = tanhf(sum);
    }

    /* Output layer: out = (h @ W2.T) * M2 + (ctx @ W2c.T) + b2 */
    for (uint16_t i = 0; i < 2 * D; i++) {
        float sum = layer->b2[i];

        /* h @ W2.T * M2_tiled */
        uint16_t d_idx = i % D;  /* Which dimension of D */
        for (uint16_t j = 0; j < H; j++) {
            sum += h[j] * layer->W2[i * H + j] * layer->M2[d_idx * H + j];
        }

        /* context @ W2c.T */
        for (uint16_t j = 0; j < C; j++) {
            sum += context[j] * layer->W2c[i * C + j];
        }

        out[i] = sum;
    }

    /* Split into mu and alpha */
    for (uint16_t i = 0; i < D; i++) {
        mu_out[i] = out[i];
        /* Clip alpha to [-7, 7] for numerical stability */
        alpha_out[i] = fminf(fmaxf(out[D + i], -7.0f), 7.0f);
    }
}

/* =============================================================================
 * Inverse Layer (for Sampling)
 * ========================================================================== */

void maf_inverse_layer(const maf_layer_t* layer,
                       maf_workspace_t* ws,
                       const float* y_perm,
                       const float* context,
                       float* x_out) {
    uint16_t D = layer->param_dim;

    /* Use workspace buffers */
    float* u = ws->u;
    float* mu = ws->mu;
    float* alpha = ws->alpha;
    
    /* Clear u buffer (important for autoregressive property) */
    memset(u, 0, D * sizeof(float));

    /* Autoregressive inversion: for each dimension in order */
    for (uint16_t i = 0; i < D; i++) {
        /* Compute mu and alpha conditioned on u[:i] */
        maf_made_forward(layer, ws, u, context, mu, alpha);

        /* Invert: u[i] = y_perm[i] * exp(alpha[i]) + mu[i] */
        u[i] = y_perm[i] * expf(alpha[i]) + mu[i];
    }

    /* Apply inverse permutation */
    for (uint16_t i = 0; i < D; i++) {
        x_out[i] = u[layer->inv_perm[i]];
    }
}

/* =============================================================================
 * Sampling
 * ========================================================================== */

int maf_sample_from_noise(const maf_model_t* model,
                          const float* features,
                          const float* base_noise,
                          uint32_t n_samples,
                          float* samples_out) {
    if (model == NULL || features == NULL || base_noise == NULL || samples_out == NULL) {
        return -1;
    }

    /* Create workspace for this batch */
    maf_workspace_t* ws = maf_create_workspace(model);
    if (ws == NULL) {
        return -2;
    }

    uint16_t D = model->param_dim;
    float* x = ws->x;
    float* y_perm = ws->y_perm;

    /* Generate n_samples */
    for (uint32_t s = 0; s < n_samples; s++) {
        /* Start with provided base noise */
        memcpy(x, &base_noise[s * D], D * sizeof(float));

        /* Invert flow stack (reverse order) */
        for (int k = (int)model->n_flows - 1; k >= 0; k--) {
            const maf_layer_t* layer = &model->layers[k];

            /* Copy x to y_perm for input to inverse layer */
            memcpy(y_perm, x, D * sizeof(float));

            /* Invert layer */
            maf_inverse_layer(layer, ws, y_perm, features, x);
        }

        /* Copy result to output */
        memcpy(&samples_out[s * D], x, D * sizeof(float));
    }

    maf_free_workspace(ws);
    return 0;
}

int maf_sample(const maf_model_t* model,
               const float* features,
               uint32_t n_samples,
               float* samples_out,
               uint32_t seed) {
    if (model == NULL || features == NULL || samples_out == NULL) {
        return -1;
    }

    maf_seed_rng(seed);
    uint16_t D = model->param_dim;

    /* Allocate base noise buffer */
    float* base_noise = (float*)malloc(n_samples * D * sizeof(float));
    if (base_noise == NULL) {
        return -2;
    }

    /* Generate standard Gaussian noise */
    for (uint32_t s = 0; s < n_samples; s++) {
        for (uint16_t i = 0; i < D; i++) {
            base_noise[s * D + i] = maf_randn();
        }
    }

    /* Use the deterministic transformation */
    int ret = maf_sample_from_noise(model, features, base_noise, n_samples, samples_out);

    free(base_noise);
    return ret;
}

/* =============================================================================
 * Log Probability
 * ========================================================================== */

float maf_log_prob(const maf_model_t* model,
                   maf_workspace_t* ws,
                   const float* features,
                   const float* params) {
    if (model == NULL || features == NULL || params == NULL) {
        return -INFINITY;
    }

    uint16_t D = model->param_dim;

    /* Use workspace buffers */
    float* u = ws->u;
    float* u_perm = ws->u_perm;
    float* mu = ws->mu;
    float* alpha = ws->alpha;

    memcpy(u, params, D * sizeof(float));
    float log_det = 0.0f;

    /* Forward through flow stack */
    for (uint16_t k = 0; k < model->n_flows; k++) {
        const maf_layer_t* layer = &model->layers[k];

        /* Apply permutation */
        for (uint16_t i = 0; i < D; i++) {
            u_perm[i] = u[layer->perm[i]];
        }

        /* Forward pass */
        maf_made_forward(layer, ws, u_perm, features, mu, alpha);

        /* Transform: u = (u - mu) * exp(-alpha) */
        for (uint16_t i = 0; i < D; i++) {
            u[i] = (u_perm[i] - mu[i]) * expf(-alpha[i]);
            log_det -= alpha[i];
        }
    }

    /* Base distribution: N(0, I) */
    float base_logp = 0.0f;
    for (uint16_t i = 0; i < D; i++) {
        base_logp -= 0.5f * u[i] * u[i];
    }
    base_logp -= 0.5f * D * logf(2.0f * M_PI);

    return base_logp + log_det;
}

/* =============================================================================
 * Training Utilities
 * ========================================================================== */

maf_cache_t* maf_create_cache(const maf_model_t* model) {
    if (model == NULL) return NULL;

    maf_cache_t* cache = (maf_cache_t*)malloc(sizeof(maf_cache_t));
    if (!cache) return NULL;

    cache->n_flows = model->n_flows;
    cache->layers = (maf_layer_cache_t*)calloc(model->n_flows, sizeof(maf_layer_cache_t));
    if (!cache->layers) {
        free(cache);
        return NULL;
    }

    for (uint16_t k = 0; k < model->n_flows; k++) {
        uint16_t D = model->layers[k].param_dim;
        uint16_t H = model->layers[k].hidden_units;

        cache->layers[k].input = (float*)malloc(D * sizeof(float));
        cache->layers[k].h = (float*)malloc(H * sizeof(float));
        cache->layers[k].mu = (float*)malloc(D * sizeof(float));
        cache->layers[k].alpha = (float*)malloc(D * sizeof(float));

        if (!cache->layers[k].input || !cache->layers[k].h ||
            !cache->layers[k].mu || !cache->layers[k].alpha) {
            maf_free_cache(cache);
            return NULL;
        }
    }

    return cache;
}

void maf_free_cache(maf_cache_t* cache) {
    if (cache == NULL) return;

    if (cache->layers) {
        for (uint16_t k = 0; k < cache->n_flows; k++) {
            free(cache->layers[k].input);
            free(cache->layers[k].h);
            free(cache->layers[k].mu);
            free(cache->layers[k].alpha);
        }
        free(cache->layers);
    }
    free(cache);
}

maf_grad_t* maf_create_grad(const maf_model_t* model) {
    if (model == NULL) return NULL;

    maf_grad_t* grad = (maf_grad_t*)malloc(sizeof(maf_grad_t));
    if (!grad) return NULL;

    grad->n_flows = model->n_flows;
    grad->layers = (maf_layer_grad_t*)calloc(model->n_flows, sizeof(maf_layer_grad_t));
    if (!grad->layers) {
        free(grad);
        return NULL;
    }

    for (uint16_t k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        uint16_t D = layer->param_dim;
        uint16_t C = layer->feature_dim;
        uint16_t H = layer->hidden_units;

        grad->layers[k].dW1y = (float*)malloc(H * D * sizeof(float));
        grad->layers[k].dW1c = (float*)malloc(H * C * sizeof(float));
        grad->layers[k].db1 = (float*)malloc(H * sizeof(float));
        grad->layers[k].dW2 = (float*)malloc(2 * D * H * sizeof(float));
        grad->layers[k].dW2c = (float*)malloc(2 * D * C * sizeof(float));
        grad->layers[k].db2 = (float*)malloc(2 * D * sizeof(float));

        if (!grad->layers[k].dW1y || !grad->layers[k].dW1c || !grad->layers[k].db1 ||
            !grad->layers[k].dW2 || !grad->layers[k].dW2c || !grad->layers[k].db2) {
            maf_free_grad(grad);
            return NULL;
        }
    }
    
    maf_zero_grad(model, grad);
    return grad;
}

void maf_free_grad(maf_grad_t* grad) {
    if (grad == NULL) return;

    if (grad->layers) {
        for (uint16_t k = 0; k < grad->n_flows; k++) {
            free(grad->layers[k].dW1y);
            free(grad->layers[k].dW1c);
            free(grad->layers[k].db1);
            free(grad->layers[k].dW2);
            free(grad->layers[k].dW2c);
            free(grad->layers[k].db2);
        }
        free(grad->layers);
    }
    free(grad);
}

void maf_zero_grad(const maf_model_t* model, maf_grad_t* grad) {
    if (model == NULL || grad == NULL) return;

    for (uint16_t k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        uint16_t D = layer->param_dim;
        uint16_t C = layer->feature_dim;
        uint16_t H = layer->hidden_units;

        memset(grad->layers[k].dW1y, 0, H * D * sizeof(float));
        memset(grad->layers[k].dW1c, 0, H * C * sizeof(float));
        memset(grad->layers[k].db1, 0, H * sizeof(float));
        memset(grad->layers[k].dW2, 0, 2 * D * H * sizeof(float));
        memset(grad->layers[k].dW2c, 0, 2 * D * C * sizeof(float));
        memset(grad->layers[k].db2, 0, 2 * D * sizeof(float));
    }
}

float maf_forward_train(const maf_model_t* model,
                        maf_workspace_t* ws,
                        maf_cache_t* cache,
                        const float* features,
                        const float* params) {
    if (model == NULL || ws == NULL || cache == NULL || features == NULL || params == NULL) {
        return -INFINITY;
    }

    uint16_t D = model->param_dim;

    /* Use workspace buffers */
    float* u = ws->u;
    float* u_perm = ws->u_perm;
    float* mu = ws->mu;
    float* alpha = ws->alpha;

    memcpy(u, params, D * sizeof(float));
    float log_det = 0.0f;

    /* Forward through flow stack */
    for (uint16_t k = 0; k < model->n_flows; k++) {
        const maf_layer_t* layer = &model->layers[k];
        maf_layer_cache_t* lcache = &cache->layers[k];

        /* Apply permutation */
        for (uint16_t i = 0; i < D; i++) {
            u_perm[i] = u[layer->perm[i]];
        }

        /* Store input to cache */
        memcpy(lcache->input, u_perm, D * sizeof(float));

        /* Forward pass */
        maf_made_forward(layer, ws, u_perm, features, mu, alpha);

        /* Store activations to cache */
        memcpy(lcache->h, ws->h, layer->hidden_units * sizeof(float));
        memcpy(lcache->mu, mu, D * sizeof(float));
        memcpy(lcache->alpha, alpha, D * sizeof(float));

        /* Transform: u = (u - mu) * exp(-alpha) */
        for (uint16_t i = 0; i < D; i++) {
            u[i] = (u_perm[i] - mu[i]) * expf(-alpha[i]);
            log_det -= alpha[i];
        }
    }
/* ... */

    /* Base distribution: N(0, I) */
    float base_logp = 0.0f;
    for (uint16_t i = 0; i < D; i++) {
        base_logp -= 0.5f * u[i] * u[i];
    }
    base_logp -= 0.5f * D * logf(2.0f * M_PI);

    return base_logp + log_det;
}

/* =============================================================================
 * Backward Pass
 * ========================================================================== */

static void maf_layer_backward(const maf_layer_t* layer,
                               const maf_layer_cache_t* lcache,
                               maf_layer_grad_t* lgrad,
                               const float* features,
                               const float* grad_from_top, /* delta_out [D] */
                               float* grad_to_bottom)      /* delta_in [D] */
{
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    /* Allocate temporary gradient buffers */
    float* d_out = (float*)calloc(2 * D, sizeof(float));
    float* d_h = (float*)calloc(H, sizeof(float));

    if (!d_out || !d_h) {
        free(d_out);
        free(d_h);
        return;
    }

    /* 1. Gradients wrt mu and alpha */
    /* u_out = (u_in - mu) * exp(-alpha) */
    /* delta_out is dJ/du_out */
    for (uint16_t i = 0; i < D; i++) {
        float u_in = lcache->input[i];
        float mu = lcache->mu[i];
        float alpha = lcache->alpha[i];
        float exp_neg_alpha = expf(-alpha);
        float u_out = (u_in - mu) * exp_neg_alpha;

        /* dJ/dmu = dJ/du_out * du_out/dmu = delta * (-exp(-alpha)) */
        float d_mu = grad_from_top[i] * (-exp_neg_alpha);

        /* dJ/dalpha = dJ/du_out * du_out/dalpha + dJ/dalpha_explicit
           du_out/dalpha = (u_in - mu) * exp(-alpha) * (-1) = -u_out
           J = ... + sum(alpha) => dJ/dalpha_explicit = 1 */
        float d_alpha = grad_from_top[i] * (-u_out) + 1.0f;
           
        d_out[i] = d_mu;
        d_out[D + i] = d_alpha;
    }

    /* 2. Backprop through Output Layer (W2, b2) */
    /* out = (h @ W2.T) * M2 + (ctx @ W2c.T) + b2 */
    /* d_out is [2D] */

    for (uint16_t i = 0; i < 2 * D; i++) {
        float delta = d_out[i];
        
        /* Accumulate db2 */
        lgrad->db2[i] += delta;

        /* Accumulate dW2c: dJ/dW2c_ij = delta_i * context_j */
        for (uint16_t j = 0; j < C; j++) {
            lgrad->dW2c[i * C + j] += delta * features[j];
        }

        /* Backprop to h and Accumulate dW2 */
        
        uint16_t d_idx = i % D;
        for (uint16_t j = 0; j < H; j++) {
            float m2_val = layer->M2[d_idx * H + j];
            if (m2_val != 0.0f) {
                d_h[j] += delta * layer->W2[i * H + j] * m2_val;
                lgrad->dW2[i * H + j] += delta * lcache->h[j] * m2_val;
            }
        }
    }

    /* 3. Backprop through Tanh */
    /* h = tanh(pre_h) => d_pre_h = d_h * (1 - h^2) */
    for (uint16_t i = 0; i < H; i++) {
        float h_val = lcache->h[i];
        d_h[i] = d_h[i] * (1.0f - h_val * h_val);
    }

    /* 4. Backprop through Input Layer (W1, b1) */
    /* pre_h = (u_in @ W1y.T) * M1 + (ctx @ W1c.T) + b1 */

    /* Initialize grad_to_bottom (part 1: from MADE) */
    float* d_u_in = (float*)calloc(D, sizeof(float));
    if (!d_u_in) {
        free(d_out);
        free(d_h);
        return;
    }

    for (uint16_t i = 0; i < H; i++) {
        float delta = d_h[i];

        /* Accumulate db1 */
        lgrad->db1[i] += delta;

        /* Accumulate dW1c */
        for (uint16_t j = 0; j < C; j++) {
            lgrad->dW1c[i * C + j] += delta * features[j];
        }

        /* Backprop to u_in and Accumulate dW1y */
        for (uint16_t j = 0; j < D; j++) {
            float m1_val = layer->M1[i * D + j];
            if (m1_val != 0.0f) {
                d_u_in[j] += delta * layer->W1y[i * D + j] * m1_val;
                lgrad->dW1y[i * D + j] += delta * lcache->input[j] * m1_val;
            }
        }
    }

    /* 5. Combine gradients for u_in */
    /* Direct path: u_out = (u_in - mu) * exp(-alpha) */
    /* dJ/du_in_direct = dJ/du_out * du_out/du_in = grad_from_top * exp(-alpha) */
    
    for (uint16_t i = 0; i < D; i++) {
        float exp_neg_alpha = expf(-lcache->alpha[i]);
        d_u_in[i] += grad_from_top[i] * exp_neg_alpha;
    }

    /* 6. Inverse Permutation to get grad_to_bottom */
    for (uint16_t i = 0; i < D; i++) {
        grad_to_bottom[layer->perm[i]] = d_u_in[i];
    }

    free(d_out);
    free(d_h);
    free(d_u_in);
}

int maf_backward(const maf_model_t* model,
                 const maf_cache_t* cache,
                 maf_grad_t* grad,
                 const float* features,
                 const float* params) {
    if (model == NULL || cache == NULL || grad == NULL || features == NULL || params == NULL) {
        return -1;
    }

    uint16_t D = model->param_dim;

    /* Allocate gradient buffers for flow */
    float* delta = (float*)malloc(D * sizeof(float));
    float* prev_delta = (float*)malloc(D * sizeof(float));

    if (!delta || !prev_delta) {
        free(delta);
        free(prev_delta);
        return -2;
    }

    /* Initialize delta with gradient of base distribution */
    /* Recompute final u */
    {
        const maf_layer_cache_t* last_cache = &cache->layers[model->n_flows - 1];
        for (uint16_t i = 0; i < D; i++) {
             float u_in = last_cache->input[i];
             float mu = last_cache->mu[i];
             float alpha = last_cache->alpha[i];
             delta[i] = (u_in - mu) * expf(-alpha);
        }
    }

    /* Backpropagate through layers */
    for (int k = model->n_flows - 1; k >= 0; k--) {
        maf_layer_backward(&model->layers[k],
                           &cache->layers[k],
                           &grad->layers[k],
                           features,
                           delta,
                           prev_delta);
        
        /* Swap buffers for next iteration */
        memcpy(delta, prev_delta, D * sizeof(float));
    }

    free(delta);
    free(prev_delta);
    return 0;
}

void maf_sgd_step(maf_model_t* model, const maf_grad_t* grad, float lr) {
    if (model == NULL || grad == NULL) return;

    for (uint16_t k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        const maf_layer_grad_t* lgrad = &grad->layers[k];
        
        uint16_t D = layer->param_dim;
        uint16_t C = layer->feature_dim;
        uint16_t H = layer->hidden_units;

        for (uint32_t i = 0; i < H * D; i++) layer->W1y[i] -= lr * lgrad->dW1y[i];
        for (uint32_t i = 0; i < H * C; i++) layer->W1c[i] -= lr * lgrad->dW1c[i];
        for (uint32_t i = 0; i < H; i++)     layer->b1[i] -= lr * lgrad->db1[i];
        
        for (uint32_t i = 0; i < 2*D*H; i++) layer->W2[i] -= lr * lgrad->dW2[i];
        for (uint32_t i = 0; i < 2*D*C; i++) layer->W2c[i] -= lr * lgrad->dW2c[i];
        for (uint32_t i = 0; i < 2*D; i++)   layer->b2[i] -= lr * lgrad->db2[i];
    }
}

maf_adam_t* maf_create_adam(const maf_model_t* model, float lr, float beta1, float beta2, float epsilon) {
    if (model == NULL) return NULL;

    maf_adam_t* adam = (maf_adam_t*)malloc(sizeof(maf_adam_t));
    if (!adam) return NULL;

    adam->n_flows = model->n_flows;
    adam->t = 0;
    adam->lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;

    adam->layers = (maf_layer_adam_t*)calloc(model->n_flows, sizeof(maf_layer_adam_t));
    if (!adam->layers) {
        free(adam);
        return NULL;
    }

    for (uint16_t k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        maf_layer_adam_t* ladam = &adam->layers[k];
        uint16_t D = layer->param_dim;
        uint16_t C = layer->feature_dim;
        uint16_t H = layer->hidden_units;

        /* Allocate first moments (m) */
        ladam->mW1y = (float*)calloc(H * D, sizeof(float));
        ladam->mW1c = (float*)calloc(H * C, sizeof(float));
        ladam->mb1  = (float*)calloc(H, sizeof(float));
        ladam->mW2  = (float*)calloc(2 * D * H, sizeof(float));
        ladam->mW2c = (float*)calloc(2 * D * C, sizeof(float));
        ladam->mb2  = (float*)calloc(2 * D, sizeof(float));

        /* Allocate second moments (v) */
        ladam->vW1y = (float*)calloc(H * D, sizeof(float));
        ladam->vW1c = (float*)calloc(H * C, sizeof(float));
        ladam->vb1  = (float*)calloc(H, sizeof(float));
        ladam->vW2  = (float*)calloc(2 * D * H, sizeof(float));
        ladam->vW2c = (float*)calloc(2 * D * C, sizeof(float));
        ladam->vb2  = (float*)calloc(2 * D, sizeof(float));

        if (!ladam->mW1y || !ladam->mW1c || !ladam->mb1 ||
            !ladam->mW2  || !ladam->mW2c || !ladam->mb2 ||
            !ladam->vW1y || !ladam->vW1c || !ladam->vb1 ||
            !ladam->vW2  || !ladam->vW2c || !ladam->vb2) {
            maf_free_adam(adam);
            return NULL;
        }
    }

    return adam;
}

void maf_free_adam(maf_adam_t* adam) {
    if (adam == NULL) return;

    if (adam->layers) {
        for (uint16_t k = 0; k < adam->n_flows; k++) {
            maf_layer_adam_t* ladam = &adam->layers[k];
            free(ladam->mW1y); free(ladam->mW1c); free(ladam->mb1);
            free(ladam->mW2);  free(ladam->mW2c); free(ladam->mb2);
            free(ladam->vW1y); free(ladam->vW1c); free(ladam->vb1);
            free(ladam->vW2);  free(ladam->vW2c); free(ladam->vb2);
        }
        free(adam->layers);
    }
    free(adam);
}

static void adam_update_param(float* param, 
                              const float* grad, 
                              float* m, 
                              float* v, 
                              uint32_t size, 
                              float lr, 
                              float beta1, 
                              float beta2, 
                              float epsilon,
                              float beta1_t,
                              float beta2_t) {
    for (uint32_t i = 0; i < size; i++) {
        float g = grad[i];
        
        /* Update moments */
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        
        /* Bias correction */
        float m_hat = m[i] / (1.0f - beta1_t);
        float v_hat = v[i] / (1.0f - beta2_t);
        
        /* Update parameter */
        param[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

void maf_adam_step(maf_model_t* model, maf_adam_t* adam, const maf_grad_t* grad) {
    if (model == NULL || adam == NULL || grad == NULL) return;

    adam->t++;
    float beta1_t = powf(adam->beta1, adam->t);
    float beta2_t = powf(adam->beta2, adam->t);

    for (uint16_t k = 0; k < model->n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        maf_layer_adam_t* ladam = &adam->layers[k];
        const maf_layer_grad_t* lgrad = &grad->layers[k];
        
        uint16_t D = layer->param_dim;
        uint16_t C = layer->feature_dim;
        uint16_t H = layer->hidden_units;

        adam_update_param(layer->W1y, lgrad->dW1y, ladam->mW1y, ladam->vW1y, H*D, adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
        adam_update_param(layer->W1c, lgrad->dW1c, ladam->mW1c, ladam->vW1c, H*C, adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
        adam_update_param(layer->b1,  lgrad->db1,  ladam->mb1,  ladam->vb1,  H,   adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
        
        adam_update_param(layer->W2,  lgrad->dW2,  ladam->mW2,  ladam->vW2,  2*D*H, adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
        adam_update_param(layer->W2c, lgrad->dW2c, ladam->mW2c, ladam->vW2c, 2*D*C, adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
        adam_update_param(layer->b2,  lgrad->db2,  ladam->mb2,  ladam->vb2,  2*D,   adam->lr, adam->beta1, adam->beta2, adam->epsilon, beta1_t, beta2_t);
    }
}
