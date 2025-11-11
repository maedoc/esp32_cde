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
                      const float* y,
                      const float* context,
                      float* mu_out,
                      float* alpha_out) {
    uint16_t D = layer->param_dim;
    uint16_t C = layer->feature_dim;
    uint16_t H = layer->hidden_units;

    /* Allocate temporary hidden state */
    float* h = (float*)malloc(H * sizeof(float));
    if (h == NULL) {
        return;
    }

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
    float* out = (float*)malloc(2 * D * sizeof(float));
    if (out == NULL) {
        free(h);
        return;
    }

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

    free(h);
    free(out);
}

/* =============================================================================
 * Inverse Layer (for Sampling)
 * ========================================================================== */

void maf_inverse_layer(const maf_layer_t* layer,
                       const float* y_perm,
                       const float* context,
                       float* x_out) {
    uint16_t D = layer->param_dim;

    /* Allocate working buffer for autoregressive inversion */
    float* u = (float*)calloc(D, sizeof(float));
    float* mu = (float*)malloc(D * sizeof(float));
    float* alpha = (float*)malloc(D * sizeof(float));

    if (!u || !mu || !alpha) {
        free(u);
        free(mu);
        free(alpha);
        return;
    }

    /* Autoregressive inversion: for each dimension in order */
    for (uint16_t i = 0; i < D; i++) {
        /* Compute mu and alpha conditioned on u[:i] */
        maf_made_forward(layer, u, context, mu, alpha);

        /* Invert: u[i] = y_perm[i] * exp(alpha[i]) + mu[i] */
        u[i] = y_perm[i] * expf(alpha[i]) + mu[i];
    }

    /* Apply inverse permutation */
    for (uint16_t i = 0; i < D; i++) {
        x_out[layer->inv_perm[i]] = u[i];
    }

    free(u);
    free(mu);
    free(alpha);
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

    uint16_t D = model->param_dim;

    /* Allocate working buffers */
    float* x = (float*)malloc(D * sizeof(float));
    float* y_perm = (float*)malloc(D * sizeof(float));

    if (!x || !y_perm) {
        free(x);
        free(y_perm);
        return -2;
    }

    /* Generate n_samples */
    for (uint32_t s = 0; s < n_samples; s++) {
        /* Start with provided base noise */
        memcpy(x, &base_noise[s * D], D * sizeof(float));

        /* Invert flow stack (reverse order) */
        for (int k = (int)model->n_flows - 1; k >= 0; k--) {
            const maf_layer_t* layer = &model->layers[k];

            /* Apply permutation to x to get y_perm */
            for (uint16_t i = 0; i < D; i++) {
                y_perm[i] = x[layer->perm[i]];
            }

            /* Invert layer */
            maf_inverse_layer(layer, y_perm, features, x);
        }

        /* Copy result to output */
        memcpy(&samples_out[s * D], x, D * sizeof(float));
    }

    free(x);
    free(y_perm);

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
                   const float* features,
                   const float* params) {
    if (model == NULL || features == NULL || params == NULL) {
        return -INFINITY;
    }

    uint16_t D = model->param_dim;

    /* Allocate working buffers */
    float* u = (float*)malloc(D * sizeof(float));
    float* u_perm = (float*)malloc(D * sizeof(float));
    float* mu = (float*)malloc(D * sizeof(float));
    float* alpha = (float*)malloc(D * sizeof(float));

    if (!u || !u_perm || !mu || !alpha) {
        free(u);
        free(u_perm);
        free(mu);
        free(alpha);
        return -INFINITY;
    }

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
        maf_made_forward(layer, u_perm, features, mu, alpha);

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

    free(u);
    free(u_perm);
    free(mu);
    free(alpha);

    return base_logp + log_det;
}
