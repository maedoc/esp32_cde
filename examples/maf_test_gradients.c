#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "maf.h"

#define EPSILON 1e-4f
#define TOLERANCE 0.05f /* Relaxed tolerance for float precision */

/* Helper to generate random float */
float rand_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f; /* -1 to 1 */
}

/* Helper to create a random model */
maf_model_t* create_random_model(uint16_t n_flows, uint16_t D, uint16_t C, uint16_t H) {
    maf_weights_t w = {0};
    w.n_flows = n_flows;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;

    /* Allocate flat arrays for weights */
    size_t total_M1 = n_flows * H * D;
    size_t total_M2 = n_flows * D * H;
    size_t total_perm = n_flows * D;
    size_t total_W1y = n_flows * H * D;
    size_t total_W1c = n_flows * H * C;
    size_t total_b1 = n_flows * H;
    size_t total_W2 = n_flows * 2 * D * H;
    size_t total_W2c = n_flows * 2 * D * C;
    size_t total_b2 = n_flows * 2 * D;

    float* M1 = malloc(total_M1 * sizeof(float));
    float* M2 = malloc(total_M2 * sizeof(float));
    uint16_t* perm = malloc(total_perm * sizeof(uint16_t));
    uint16_t* inv_perm = malloc(total_perm * sizeof(uint16_t));
    float* W1y = malloc(total_W1y * sizeof(float));
    float* W1c = malloc(total_W1c * sizeof(float));
    float* b1 = malloc(total_b1 * sizeof(float));
    float* W2 = malloc(total_W2 * sizeof(float));
    float* W2c = malloc(total_W2c * sizeof(float));
    float* b2 = malloc(total_b2 * sizeof(float));

    /* Initialize weights randomly and masks */
    for (size_t i = 0; i < total_M1; i++) M1[i] = (rand() % 2); /* Binary mask */
    for (size_t i = 0; i < total_M2; i++) M2[i] = (rand() % 2);
    
    for (uint16_t k = 0; k < n_flows; k++) {
        for (uint16_t i = 0; i < D; i++) {
            perm[k * D + i] = i; /* Identity perm for simplicity */
            inv_perm[k * D + i] = i;
        }
        /* Random shuffle */
        for (uint16_t i = 0; i < D; i++) {
            int j = rand() % D;
            uint16_t tmp = perm[k*D + i];
            perm[k*D + i] = perm[k*D + j];
            perm[k*D + j] = tmp;
        }
        /* Compute inv perm */
        for (uint16_t i = 0; i < D; i++) {
            inv_perm[k * D + perm[k*D + i]] = i;
        }
    }

    for (size_t i = 0; i < total_W1y; i++) W1y[i] = rand_float() * 0.1f;
    for (size_t i = 0; i < total_W1c; i++) W1c[i] = rand_float() * 0.1f;
    for (size_t i = 0; i < total_b1; i++) b1[i] = rand_float() * 0.1f;
    for (size_t i = 0; i < total_W2; i++) W2[i] = rand_float() * 0.1f;
    for (size_t i = 0; i < total_W2c; i++) W2c[i] = rand_float() * 0.1f;
    for (size_t i = 0; i < total_b2; i++) b2[i] = rand_float() * 0.1f;

    w.M1_data = M1;
    w.M2_data = M2;
    w.perm_data = perm;
    w.inv_perm_data = inv_perm;
    w.W1y_data = W1y;
    w.W1c_data = W1c;
    w.b1_data = b1;
    w.W2_data = W2;
    w.W2c_data = W2c;
    w.b2_data = b2;

    maf_model_t* model = maf_load_model(&w);

    /* Free temp arrays (copied by load_model) */
    free(M1); free(M2); free(perm); free(inv_perm);
    free(W1y); free(W1c); free(b1); free(W2); free(W2c); free(b2);

    return model;
}

void check_parameter(maf_model_t* model, 
                     maf_workspace_t* ws, 
                     maf_cache_t* cache, 
                     maf_grad_t* grad, 
                     const float* features, 
                     const float* params,
                     float* param_ptr, 
                     float grad_ana,
                     const char* param_name) 
{
    float orig = *param_ptr;
    
    /* Perturb Up */
    *param_ptr = orig + EPSILON;
    float logp_plus = maf_log_prob(model, ws, features, params); // Use standard log_prob (no cache needed) 
    /* Loss J = -logp. */
    float loss_plus = -logp_plus;

    /* Perturb Down */
    *param_ptr = orig - EPSILON;
    float logp_minus = maf_log_prob(model, ws, features, params);
    float loss_minus = -logp_minus;

    /* Restore */
    *param_ptr = orig;

    /* Numerical Gradient */
    float grad_num = (loss_plus - loss_minus) / (2.0f * EPSILON);

    /* Comparison */
    float numerator = fabsf(grad_num - grad_ana);
    float denominator = fabsf(grad_num) + fabsf(grad_ana) + 1e-8f;
    float rel_error = numerator / denominator;

    /* Check absolute difference for small gradients */
    if (fabsf(grad_num) < 5e-3f && fabsf(grad_ana) < 5e-3f) {
        /* Pass: Both are very small */
    }
    else if (rel_error > TOLERANCE) {
        printf("FAIL %s: Num=%f, Ana=%f, RelErr=%f\n", param_name, grad_num, grad_ana, rel_error);
    } else {
        // printf("PASS %s: Num=%f, Ana=%f\n", param_name, grad_num, grad_ana);
    }
}

int main() {
    srand(time(NULL));
    printf("Running MAF Gradient Check...\n");

    uint16_t n_flows = 2;
    uint16_t D = 2;
    uint16_t C = 1;
    uint16_t H = 4;

    maf_model_t* model = create_random_model(n_flows, D, C, H);
    maf_workspace_t* ws = maf_create_workspace(model);
    maf_cache_t* cache = maf_create_cache(model);
    maf_grad_t* grad = maf_create_grad(model);

    float features[1] = {0.5f};
    float params[2] = {0.1f, -0.2f};

    /* 1. Analytical Gradient */
    float log_prob = maf_forward_train(model, ws, cache, features, params);
    printf("Forward Log Prob: %f\n", log_prob);

    maf_backward(model, cache, grad, features, params);

    /* 2. Check Gradients for Random Parameters */
    /* Loop through layers and check a few weights */
    for (int k = 0; k < n_flows; k++) {
        maf_layer_t* layer = &model->layers[k];
        maf_layer_grad_t* lgrad = &grad->layers[k];
        char name[64];

        /* Check W1y */
        for (int i = 0; i < H*D; i+=3) { // stride to check subset
            if (layer->M1[i] == 0.0f) continue; /* Skip masked weights */
            snprintf(name, 64, "L%d.W1y[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->W1y[i], lgrad->dW1y[i], name);
        }

        /* Check W1c */
        for (int i = 0; i < H*C; i++) {
            snprintf(name, 64, "L%d.W1c[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->W1c[i], lgrad->dW1c[i], name);
        }

        /* Check b1 */
        for (int i = 0; i < H; i++) {
            snprintf(name, 64, "L%d.b1[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->b1[i], lgrad->db1[i], name);
        }

        /* Check W2 */
        for (int i = 0; i < 2*D*H; i+=5) {
            uint16_t d_idx = (i / H) % D;
            uint16_t h_idx = i % H;
            if (layer->M2[d_idx * H + h_idx] == 0.0f) continue;
            
            snprintf(name, 64, "L%d.W2[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->W2[i], lgrad->dW2[i], name);
        }

        /* Check W2c */
        for (int i = 0; i < 2*D*C; i++) {
            snprintf(name, 64, "L%d.W2c[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->W2c[i], lgrad->dW2c[i], name);
        }

        /* Check b2 */
        for (int i = 0; i < 2*D; i++) {
            snprintf(name, 64, "L%d.b2[%d]", k, i);
            check_parameter(model, ws, cache, grad, features, params, 
                            &layer->b2[i], lgrad->db2[i], name);
        }
    }

    printf("Gradient Check Complete.\n");

    maf_free_grad(grad);
    maf_free_cache(cache);
    maf_free_workspace(ws);
    maf_free_model(model);

    return 0;
}
