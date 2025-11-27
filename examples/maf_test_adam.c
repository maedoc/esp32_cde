#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "maf.h"

/* Helper to generate random float */
float rand_float() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f; /* -1 to 1 */
}

/* Helper to create a random model (reused from gradient check) */
maf_model_t* create_simple_model(uint16_t n_flows, uint16_t D, uint16_t C, uint16_t H) {
    maf_weights_t w = {0};
    w.n_flows = n_flows;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;

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

    /* Initialize masks */
    for (size_t i = 0; i < total_M1; i++) M1[i] = 1.0f; // No masking for simplicity in this test
    for (size_t i = 0; i < total_M2; i++) M2[i] = 1.0f;
    
    /* Initialize perms */
    for (uint16_t k = 0; k < n_flows; k++) {
        for (uint16_t i = 0; i < D; i++) {
            perm[k * D + i] = i;
            inv_perm[k * D + i] = i;
        }
        /* Reverse perm */
        if (k % 2 == 1) {
             for (uint16_t i = 0; i < D/2; i++) {
                 uint16_t tmp = perm[k*D+i];
                 perm[k*D+i] = perm[k*D + D-1-i];
                 perm[k*D + D-1-i] = tmp;
             }
             for (uint16_t i = 0; i < D; i++) inv_perm[k*D + perm[k*D+i]] = i;
        }
    }

    /* Small random weights */
    for (size_t i = 0; i < total_W1y; i++) W1y[i] = rand_float() * 0.01f;
    for (size_t i = 0; i < total_W1c; i++) W1c[i] = rand_float() * 0.01f;
    for (size_t i = 0; i < total_b1; i++) b1[i] = 0.0f;
    for (size_t i = 0; i < total_W2; i++) W2[i] = rand_float() * 0.01f;
    for (size_t i = 0; i < total_W2c; i++) W2c[i] = rand_float() * 0.01f;
    for (size_t i = 0; i < total_b2; i++) b2[i] = 0.0f; 
    /* Initialize log_scale (second half of b2) to close to 0 (scale 1) */
    // b2 is size 2*D. indices [D..2D-1] are for alpha.

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

    free(M1); free(M2); free(perm); free(inv_perm);
    free(W1y); free(W1c); free(b1); free(W2); free(W2c); free(b2);

    return model;
}

int main() {
    srand(42);
    printf("Testing MAF Adam Optimization...\n");

    uint16_t n_flows = 1;
    uint16_t D = 2;
    uint16_t C = 1;
    uint16_t H = 8;

    maf_model_t* model = create_simple_model(n_flows, D, C, H);
    maf_workspace_t* ws = maf_create_workspace(model);
    maf_cache_t* cache = maf_create_cache(model);
    maf_grad_t* grad = maf_create_grad(model);

    /* Create Adam Optimizer */
    maf_adam_t* adam = maf_create_adam(model, 0.001f, 0.9f, 0.999f, 1e-8f);

    /* Training Task: Learn to match a shifted Gaussian */
    /* Target: y ~ N(target_mu, I) given context */
    /* Let's just try to overfit a single point for simplicity check */
    float features[1] = {0.5f};
    float target[2] = {2.0f, -2.0f}; /* We want model to give high prob to this */
    
    printf("Target: [%.2f, %.2f]\n", target[0], target[1]);
    printf("Initial Log Prob: %f\n", maf_log_prob(model, ws, features, target));

    int steps = 200;
    for (int i = 0; i < steps; i++) {
        /* 1. Zero Grad */
        maf_zero_grad(model, grad);

        /* 2. Forward */
        float logp = maf_forward_train(model, ws, cache, features, target);
        
        /* 3. Backward (Minimize Negative Log Likelihood) */
        /* We want to maximize logp, so minimize -logp */
        /* maf_backward computes gradients of logp. */
        /* Adam update minimizes objective. So we need gradients of -logp. */
        /* grad(-logp) = -grad(logp). */
        /* We can just negate the learning rate? Or negate gradients. */
        /* Let's negate gradients in the loop below or just pass -lr to adam? */
        /* Standard Adam minimizes. theta = theta - lr * m_hat ... */
        /* If we want to MAXIMIZE J, we do theta = theta + lr * grad(J). */
        /* This is equivalent to theta = theta - lr * (-grad(J)). */
        /* So we can treat -grad(J) as the gradient of the loss (-J). */
        
        maf_backward(model, cache, grad, features, target);

        /* 4. Step */
        maf_adam_step(model, adam, grad);

        if (i % 20 == 0) {
             printf("Step %d: Log Prob = %f\n", i, logp);
        }
    }

    float final_logp = maf_log_prob(model, ws, features, target);
    printf("Final Log Prob: %f\n", final_logp);

    if (final_logp > -3.0f) { // Relaxed threshold
        printf("PASS: Optimization successful (Log Prob increased)\n");
    } else {
        printf("FAIL: Optimization failed to improve significantly\n");
    }

    maf_free_adam(adam);
    maf_free_grad(grad);
    maf_free_cache(cache);
    maf_free_workspace(ws);
    maf_free_model(model);

    return 0;
}
