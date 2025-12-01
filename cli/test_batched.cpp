#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <algorithm>

#include "maf.h"
#include "maf_batched.h"

#define BATCH_SIZE 32

// Helper to print array
void print_arr(const char* name, const float* arr, int n) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min(n, 5); i++) std::cout << arr[i] << " ";
    if (n > 5) std::cout << "...";
    std::cout << "]" << std::endl;
}

// Helper to compare arrays
bool check_close(const float* a, const float* b, int n, float tol = 1e-4) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    std::cout << "Max diff: " << max_diff << std::endl;
    return true;
}

maf_model_t* create_random_model(int n_flows, int D, int C, int H) {
    maf_weights_t w = {0};
    w.n_flows = n_flows;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;

    // Allocate memory for weights
    // Note: We need to keep this memory alive or copy it.
    // maf_load_model copies it.
    
    std::vector<float> M1(n_flows * H * D);
    std::vector<float> M2(n_flows * D * H);
    std::vector<uint16_t> perm(n_flows * D);
    std::vector<uint16_t> inv_perm(n_flows * D);
    std::vector<float> W1y(n_flows * H * D);
    std::vector<float> W1c(n_flows * H * C);
    std::vector<float> b1(n_flows * H);
    std::vector<float> W2(n_flows * 2 * D * H);
    std::vector<float> W2c(n_flows * 2 * D * C);
    std::vector<float> b2(n_flows * 2 * D);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> mask_dist(0.0f, 1.0f);

    // Random init
    for (auto& v : W1y) v = dist(rng);
    for (auto& v : W1c) v = dist(rng);
    for (auto& v : b1) v = 0.0f;
    for (auto& v : W2) v = dist(rng);
    for (auto& v : W2c) v = dist(rng);
    for (auto& v : b2) v = 0.0f;

    // Masks (simplified: full connectivity for test, or identity)
    // To properly test MAF structure, masks should be autoregressive.
    // For correctness of implementation (matrix math), random masks (0 or 1) are fine.
    for (auto& v : M1) v = mask_dist(rng) > 0.5f ? 1.0f : 0.0f;
    for (auto& v : M2) v = mask_dist(rng) > 0.5f ? 1.0f : 0.0f;

    // Perms
    for (int k = 0; k < n_flows; k++) {
        for (int i = 0; i < D; i++) {
            perm[k * D + i] = i;
            inv_perm[k * D + i] = i;
        }
        if (k % 2 == 1) {
            std::reverse(perm.begin() + k*D, perm.begin() + (k+1)*D);
            for(int i=0; i<D; ++i) inv_perm[k*D + perm[k*D+i]] = i;
        }
    }

    w.M1_data = M1.data();
    w.M2_data = M2.data();
    w.perm_data = perm.data();
    w.inv_perm_data = inv_perm.data();
    w.W1y_data = W1y.data();
    w.W1c_data = W1c.data();
    w.b1_data = b1.data();
    w.W2_data = W2.data();
    w.W2c_data = W2c.data();
    w.b2_data = b2.data();

    return maf_load_model(&w);
}

int main() {
    std::cout << "--- Testing MAF Batched Implementation ---" << std::endl;
    
    int n_flows = 2;
    int D = 4;
    int C = 3;
    int H = 8;
    int B = BATCH_SIZE;

    maf_model_t* model = create_random_model(n_flows, D, C, H);
    
    // Generate Batch Data
    // Standard C: features [B][C] -> flattened [B*C]
    // Params: [B][D] -> flattened [B*D]
    std::vector<float> features_host(B * C);
    std::vector<float> params_host(B * D);
    
    std::mt19937 rng(123);
    std::normal_distribution<float> randn(0.0f, 1.0f);
    
    for (auto& v : features_host) v = randn(rng);
    for (auto& v : params_host) v = randn(rng);
    
    // Transpose for Batched API (SoA)
    // Features: [C, B]
    // Params: [D, B]
    std::vector<float> features_device(B * C);
    std::vector<float> params_device(B * D);
    
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            features_device[c * B + b] = features_host[b * C + c];
        }
        for (int d = 0; d < D; d++) {
            params_device[d * B + b] = params_host[b * D + d];
        }
    }

    // ==========================================================
    // 1. Standard C Implementation Loop
    // ==========================================================
    maf_workspace_t* ws = maf_create_workspace(model);
    maf_cache_t* cache = maf_create_cache(model);
    maf_grad_t* grad_c = maf_create_grad(model); // Accumulator
    
    std::vector<float> log_probs_c(B);
    
    maf_zero_grad(model, grad_c);
    
    for (int b = 0; b < B; b++) {
        float* f_ptr = &features_host[b * C];
        float* p_ptr = &params_host[b * D];
        
        log_probs_c[b] = maf_forward_train(model, ws, cache, f_ptr, p_ptr);
        maf_backward(model, cache, grad_c, f_ptr, p_ptr);
    }
    
    std::cout << "Standard C Forward/Backward Complete." << std::endl;
    print_arr("LogProbs (C)", log_probs_c.data(), B);

    // ==========================================================
    // 2. Batched Implementation
    // ==========================================================
    maf_workspace_batched_t* ws_b = maf_create_workspace_batched(model, B);
    maf_cache_batched_t* cache_b = maf_create_cache_batched(model, B);
    maf_grad_t* grad_b = maf_create_grad(model); // Accumulator shared structure
    
    maf_zero_grad(model, grad_b);
    
    std::vector<float> log_probs_b(B);
    
    maf_forward_train_batch(model, ws_b, cache_b, features_device.data(), params_device.data(), log_probs_b.data(), B);
    
    std::cout << "Batched Forward Complete." << std::endl;
    print_arr("LogProbs (Batched)", log_probs_b.data(), B);
    
    if (!check_close(log_probs_c.data(), log_probs_b.data(), B)) {
        std::cout << "FAIL: Log Probs mismatch!" << std::endl;
        return 1;
    }
    
    maf_backward_batch(model, cache_b, grad_b, features_device.data(), params_device.data(), B);
    
    std::cout << "Batched Backward Complete." << std::endl;
    
    // ==========================================================
    // 3. Compare Gradients
    // ==========================================================
    bool grads_match = true;
    for (int k = 0; k < model->n_flows; k++) {
        maf_layer_grad_t* g_c = &grad_c->layers[k];
        maf_layer_grad_t* g_b = &grad_b->layers[k];
        int D = model->layers[k].param_dim;
        int H = model->layers[k].hidden_units;
        int C = model->layers[k].feature_dim;
        
        if (!check_close(g_c->dW1y, g_b->dW1y, H*D, 1e-3f)) { std::cout << "Layer " << k << " dW1y mismatch" << std::endl; grads_match = false; }
        if (!check_close(g_c->dW1c, g_b->dW1c, H*C, 1e-3f)) { std::cout << "Layer " << k << " dW1c mismatch" << std::endl; grads_match = false; }
        if (!check_close(g_c->db1, g_b->db1, H, 1e-3f)) { std::cout << "Layer " << k << " db1 mismatch" << std::endl; grads_match = false; }
        if (!check_close(g_c->dW2, g_b->dW2, 2*D*H, 1e-3f)) { std::cout << "Layer " << k << " dW2 mismatch" << std::endl; grads_match = false; }
        if (!check_close(g_c->dW2c, g_b->dW2c, 2*D*C, 1e-3f)) { std::cout << "Layer " << k << " dW2c mismatch" << std::endl; grads_match = false; }
        if (!check_close(g_c->db2, g_b->db2, 2*D, 1e-3f)) { std::cout << "Layer " << k << " db2 mismatch" << std::endl; grads_match = false; }
    }
    
    if (grads_match) {
        std::cout << "SUCCESS: Gradients match!" << std::endl;
    } else {
        std::cout << "FAIL: Gradients mismatch." << std::endl;
        return 1;
    }
    
    // Clean up
    maf_free_workspace(ws);
    maf_free_cache(cache);
    maf_free_grad(grad_c);
    maf_free_workspace_batched(ws_b);
    maf_free_cache_batched(cache_b);
    maf_free_grad(grad_b);
    maf_free_model(model);

    return 0;
}
