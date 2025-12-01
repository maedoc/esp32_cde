#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>

#include "maf.h"
#include "maf_batched.h"

#define BATCH_SIZE 32
#define N_STEPS 500 // Number of batch updates to perform
#define LEARNING_RATE 1e-3f

using namespace std;
using namespace std::chrono;

// --- Helpers ---

void copy_model_weights(maf_model_t* dest, const maf_model_t* src) {
    if (dest->n_flows != src->n_flows) return;
    for (int k = 0; k < src->n_flows; k++) {
        maf_layer_t* dl = &dest->layers[k];
        const maf_layer_t* sl = &src->layers[k];
        int D = sl->param_dim;
        int C = sl->feature_dim;
        int H = sl->hidden_units;
        
        memcpy(dl->W1y, sl->W1y, H*D*sizeof(float));
        memcpy(dl->W1c, sl->W1c, H*C*sizeof(float));
        memcpy(dl->b1,  sl->b1,  H*sizeof(float));
        memcpy(dl->W2,  sl->W2,  2*D*H*sizeof(float));
        memcpy(dl->W2c, sl->W2c, 2*D*C*sizeof(float));
        memcpy(dl->b2,  sl->b2,  2*D*sizeof(float));
        
        memcpy(dl->M1, sl->M1, H*D*sizeof(float));
        memcpy(dl->M2, sl->M2, D*H*sizeof(float));
        memcpy(dl->perm, sl->perm, D*sizeof(uint16_t));
        memcpy(dl->inv_perm, sl->inv_perm, D*sizeof(uint16_t));
    }
}

maf_model_t* create_model(int n_flows, int D, int C, int H) {
    // Create a dummy weights struct to use maf_load_model
    // We will perform deep allocation manually or just use maf_load_model with dummy data then random init
    // Actually, maf_load_model allocates buffers. Let's create a valid weights struct.
    
    maf_weights_t w = {0};
    w.n_flows = n_flows;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;

    int n_layers = n_flows;
    vector<float> zeros_f(n_layers * max(H*D, 2*D*H), 0.0f); // Safe overestimate buffer
    vector<uint16_t> zeros_u(n_layers * D, 0);

    w.M1_data = zeros_f.data();
    w.M2_data = zeros_f.data();
    w.perm_data = zeros_u.data();
    w.inv_perm_data = zeros_u.data();
    w.W1y_data = zeros_f.data();
    w.W1c_data = zeros_f.data();
    w.b1_data = zeros_f.data();
    w.W2_data = zeros_f.data();
    w.W2c_data = zeros_f.data();
    w.b2_data = zeros_f.data();

    maf_model_t* model = maf_load_model(&w);
    
    // Now Random Init Properly
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-0.1f, 0.1f);
    bernoulli_distribution mask_dist(0.5);

    for (int k = 0; k < n_flows; k++) {
        maf_layer_t* l = &model->layers[k];
        for(int i=0; i<H*D; ++i) { l->W1y[i] = dist(rng); l->M1[i] = mask_dist(rng) ? 1.0f : 0.0f; }
        for(int i=0; i<H*C; ++i) l->W1c[i] = dist(rng);
        for(int i=0; i<H; ++i)   l->b1[i] = 0.0f;
        
        for(int i=0; i<2*D*H; ++i) { l->W2[i] = dist(rng); } // M2 is smaller D*H
        for(int i=0; i<D*H; ++i)   l->M2[i] = mask_dist(rng) ? 1.0f : 0.0f;

        for(int i=0; i<2*D*C; ++i) l->W2c[i] = dist(rng);
        for(int i=0; i<2*D; ++i)   l->b2[i] = 0.0f;

        for(int i=0; i<D; ++i) { l->perm[i] = i; l->inv_perm[i] = i; }
        if (k%2==1) {
            for(int i=0; i<D/2; ++i) swap(l->perm[i], l->perm[D-1-i]);
            for(int i=0; i<D; ++i) l->inv_perm[l->perm[i]] = i;
        }
    }
    return model;
}

float compute_mean_abs_diff(const maf_model_t* m1, const maf_model_t* m2) {
    double diff = 0.0;
    double count = 0.0;
    
    for (int k=0; k<m1->n_flows; k++) {
        int D = m1->layers[k].param_dim;
        int H = m1->layers[k].hidden_units;
        int C = m1->layers[k].feature_dim;
        
        auto add_diff = [&](float* a, float* b, int n) {
            for(int i=0; i<n; i++) diff += abs(a[i] - b[i]);
            count += n;
        };
        
        add_diff(m1->layers[k].W1y, m2->layers[k].W1y, H*D);
        add_diff(m1->layers[k].W1c, m2->layers[k].W1c, H*C);
        add_diff(m1->layers[k].b1,  m2->layers[k].b1,  H);
        add_diff(m1->layers[k].W2,  m2->layers[k].W2,  2*D*H);
        add_diff(m1->layers[k].W2c, m2->layers[k].W2c, 2*D*C);
        add_diff(m1->layers[k].b2,  m2->layers[k].b2,  2*D);
    }
    return (float)(diff / count);
}

// --- Benchmarks ---

int main() {
    // Settings
    int n_flows = 5;
    int D = 16;
    int C = 8;
    int H = 32; // Larger model to make compute dominate overhead
    int B = 32;
    int steps = N_STEPS;

    cout << "--- MAF Benchmark: Standard vs Batched ---" << endl;
    cout << "Model: Flows=" << n_flows << ", D=" << D << ", C=" << C << ", H=" << H << endl;
    cout << "Training Steps: " << steps << " (Batch Size " << B << ")" << endl;

    // Init Data (Random Batch)
    // We use the SAME batch repeatedly to simulate overfitting training loop (avoids data gen overhead)
    vector<float> feats(B * C);
    vector<float> params(B * D);
    vector<float> feats_dev(B * C); // Transposed
    vector<float> params_dev(B * D); // Transposed

    mt19937 rng(101);
    normal_distribution<float> nd(0,1);
    for(auto& v : feats) v = nd(rng);
    for(auto& v : params) v = nd(rng);

    // Prepare Transposed Data for Batched
    for(int b=0; b<B; b++) {
        for(int c=0; c<C; c++) feats_dev[c*B+b] = feats[b*C+c];
        for(int d=0; d<D; d++) params_dev[d*B+b] = params[b*D+d];
    }

    // Init Models
    maf_model_t* model_std = create_model(n_flows, D, C, H);
    maf_model_t* model_batch = create_model(n_flows, D, C, H);
    copy_model_weights(model_batch, model_std); // Exact copy

    // --- Standard Benchmark ---
    maf_workspace_t* ws = maf_create_workspace(model_std);
    maf_cache_t* cache = maf_create_cache(model_std);
    maf_grad_t* grad = maf_create_grad(model_std);
    maf_adam_t* adam = maf_create_adam(model_std, LEARNING_RATE, 0.9f, 0.999f, 1e-8f);
    
    cout << "Running Standard C Training..." << flush;
    auto start_std = high_resolution_clock::now();
    
    for(int s=0; s<steps; s++) {
        maf_zero_grad(model_std, grad);
        for(int b=0; b<B; b++) {
            float* f = &feats[b*C];
            float* p = &params[b*D];
            maf_forward_train(model_std, ws, cache, f, p);
            maf_backward(model_std, cache, grad, f, p);
        }
        maf_adam_step(model_std, adam, grad);
    }
    
    auto end_std = high_resolution_clock::now();
    double time_std = duration_cast<duration<double>>(end_std - start_std).count();
    cout << " Done. Time: " << time_std << "s" << endl;

    // --- Batched Benchmark ---
    maf_workspace_batched_t* ws_b = maf_create_workspace_batched(model_batch, B);
    maf_cache_batched_t* cache_b = maf_create_cache_batched(model_batch, B);
    maf_grad_t* grad_b = maf_create_grad(model_batch);
    maf_adam_t* adam_b = maf_create_adam(model_batch, LEARNING_RATE, 0.9f, 0.999f, 1e-8f);
    vector<float> logps_b(B);

    cout << "Running Batched C++ Training..." << flush;
    auto start_batch = high_resolution_clock::now();

    for(int s=0; s<steps; s++) {
        maf_zero_grad(model_batch, grad_b);
        maf_forward_train_batch(model_batch, ws_b, cache_b, feats_dev.data(), params_dev.data(), logps_b.data(), B);
        maf_backward_batch(model_batch, cache_b, grad_b, feats_dev.data(), params_dev.data(), B);
        maf_adam_step_vectorized(model_batch, adam_b, grad_b);
    }

    auto end_batch = high_resolution_clock::now();
    double time_batch = duration_cast<duration<double>>(end_batch - start_batch).count();
    cout << " Done. Time: " << time_batch << "s" << endl;

    // --- Comparison ---
    float weight_diff = compute_mean_abs_diff(model_std, model_batch);
    double speedup = time_std / time_batch;

    cout << "\n--- Results ---" << endl;
    cout << "Standard Time: " << fixed << setprecision(4) << time_std << "s" << endl;
    cout << "Batched Time:  " << fixed << setprecision(4) << time_batch << "s" << endl;
    cout << "Speedup:       " << fixed << setprecision(2) << speedup << "x" << endl;
    cout << "Weight Diff:   " << scientific << weight_diff << endl;

    if (speedup > 1.0) cout << "PERFORMANCE: PASS (Faster)" << endl;
    else cout << "PERFORMANCE: FAIL (Slower)" << endl;

    if (weight_diff < 1e-3) cout << "ACCURACY:    PASS (Weights Identical)" << endl;
    else cout << "ACCURACY:    FAIL (Weights Diverged)" << endl;

    // Cleanup
    maf_free_workspace(ws); maf_free_cache(cache); maf_free_grad(grad); maf_free_adam(adam);
    maf_free_workspace_batched(ws_b); maf_free_cache_batched(cache_b); maf_free_grad(grad_b); maf_free_adam(adam_b);
    maf_free_model(model_std); maf_free_model(model_batch);

    return 0;
}
