#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cstring>

#include "maf.h"
#include "maf_batched.h"

using namespace std;

// --- Lorenz Generator ---
struct Point3D { float x, y, z; };

vector<float> generate_lorenz_data(int n_steps, float dt=0.01f) {
    vector<float> data;
    data.reserve(n_steps * 3);
    
    float x = 0.1f, y = 0.0f, z = 0.0f;
    float sigma = 10.0f, rho = 28.0f, beta = 8.0f/3.0f;

    // Burn-in
    for(int i=0; i<1000; i++) {
        float dx = sigma * (y - x);
        float dy = x * (rho - z) - y;
        float dz = x * y - beta * z;
        x += dx * dt; y += dy * dt; z += dz * dt;
    }

    // Generate
    for(int i=0; i<n_steps; i++) {
        float dx = sigma * (y - x);
        float dy = x * (rho - z) - y;
        float dz = x * y - beta * z;
        x += dx * dt; y += dy * dt; z += dz * dt;
        
        // Normalize roughly to [-3, 3] for easier training
        data.push_back(x / 10.0f);
        data.push_back(y / 10.0f);
        data.push_back((z - 25.0f) / 10.0f);
    }
    return data;
}

// --- Helpers ---

maf_model_t* create_model(int n_flows, int D, int H) {
    maf_weights_t w = {0};
    w.n_flows = n_flows;
    w.param_dim = D;
    w.feature_dim = 1; // Dummy feature dim (we'll input constant 0 or noise)
    w.hidden_units = H;

    int n_layers = n_flows;
    vector<float> zeros_f(n_layers * max(H*D, 2*D*H), 0.0f);
    vector<uint16_t> zeros_u(n_layers * D, 0);

    w.M1_data = zeros_f.data(); w.M2_data = zeros_f.data();
    w.perm_data = zeros_u.data(); w.inv_perm_data = zeros_u.data();
    w.W1y_data = zeros_f.data(); w.W1c_data = zeros_f.data(); w.b1_data = zeros_f.data();
    w.W2_data = zeros_f.data(); w.W2c_data = zeros_f.data(); w.b2_data = zeros_f.data();

    maf_model_t* model = maf_load_model(&w);
    
    // Random Init
    mt19937 rng(42);
    uniform_real_distribution<float> dist(-0.05f, 0.05f);
    bernoulli_distribution mask_dist(0.5);

    for (int k = 0; k < n_flows; k++) {
        maf_layer_t* l = &model->layers[k];
        // Init Weights
        for(int i=0; i<H*D; ++i) l->W1y[i] = dist(rng);
        for(int i=0; i<H*1; ++i) l->W1c[i] = 0.0f; // No context usage for unconditional
        for(int i=0; i<H; ++i)   l->b1[i] = 0.0f;
        for(int i=0; i<2*D*H; ++i) l->W2[i] = dist(rng);
        for(int i=0; i<2*D*1; ++i) l->W2c[i] = 0.0f;
        for(int i=0; i<2*D; ++i)   l->b2[i] = 0.0f; 
        // Init Log Scale Bias to slightly positive to encourage exploration early? Or 0.
        // b2 is [2*D]. First D is mu, second D is alpha.
        // maf.c clips alpha to [-7, 7]. 0 is fine (scale=1).

        // Init Masks (Autoregressive)
        // Simple random masks often fail AR property check.
        // We use identity/sequential for simplicity in this test or rely on dense + random.
        // For a true MAF, we need strictly triangular masks.
        // To keep this test simple and about "implementation equivalence", 
        // we will use random masks. Even if it's not a valid probability density, 
        // the math ops are the same.
        for(int i=0; i<H*D; ++i) l->M1[i] = mask_dist(rng) ? 1.0f : 0.0f;
        for(int i=0; i<D*H; ++i) l->M2[i] = mask_dist(rng) ? 1.0f : 0.0f;

        // Permutations
        for(int i=0; i<D; ++i) { l->perm[i] = i; l->inv_perm[i] = i; }
        if (k%2==1) {
            for(int i=0; i<D/2; ++i) swap(l->perm[i], l->perm[D-1-i]);
            for(int i=0; i<D; ++i) l->inv_perm[l->perm[i]] = i;
        }
    }
    return model;
}

void copy_weights(maf_model_t* dest, const maf_model_t* src) {
    for (int k = 0; k < src->n_flows; k++) {
        maf_layer_t* dl = &dest->layers[k];
        const maf_layer_t* sl = &src->layers[k];
        int D = sl->param_dim;
        int H = sl->hidden_units;
        int C = sl->feature_dim;
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

void calculate_stats(const vector<float>& data, int dim, vector<float>& mean, vector<float>& stddev) {
    int n = data.size() / dim;
    mean.assign(dim, 0.0f);
    stddev.assign(dim, 0.0f);
    
    for(int i=0; i<n; i++) {
        for(int d=0; d<dim; d++) {
            mean[d] += data[i*dim + d];
        }
    }
    for(int d=0; d<dim; d++) mean[d] /= n;
    
    for(int i=0; i<n; i++) {
        for(int d=0; d<dim; d++) {
            float diff = data[i*dim + d] - mean[d];
            stddev[d] += diff * diff;
        }
    }
    for(int d=0; d<dim; d++) stddev[d] = sqrt(stddev[d] / n);
}

int main() {
    int B = 32;
    int n_batches = 50; // Train on 50 * 32 = 1600 samples
    int n_epochs = 5;
    int D = 3; // x, y, z
    int C = 1; // Dummy context
    int H = 16;

    cout << "Generating Lorenz Data..." << endl;
    vector<float> data = generate_lorenz_data(n_batches * B);
    vector<float> dummy_ctx(n_batches * B * C, 0.0f);

    // Transpose data for Batched (Column Major)
    // Standard data is [N, D] (Row Major).
    // Batched expects [D, B] for each batch.
    
    cout << "Initializing Models..." << endl;
    maf_model_t* m_std = create_model(3, D, H);
    maf_model_t* m_batch = create_model(3, D, H);
    copy_weights(m_batch, m_std);

    // Train Standard
    cout << "Training Standard (C)..." << endl;
    maf_workspace_t* ws = maf_create_workspace(m_std);
    maf_cache_t* cache = maf_create_cache(m_std);
    maf_grad_t* grad = maf_create_grad(m_std);
    maf_adam_t* adam = maf_create_adam(m_std, 1e-3f, 0.9f, 0.999f, 1e-8f);

    for(int e=0; e<n_epochs; e++) {
        for(int i=0; i<n_batches; i++) {
            maf_zero_grad(m_std, grad);
            for(int b=0; b<B; b++) {
                int idx = i*B + b;
                maf_forward_train(m_std, ws, cache, &dummy_ctx[idx], &data[idx*D]);
                maf_backward(m_std, cache, grad, &dummy_ctx[idx], &data[idx*D]);
            }
            maf_adam_step(m_std, adam, grad);
        }
    }

    // Train Batched
    cout << "Training Batched (C++)..." << endl;
    maf_workspace_batched_t* ws_b = maf_create_workspace_batched(m_batch, B);
    maf_cache_batched_t* cache_b = maf_create_cache_batched(m_batch, B);
    maf_grad_t* grad_b = maf_create_grad(m_batch);
    maf_adam_t* adam_b = maf_create_adam(m_batch, 1e-3f, 0.9f, 0.999f, 1e-8f);
    vector<float> logps(B);

    // Buffers for transposed batch
    vector<float> batch_p_dev(B * D);
    vector<float> batch_c_dev(B * C);

    for(int e=0; e<n_epochs; e++) {
        for(int i=0; i<n_batches; i++) {
            // Prepare batch (Transpose)
            for(int b=0; b<B; b++) {
                int idx = i*B + b;
                for(int d=0; d<D; d++) batch_p_dev[d*B + b] = data[idx*D + d];
                for(int c=0; c<C; c++) batch_c_dev[c*B + b] = dummy_ctx[idx*C + c];
            }

            maf_zero_grad(m_batch, grad_b);
            maf_forward_train_batch(m_batch, ws_b, cache_b, batch_c_dev.data(), batch_p_dev.data(), logps.data(), B);
            maf_backward_batch(m_batch, cache_b, grad_b, batch_c_dev.data(), batch_p_dev.data(), B);
            maf_adam_step_vectorized(m_batch, adam_b, grad_b);
        }
    }

    // Sampling
    cout << "Sampling..." << endl;
    int n_samples = 2000;
    vector<float> samples_std(n_samples * D);
    vector<float> samples_batch(n_samples * D);
    vector<float> sample_ctx(C, 0.0f);

    // Use same seed for deterministic comparison of transformation
    // But note: models have diverged slightly due to float math, so samples won't be bit-exact.
    // We want distribution check.
    
    maf_sample(m_std, sample_ctx.data(), n_samples, samples_std.data(), 12345);
    maf_sample(m_batch, sample_ctx.data(), n_samples, samples_batch.data(), 12345);

    // Stats
    vector<float> mean_gt, std_gt;
    vector<float> mean_std, std_std;
    vector<float> mean_batch, std_batch;

    calculate_stats(data, D, mean_gt, std_gt);
    calculate_stats(samples_std, D, mean_std, std_std);
    calculate_stats(samples_batch, D, mean_batch, std_batch);

    cout << "\n--- Statistical Comparison ---" << endl;
    cout << "Dim | GT Mean | Std Mean | Bat Mean || GT Std | Std Std | Bat Std" << endl;
    cout << "-----------------------------------------------------------------" << endl;
    bool pass = true;
    for(int d=0; d<D; d++) {
        cout << "  " << d << " | " 
             << fixed << setprecision(3) << mean_gt[d] << " | " << mean_std[d] << " | " << mean_batch[d] << " || "
             << std_gt[d] << " | " << std_std[d] << " | " << std_batch[d] << endl;
             
        if (abs(mean_std[d] - mean_batch[d]) > 0.1f) pass = false;
        if (abs(std_std[d] - std_batch[d]) > 0.1f) pass = false;
    }

    cout << "\nComparison Result: " << (pass ? "PASS (Distributions Match)" : "FAIL (Distributions Diverged)") << endl;

    // Clean up
    maf_free_model(m_std); maf_free_model(m_batch);
    maf_free_workspace(ws); maf_free_cache(cache); maf_free_grad(grad); maf_free_adam(adam);
    maf_free_workspace_batched(ws_b); maf_free_cache_batched(cache_b); maf_free_grad(grad_b); maf_free_adam(adam_b);

    return pass ? 0 : 1;
}
