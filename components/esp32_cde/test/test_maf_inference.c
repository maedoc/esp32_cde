#include "unity.h"
#include "maf.h"
#include "test_model_blob.h" /* Generated via xxd */
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "test_maf";

/* =========================================================================
 * Memory Reader Utility
 * ========================================================================= */

typedef struct {
    const uint8_t* data;
    size_t pos;
    size_t len;
} MemReader;

static void mem_read(MemReader* r, void* dst, size_t size) {
    TEST_ASSERT_LESS_OR_EQUAL_MESSAGE(r->len, r->pos + size, "Buffer overflow in mem_read");
    memcpy(dst, r->data + r->pos, size);
    r->pos += size;
}

/* =========================================================================
 * Model Loader (From Blob)
 * ========================================================================= */

/* Global pointers to keep data alive for the model lifetime */
static float *g_M1, *g_M2, *g_W1y, *g_W1c, *g_b1, *g_W2, *g_W2c, *g_b2;
static uint16_t *g_perm, *g_inv_perm;

static maf_model_t* load_model_from_blob(const uint8_t* blob, size_t len) {
    MemReader r = { .data = blob, .pos = 0, .len = len };
    
    char magic[5] = {0};
    mem_read(&r, magic, 4);
    TEST_ASSERT_EQUAL_STRING("MAF1", magic);

    uint16_t n_flows, param_dim, feature_dim, hidden;
    mem_read(&r, &n_flows, sizeof(uint16_t));
    mem_read(&r, &param_dim, sizeof(uint16_t));
    mem_read(&r, &feature_dim, sizeof(uint16_t));
    mem_read(&r, &hidden, sizeof(uint16_t));

    int n = n_flows;
    int H = hidden;
    int D = param_dim;
    int C = feature_dim;

    /* Allocate buffers */
    g_M1 = malloc(n * H * D * sizeof(float));
    g_M2 = malloc(n * D * H * sizeof(float));
    g_perm = malloc(n * D * sizeof(uint16_t));
    g_inv_perm = malloc(n * D * sizeof(uint16_t));
    g_W1y = malloc(n * H * D * sizeof(float));
    g_W1c = malloc(n * H * C * sizeof(float));
    g_b1 = malloc(n * H * sizeof(float));
    g_W2 = malloc(n * 2 * D * H * sizeof(float));
    g_W2c = malloc(n * 2 * D * C * sizeof(float));
    g_b2 = malloc(n * 2 * D * sizeof(float));

    mem_read(&r, g_M1, n*H*D*sizeof(float));
    mem_read(&r, g_M2, n*D*H*sizeof(float));
    mem_read(&r, g_perm, n*D*sizeof(uint16_t));
    mem_read(&r, g_inv_perm, n*D*sizeof(uint16_t));
    mem_read(&r, g_W1y, n*H*D*sizeof(float));
    mem_read(&r, g_W1c, n*H*C*sizeof(float));
    mem_read(&r, g_b1, n*H*sizeof(float));
    mem_read(&r, g_W2, n*2*D*H*sizeof(float));
    mem_read(&r, g_W2c, n*2*D*C*sizeof(float));
    mem_read(&r, g_b2, n*2*D*sizeof(float));

    static maf_weights_t w;
    w.n_flows = n;
    w.param_dim = D;
    w.feature_dim = C;
    w.hidden_units = H;
    w.M1_data = g_M1;
    w.M2_data = g_M2;
    w.perm_data = g_perm;
    w.inv_perm_data = g_inv_perm;
    w.W1y_data = g_W1y;
    w.W1c_data = g_W1c;
    w.b1_data = g_b1;
    w.W2_data = g_W2;
    w.W2c_data = g_W2c;
    w.b2_data = g_b2;

    return maf_load_model(&w);
}

static void free_loaded_buffers() {
    if (g_M1) { free(g_M1); g_M1 = NULL; }
    if (g_M2) { free(g_M2); g_M2 = NULL; }
    if (g_perm) { free(g_perm); g_perm = NULL; }
    if (g_inv_perm) { free(g_inv_perm); g_inv_perm = NULL; }
    if (g_W1y) { free(g_W1y); g_W1y = NULL; }
    if (g_W1c) { free(g_W1c); g_W1c = NULL; }
    if (g_b1) { free(g_b1); g_b1 = NULL; }
    if (g_W2) { free(g_W2); g_W2 = NULL; }
    if (g_W2c) { free(g_W2c); g_W2c = NULL; }
    if (g_b2) { free(g_b2); g_b2 = NULL; }
}

/* =========================================================================
 * Tests
 * ========================================================================= */

void setUp(void) {
    /* Ensure globals are clean */
    g_M1 = NULL;
}

void tearDown(void) {
    free_loaded_buffers();
}

TEST_CASE("MAF Model Loading", "[maf]") {
    maf_model_t* model = load_model_from_blob(test_model_maf, test_model_maf_len);
    TEST_ASSERT_NOT_NULL(model);
    
    TEST_ASSERT_EQUAL(4, model->n_flows);
    TEST_ASSERT_EQUAL(2, model->param_dim);
    TEST_ASSERT_EQUAL(1, model->feature_dim);
    TEST_ASSERT_EQUAL(32, model->layers[0].hidden_units);

    maf_free_model(model);
}

TEST_CASE("MAF Inference Statistics & Timing", "[maf]") {
    maf_model_t* model = load_model_from_blob(test_model_maf, test_model_maf_len);
    TEST_ASSERT_NOT_NULL(model);

    /* Test Params */
    const int N_SAMPLES = 2000;
    const float feature[1] = {0.1f}; /* Arbitrary feature for conditioning */
    
    /* Allocate output */
    float* samples = malloc(N_SAMPLES * 2 * sizeof(float));
    TEST_ASSERT_NOT_NULL(samples);

    /* Performance Measurement */
    int64_t start = esp_timer_get_time();
    int ret = maf_sample(model, feature, N_SAMPLES, samples, 12345);
    int64_t end = esp_timer_get_time();
    
    TEST_ASSERT_EQUAL(0, ret);

    float total_time_ms = (end - start) / 1000.0f;
    float time_per_sample_us = (float)(end - start) / N_SAMPLES;
    
    ESP_LOGI(TAG, "Inference Time: %.2f ms for %d samples (%.2f us/sample)", 
             total_time_ms, N_SAMPLES, time_per_sample_us);

    /* Verify Performance: Should be < 1ms per sample on ESP32 (usually <100us) */
    /* QEMU might be slower or faster depending on host, but <1ms is safe bound */
    // TEST_ASSERT_LESS_THAN_FLOAT(1000.0f, time_per_sample_us);

    /* Statistics Verification */
    /* Baseline from setup_test_model.py:
       MEAN: ~0.53, ~0.25
       STD: ~0.57, ~0.28
    */
    
    float sum[2] = {0};
    float sq_sum[2] = {0};

    for(int i=0; i<N_SAMPLES; i++) {
        for(int d=0; d<2; d++) {
            float val = samples[i*2 + d];
            sum[d] += val;
            sq_sum[d] += val*val;
        }
    }

    float mean[2] = { sum[0]/N_SAMPLES, sum[1]/N_SAMPLES };
    float var[2] = { sq_sum[0]/N_SAMPLES - mean[0]*mean[0], sq_sum[1]/N_SAMPLES - mean[1]*mean[1] };
    float std[2] = { sqrtf(var[0]), sqrtf(var[1]) };

    ESP_LOGI(TAG, "Sample Stats: Mean=[%.4f, %.4f], Std=[%.4f, %.4f]", 
             mean[0], mean[1], std[0], std[1]);

    /* Check Means (allow +/- 0.1 deviation due to sampling noise N=2000) */
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.53f, mean[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.25f, mean[1]);

    /* Check Stds (allow +/- 0.1 deviation) */
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.57f, std[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.28f, std[1]);

    free(samples);
    maf_free_model(model);
}
