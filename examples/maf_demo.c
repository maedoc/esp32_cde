/**
 * @file maf_demo.c
 * @brief Demonstration of MAF conditional density estimation on ESP32
 *
 * This example shows how to:
 * 1. Load a pre-trained MAF model
 * 2. Generate conditional samples
 * 3. Compute log probabilities
 * 4. Monitor memory usage
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "maf.h"

/* Include your exported model header here */
/* #include "maf_test_model.h" */

static const char *TAG = "maf_demo";

/**
 * Example: Manually create a small test model
 * In production, you would use an exported model from Python
 */
static void demo_maf_basic(void)
{
    ESP_LOGI(TAG, "=== MAF Basic Demo ===");

    /* In a real application, you would do:
     * maf_model_t* model = maf_load_model(&your_exported_weights);
     *
     * For this demo, we'll show the structure
     */

    ESP_LOGI(TAG, "Load a pre-trained model using:");
    ESP_LOGI(TAG, "  maf_model_t* model = maf_load_model(&model_weights);");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "The model_weights structure comes from:");
    ESP_LOGI(TAG, "  python export_maf_to_c.py --output model.h");
}

/**
 * Example: Generate conditional samples
 */
static void demo_maf_sampling(maf_model_t* model)
{
    if (model == NULL) {
        ESP_LOGW(TAG, "No model loaded, skipping sampling demo");
        return;
    }

    ESP_LOGI(TAG, "=== MAF Sampling Demo ===");

    /* Define conditioning features */
    float features[] = {0.5f};  /* feature_dim = 1 */

    /* Allocate output buffer */
    uint32_t n_samples = 5;
    uint16_t param_dim = model->param_dim;
    float* samples = (float*)malloc(n_samples * param_dim * sizeof(float));

    if (samples == NULL) {
        ESP_LOGE(TAG, "Failed to allocate sample buffer");
        return;
    }

    /* Generate samples */
    int64_t start_time = esp_timer_get_time();
    int ret = maf_sample(model, features, n_samples, samples, 42);
    int64_t end_time = esp_timer_get_time();

    if (ret != 0) {
        ESP_LOGE(TAG, "Sampling failed with code %d", ret);
        free(samples);
        return;
    }

    ESP_LOGI(TAG, "Generated %lu samples in %lld μs (%.2f ms)",
             n_samples,
             end_time - start_time,
             (end_time - start_time) / 1000.0);

    /* Print samples */
    ESP_LOGI(TAG, "Samples (conditioned on feature=%.2f):", features[0]);
    for (uint32_t i = 0; i < n_samples; i++) {
        ESP_LOGI(TAG, "  Sample %lu: [", i);
        for (uint16_t j = 0; j < param_dim; j++) {
            printf("%.3f", samples[i * param_dim + j]);
            if (j < param_dim - 1) printf(", ");
        }
        printf("]\n");
    }

    free(samples);
}

/**
 * Example: Compute log probability
 */
static void demo_maf_log_prob(maf_model_t* model)
{
    if (model == NULL) {
        ESP_LOGW(TAG, "No model loaded, skipping log prob demo");
        return;
    }

    ESP_LOGI(TAG, "=== MAF Log Probability Demo ===");

    /* Test points */
    float features[] = {0.5f};
    float test_points[][2] = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {-1.0f, 0.5f},
        {0.5f, -0.5f}
    };

    int n_tests = sizeof(test_points) / sizeof(test_points[0]);
    
    /* Create workspace */
    maf_workspace_t* ws = maf_create_workspace(model);
    if (ws == NULL) {
        ESP_LOGE(TAG, "Failed to allocate workspace");
        return;
    }

    ESP_LOGI(TAG, "Computing log p(y|x=%.2f):", features[0]);

    for (int i = 0; i < n_tests; i++) {
        int64_t start_time = esp_timer_get_time();
        float logp = maf_log_prob(model, ws, features, test_points[i]);
        int64_t end_time = esp_timer_get_time();

        ESP_LOGI(TAG, "  y=[%.2f, %.2f]: log_p = %.4f (%.1f μs)",
                 test_points[i][0],
                 test_points[i][1],
                 logp,
                 (float)(end_time - start_time));
    }
    
    maf_free_workspace(ws);
}

/**
 * Example: Memory usage and diagnostics
 */
static void demo_maf_diagnostics(maf_model_t* model)
{
    if (model == NULL) {
        ESP_LOGW(TAG, "No model loaded, skipping diagnostics");
        return;
    }

    ESP_LOGI(TAG, "=== MAF Model Diagnostics ===");

    /* Model configuration */
    ESP_LOGI(TAG, "Model configuration:");
    ESP_LOGI(TAG, "  n_flows: %u", model->n_flows);
    ESP_LOGI(TAG, "  param_dim: %u", model->param_dim);
    ESP_LOGI(TAG, "  feature_dim: %u", model->feature_dim);

    /* Memory usage */
    size_t mem_usage = maf_get_memory_usage(model);
    ESP_LOGI(TAG, "Memory usage: %zu bytes (%.2f KB)", mem_usage, mem_usage / 1024.0);

    /* Per-layer info */
    ESP_LOGI(TAG, "Layer details:");
    for (uint16_t i = 0; i < model->n_flows; i++) {
        maf_layer_t* layer = &model->layers[i];
        ESP_LOGI(TAG, "  Layer %u: hidden=%u, param=%u, feature=%u",
                 i, layer->hidden_units, layer->param_dim, layer->feature_dim);
    }

    /* System memory */
    ESP_LOGI(TAG, "System memory:");
    ESP_LOGI(TAG, "  Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "  Min free heap: %lu bytes", esp_get_minimum_free_heap_size());
}

/**
 * Main application
 */
void app_main(void)
{
    ESP_LOGI(TAG, "ESP32 MAF Conditional Density Estimation Demo");
    ESP_LOGI(TAG, "=============================================");

    /* Show basic usage */
    demo_maf_basic();

    /* For a real application with an exported model: */
    /*
    ESP_LOGI(TAG, "Loading MAF model...");
    maf_model_t* model = maf_load_model(&your_model_weights);

    if (model == NULL) {
        ESP_LOGE(TAG, "Failed to load model!");
        return;
    }

    ESP_LOGI(TAG, "Model loaded successfully");

    // Run demonstrations
    demo_maf_diagnostics(model);
    demo_maf_sampling(model);
    demo_maf_log_prob(model);

    // Cleanup
    maf_free_model(model);
    ESP_LOGI(TAG, "Model freed, demo complete");
    */

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "To use this demo:");
    ESP_LOGI(TAG, "1. Train a model: cd python && python export_maf_to_c.py");
    ESP_LOGI(TAG, "2. Include the generated header in this file");
    ESP_LOGI(TAG, "3. Uncomment the model loading code above");
    ESP_LOGI(TAG, "4. Build and flash to ESP32");

    /* Keep task alive */
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
