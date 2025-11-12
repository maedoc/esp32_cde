/**
 * @file esp32_cde.c
 * @brief ESP32 MAF Conditional Density Estimation - Main Application
 *
 * This is a minimal main application for the ESP32 MAF library.
 * For a complete example of how to use MAF, see examples/maf_demo.c
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "maf.h"

static const char *TAG = "esp32_cde";

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32 MAF Conditional Density Estimation");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "This project provides MAF (Masked Autoregressive Flow)");
    ESP_LOGI(TAG, "inference capabilities for ESP32.");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Usage:");
    ESP_LOGI(TAG, "  1. Train a MAF model in Python:");
    ESP_LOGI(TAG, "     cd python && python export_maf_to_c.py --output my_model.h");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  2. Include the model in your code:");
    ESP_LOGI(TAG, "     #include \"my_model.h\"");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  3. Load and use:");
    ESP_LOGI(TAG, "     maf_model_t* model = maf_load_model(&my_model_weights);");
    ESP_LOGI(TAG, "     maf_sample(model, features, n_samples, samples_out, seed);");
    ESP_LOGI(TAG, "     maf_free_model(model);");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "See examples/maf_demo.c for complete usage examples.");
    ESP_LOGI(TAG, "See MAF_QUICKSTART.md for detailed documentation.");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "System Info:");
    ESP_LOGI(TAG, "  Free heap: %lu bytes", esp_get_free_heap_size());
    ESP_LOGI(TAG, "  ESP-IDF version: %s", esp_get_idf_version());

    // Keep task alive
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
