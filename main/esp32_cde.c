#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp32_cde.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32 Conditional Density Estimation Demo");
    
    // Initialize CDE component
    cde_config_t config = {
        .max_features = 16,
        .buffer_size = 128,
        .learning_rate = 0.01f
    };
    
    cde_handle_t cde_handle = cde_init(&config);
    if (cde_handle == NULL) {
        ESP_LOGE(TAG, "Failed to initialize CDE");
        return;
    }
    
    ESP_LOGI(TAG, "CDE initialized successfully");
    
    // Demo: Add some sample data points
    float features[] = {1.0f, 2.0f, 3.0f};
    float target = 5.0f;
    
    esp_err_t ret = cde_add_sample(cde_handle, features, 3, target);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Sample added successfully");
    } else {
        ESP_LOGE(TAG, "Failed to add sample: %s", esp_err_to_name(ret));
    }
    
    // Demo: Get density estimate
    float density = cde_get_density(cde_handle, features, 3, target);
    ESP_LOGI(TAG, "Density estimate: %.6f", density);
    
    // Cleanup
    cde_deinit(cde_handle);
    ESP_LOGI(TAG, "CDE demo completed");
    
    // Keep the task running
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}