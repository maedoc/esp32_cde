/**
 * @file cde_core.c
 * @brief Core implementation of conditional density estimation
 */

#include "esp32_cde.h"
#include "cde_internal.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <math.h>
#include <string.h>

static const char *TAG = "cde_core";

cde_handle_t cde_init(const cde_config_t* config)
{
    if (config == NULL) {
        ESP_LOGE(TAG, "Invalid configuration");
        return NULL;
    }
    
    if (config->max_features == 0 || config->buffer_size == 0) {
        ESP_LOGE(TAG, "Invalid configuration parameters");
        return NULL;
    }
    
    cde_instance_t* instance = heap_caps_malloc(sizeof(cde_instance_t), MALLOC_CAP_8BIT);
    if (instance == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for CDE instance");
        return NULL;
    }
    
    memcpy(&instance->config, config, sizeof(cde_config_t));
    instance->sample_count = 0;
    instance->buffer_index = 0;
    
    // Allocate sample buffer
    instance->samples = heap_caps_malloc(sizeof(cde_sample_t) * config->buffer_size, MALLOC_CAP_8BIT);
    if (instance->samples == NULL) {
        ESP_LOGE(TAG, "Failed to allocate sample buffer");
        free(instance);
        return NULL;
    }
    
    // Initialize samples
    for (int i = 0; i < config->buffer_size; i++) {
        instance->samples[i].features = heap_caps_malloc(sizeof(float) * config->max_features, MALLOC_CAP_8BIT);
        if (instance->samples[i].features == NULL) {
            ESP_LOGE(TAG, "Failed to allocate feature buffer for sample %d", i);
            // Cleanup previous allocations
            for (int j = 0; j < i; j++) {
                heap_caps_free(instance->samples[j].features);
            }
            heap_caps_free(instance->samples);
            heap_caps_free(instance);
            return NULL;
        }
        instance->samples[i].num_features = 0;
        instance->samples[i].target = 0.0f;
        instance->samples[i].timestamp = 0;
    }
    
    ESP_LOGI(TAG, "CDE initialized: max_features=%d, buffer_size=%d", 
             config->max_features, config->buffer_size);
    
    return (cde_handle_t)instance;
}

esp_err_t cde_deinit(cde_handle_t handle)
{
    if (handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    // Free feature buffers
    for (int i = 0; i < instance->config.buffer_size; i++) {
        if (instance->samples[i].features != NULL) {
            heap_caps_free(instance->samples[i].features);
        }
    }
    
    // Free sample buffer
    if (instance->samples != NULL) {
        heap_caps_free(instance->samples);
    }
    
    // Free instance
    heap_caps_free(instance);
    
    ESP_LOGI(TAG, "CDE deinitialized");
    return ESP_OK;
}

esp_err_t cde_add_sample(cde_handle_t handle, const float* features, 
                        uint16_t num_features, float target)
{
    if (handle == NULL || features == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    if (num_features > instance->config.max_features) {
        ESP_LOGE(TAG, "Number of features (%d) exceeds maximum (%d)", 
                 num_features, instance->config.max_features);
        return ESP_ERR_INVALID_SIZE;
    }
    
    // Get current sample slot (circular buffer)
    cde_sample_t* sample = &instance->samples[instance->buffer_index];
    
    // Copy features
    memcpy(sample->features, features, sizeof(float) * num_features);
    sample->num_features = num_features;
    sample->target = target;
    sample->timestamp = esp_log_timestamp();
    
    // Update buffer management
    instance->buffer_index = (instance->buffer_index + 1) % instance->config.buffer_size;
    if (instance->sample_count < instance->config.buffer_size) {
        instance->sample_count++;
    }
    
    ESP_LOGD(TAG, "Added sample: features=%d, target=%.3f, count=%d", 
             num_features, target, instance->sample_count);
    
    return ESP_OK;
}

float cde_get_density(cde_handle_t handle, const float* features, 
                     uint16_t num_features, float target)
{
    if (handle == NULL || features == NULL) {
        return 0.0f;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    if (instance->sample_count == 0) {
        ESP_LOGW(TAG, "No samples available for density estimation");
        return 0.0f;
    }
    
    if (num_features > instance->config.max_features) {
        ESP_LOGE(TAG, "Number of features exceeds maximum");
        return 0.0f;
    }
    
    // Simple kernel density estimation implementation
    float density = 0.0f;
    float bandwidth = 0.5f; // Fixed bandwidth for simplicity
    int valid_samples = 0;
    
    for (int i = 0; i < instance->sample_count; i++) {
        cde_sample_t* sample = &instance->samples[i];
        
        if (sample->num_features != num_features) {
            continue; // Skip samples with different feature dimensions
        }
        
        // Calculate feature distance (Euclidean)
        float feature_dist = 0.0f;
        for (int j = 0; j < num_features; j++) {
            float diff = features[j] - sample->features[j];
            feature_dist += diff * diff;
        }
        feature_dist = sqrtf(feature_dist);
        
        // Calculate target distance
        float target_dist = fabsf(target - sample->target);
        
        // Gaussian kernel
        float kernel_value = expf(-(feature_dist + target_dist) / (2.0f * bandwidth * bandwidth));
        density += kernel_value;
        valid_samples++;
    }
    
    if (valid_samples > 0) {
        density /= (valid_samples * bandwidth * sqrtf(2.0f * M_PI));
    }
    
    ESP_LOGD(TAG, "Density estimate: %.6f (from %d samples)", density, valid_samples);
    
    return density;
}

uint16_t cde_get_sample_count(cde_handle_t handle)
{
    if (handle == NULL) {
        return 0;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    return instance->sample_count;
}

esp_err_t cde_reset(cde_handle_t handle)
{
    if (handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    instance->sample_count = 0;
    instance->buffer_index = 0;
    
    ESP_LOGI(TAG, "CDE reset - all samples cleared");
    return ESP_OK;
}

size_t cde_get_memory_usage(cde_handle_t handle)
{
    if (handle == NULL) {
        return 0;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    size_t usage = sizeof(cde_instance_t);
    usage += sizeof(cde_sample_t) * instance->config.buffer_size;
    usage += sizeof(float) * instance->config.max_features * instance->config.buffer_size;
    
    return usage;
}