/**
 * @file cde_math.c
 * @brief Mathematical utilities for CDE
 */

#include "esp32_cde.h"
#include "esp_log.h"
#include <math.h>

static const char *TAG = "cde_math";

/**
 * @brief Calculate Gaussian kernel value
 */
float cde_gaussian_kernel(float distance, float bandwidth)
{
    if (bandwidth <= 0.0f) {
        return 0.0f;
    }
    
    float normalized_dist = distance / bandwidth;
    return expf(-0.5f * normalized_dist * normalized_dist) / 
           (bandwidth * sqrtf(2.0f * M_PI));
}

/**
 * @brief Calculate Euclidean distance between two feature vectors
 */
float cde_euclidean_distance(const float* features1, const float* features2, uint16_t num_features)
{
    if (features1 == NULL || features2 == NULL || num_features == 0) {
        return INFINITY;
    }
    
    float sum_squared = 0.0f;
    for (uint16_t i = 0; i < num_features; i++) {
        float diff = features1[i] - features2[i];
        sum_squared += diff * diff;
    }
    
    return sqrtf(sum_squared);
}

/**
 * @brief Calculate Manhattan distance between two feature vectors
 */
float cde_manhattan_distance(const float* features1, const float* features2, uint16_t num_features)
{
    if (features1 == NULL || features2 == NULL || num_features == 0) {
        return INFINITY;
    }
    
    float sum = 0.0f;
    for (uint16_t i = 0; i < num_features; i++) {
        sum += fabsf(features1[i] - features2[i]);
    }
    
    return sum;
}

/**
 * @brief Normalize a feature vector to unit length
 */
esp_err_t cde_normalize_features(float* features, uint16_t num_features)
{
    if (features == NULL || num_features == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Calculate magnitude
    float magnitude = 0.0f;
    for (uint16_t i = 0; i < num_features; i++) {
        magnitude += features[i] * features[i];
    }
    magnitude = sqrtf(magnitude);
    
    if (magnitude < 1e-8f) {
        ESP_LOGW(TAG, "Feature vector has near-zero magnitude");
        return ESP_ERR_INVALID_STATE;
    }
    
    // Normalize
    for (uint16_t i = 0; i < num_features; i++) {
        features[i] /= magnitude;
    }
    
    return ESP_OK;
}