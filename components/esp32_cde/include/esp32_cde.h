/**
 * @file esp32_cde.h
 * @brief ESP32 Conditional Density Estimation Library
 * 
 * This library provides conditional density estimation capabilities for ESP32,
 * useful for probabilistic modeling and machine learning applications.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CDE configuration structure
 */
typedef struct {
    uint16_t max_features;      ///< Maximum number of input features
    uint16_t buffer_size;       ///< Size of the sample buffer
    float learning_rate;        ///< Learning rate for adaptive algorithms
    bool use_float_precision;   ///< Use single precision (true) or double (false)
} cde_config_t;

/**
 * @brief CDE handle type
 */
typedef void* cde_handle_t;

/**
 * @brief Sample data structure
 */
typedef struct {
    float* features;    ///< Input feature vector
    uint16_t num_features;  ///< Number of features
    float target;       ///< Target value
    uint32_t timestamp; ///< Sample timestamp
} cde_sample_t;

/**
 * @brief Initialize CDE instance
 * 
 * @param config Configuration parameters
 * @return CDE handle on success, NULL on failure
 */
cde_handle_t cde_init(const cde_config_t* config);

/**
 * @brief Deinitialize CDE instance
 * 
 * @param handle CDE handle to deinitialize
 * @return ESP_OK on success
 */
esp_err_t cde_deinit(cde_handle_t handle);

/**
 * @brief Add a training sample
 * 
 * @param handle CDE handle
 * @param features Input feature vector
 * @param num_features Number of features
 * @param target Target value
 * @return ESP_OK on success
 */
esp_err_t cde_add_sample(cde_handle_t handle, const float* features, 
                        uint16_t num_features, float target);

/**
 * @brief Get conditional density estimate
 * 
 * @param handle CDE handle
 * @param features Input feature vector
 * @param num_features Number of features
 * @param target Target value to estimate density for
 * @return Density estimate (probability density value)
 */
float cde_get_density(cde_handle_t handle, const float* features, 
                     uint16_t num_features, float target);

/**
 * @brief Get number of samples in the buffer
 * 
 * @param handle CDE handle
 * @return Number of samples
 */
uint16_t cde_get_sample_count(cde_handle_t handle);

/**
 * @brief Reset/clear all samples
 * 
 * @param handle CDE handle
 * @return ESP_OK on success
 */
esp_err_t cde_reset(cde_handle_t handle);

/**
 * @brief Get memory usage statistics
 * 
 * @param handle CDE handle
 * @return Memory usage in bytes
 */
size_t cde_get_memory_usage(cde_handle_t handle);

#ifdef __cplusplus
}
#endif