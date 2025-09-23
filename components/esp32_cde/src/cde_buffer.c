/**
 * @file cde_buffer.c
 * @brief Buffer management utilities for CDE
 */

#include "esp32_cde.h"
#include "cde_internal.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "cde_buffer";

/**
 * @brief Get the oldest sample from the circular buffer
 */
const cde_sample_t* cde_get_oldest_sample(cde_handle_t handle)
{
    if (handle == NULL) {
        return NULL;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    if (instance->sample_count == 0) {
        return NULL;
    }
    
    // If buffer is not full, oldest is at index 0
    // If buffer is full, oldest is at current write position
    int oldest_index = (instance->sample_count < instance->config.buffer_size) ? 
                       0 : instance->buffer_index;
    
    return &instance->samples[oldest_index];
}

/**
 * @brief Get the newest sample from the circular buffer
 */
const cde_sample_t* cde_get_newest_sample(cde_handle_t handle)
{
    if (handle == NULL) {
        return NULL;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    if (instance->sample_count == 0) {
        return NULL;
    }
    
    // Newest sample is always at (buffer_index - 1)
    int newest_index = (instance->buffer_index - 1 + instance->config.buffer_size) % 
                       instance->config.buffer_size;
    
    return &instance->samples[newest_index];
}

/**
 * @brief Get sample at specific index (for iteration)
 */
const cde_sample_t* cde_get_sample_at_index(cde_handle_t handle, uint16_t index)
{
    if (handle == NULL) {
        return NULL;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    
    if (index >= instance->sample_count) {
        return NULL;
    }
    
    // Calculate actual buffer index
    int buffer_index;
    if (instance->sample_count < instance->config.buffer_size) {
        // Buffer not full, simple indexing
        buffer_index = index;
    } else {
        // Buffer full, calculate from oldest position
        buffer_index = (instance->buffer_index + index) % instance->config.buffer_size;
    }
    
    return &instance->samples[buffer_index];
}

/**
 * @brief Check if the buffer is full
 */
bool cde_is_buffer_full(cde_handle_t handle)
{
    if (handle == NULL) {
        return false;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    return instance->sample_count >= instance->config.buffer_size;
}

/**
 * @brief Get buffer utilization percentage
 */
float cde_get_buffer_utilization(cde_handle_t handle)
{
    if (handle == NULL) {
        return 0.0f;
    }
    
    cde_instance_t* instance = (cde_instance_t*)handle;
    return (float)instance->sample_count / instance->config.buffer_size * 100.0f;
}