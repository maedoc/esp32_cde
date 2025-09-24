/**
 * @file cde_internal.h
 * @brief Internal definitions for CDE implementation
 */

#pragma once

#include "esp32_cde.h"

/**
 * @brief Internal CDE instance structure
 */
typedef struct {
    cde_config_t config;        ///< Configuration parameters
    cde_sample_t* samples;      ///< Sample buffer
    uint16_t sample_count;      ///< Current number of samples
    uint16_t buffer_index;      ///< Current buffer write position
} cde_instance_t;