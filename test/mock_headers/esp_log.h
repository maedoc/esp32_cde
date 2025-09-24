/**
 * @file esp_log.h
 * @brief Mock ESP-IDF logging for testing
 */

#pragma once

#include <stdio.h>
#include <stdint.h>

typedef enum {
    ESP_LOG_NONE,       ///< No log output
    ESP_LOG_ERROR,      ///< Critical errors, software module can not recover on its own
    ESP_LOG_WARN,       ///< Error conditions from which recovery measures have been taken
    ESP_LOG_INFO,       ///< Information messages which describe normal flow of events
    ESP_LOG_DEBUG,      ///< Extra information which is not necessary for normal use (values, pointers, sizes, etc).
    ESP_LOG_VERBOSE     ///< Bigger chunks of debugging information, or frequent messages which can potentially flood the output.
} esp_log_level_t;

#define ESP_LOGE(tag, format, ...) printf("E (%u) %s: " format "\n", esp_log_timestamp(), tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, format, ...) printf("W (%u) %s: " format "\n", esp_log_timestamp(), tag, ##__VA_ARGS__)
#define ESP_LOGI(tag, format, ...) printf("I (%u) %s: " format "\n", esp_log_timestamp(), tag, ##__VA_ARGS__)
#define ESP_LOGD(tag, format, ...) printf("D (%u) %s: " format "\n", esp_log_timestamp(), tag, ##__VA_ARGS__)
#define ESP_LOGV(tag, format, ...) printf("V (%u) %s: " format "\n", esp_log_timestamp(), tag, ##__VA_ARGS__)

// Mock timestamp function
static inline uint32_t esp_log_timestamp(void) {
    return 0; // Simplified mock
}