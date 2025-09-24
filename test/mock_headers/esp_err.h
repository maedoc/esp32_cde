/**
 * @file esp_err.h
 * @brief Mock ESP-IDF error definitions for testing
 */

#pragma once

#include <stdint.h>

typedef int32_t esp_err_t;

#define ESP_OK          0       ///< esp_err_t value indicating success
#define ESP_FAIL        -1      ///< Generic esp_err_t code indicating failure

#define ESP_ERR_NO_MEM          0x101   ///< Out of memory
#define ESP_ERR_INVALID_ARG     0x102   ///< Invalid argument
#define ESP_ERR_INVALID_STATE   0x103   ///< Invalid state
#define ESP_ERR_INVALID_SIZE    0x104   ///< Invalid size
#define ESP_ERR_NOT_FOUND       0x105   ///< Requested resource not found
#define ESP_ERR_NOT_SUPPORTED   0x106   ///< Operation or feature not supported
#define ESP_ERR_TIMEOUT         0x107   ///< Operation timed out

const char* esp_err_to_name(esp_err_t code);

// Mock implementation
static inline const char* esp_err_to_name_impl(esp_err_t code) {
    switch (code) {
        case ESP_OK: return "ESP_OK";
        case ESP_FAIL: return "ESP_FAIL";
        case ESP_ERR_NO_MEM: return "ESP_ERR_NO_MEM";
        case ESP_ERR_INVALID_ARG: return "ESP_ERR_INVALID_ARG";
        case ESP_ERR_INVALID_STATE: return "ESP_ERR_INVALID_STATE";
        case ESP_ERR_INVALID_SIZE: return "ESP_ERR_INVALID_SIZE";
        case ESP_ERR_NOT_FOUND: return "ESP_ERR_NOT_FOUND";
        case ESP_ERR_NOT_SUPPORTED: return "ESP_ERR_NOT_SUPPORTED";
        case ESP_ERR_TIMEOUT: return "ESP_ERR_TIMEOUT";
        default: return "UNKNOWN_ERROR";
    }
}

// For testing, map function to implementation
#define esp_err_to_name esp_err_to_name_impl