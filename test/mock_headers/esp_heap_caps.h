/**
 * @file esp_heap_caps.h
 * @brief Mock ESP-IDF heap capabilities for testing
 */

#pragma once

#include <stdlib.h>
#include <stdint.h>

#define MALLOC_CAP_8BIT     (1<<0)  ///< Memory must be 8bit accessible
#define MALLOC_CAP_32BIT    (1<<1)  ///< Memory must be 32bit accessible

// Mock heap capabilities functions
static inline void* heap_caps_malloc(size_t size, uint32_t caps) {
    (void)caps; // Ignore caps in mock
    return malloc(size);
}

static inline void heap_caps_free(void* ptr) {
    free(ptr);
}