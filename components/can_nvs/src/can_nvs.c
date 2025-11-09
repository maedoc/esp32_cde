/**
 * @file can_nvs.c
 * @brief CAN Frame NVS Storage Component Implementation
 */

#include "can_nvs.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"
#include <string.h>
#include <stdlib.h>

static const char *TAG = "can_nvs";

/**
 * @brief NVS namespace for CAN storage
 */
#define CAN_NVS_NAMESPACE "can_storage"

/**
 * @brief Packed CAN frame structure for storage
 * This structure is optimized for minimal storage size
 */
typedef struct __attribute__((packed)) {
    uint32_t identifier;        /**< CAN ID */
    uint8_t data_length_code;   /**< DLC */
    uint8_t flags;              /**< Flags */
    uint8_t data[CAN_NVS_MAX_DATA_LEN];  /**< Data payload */
} can_nvs_packed_frame_t;

/**
 * @brief Storage header for frame sequences
 */
typedef struct __attribute__((packed)) {
    uint16_t frame_count;       /**< Number of frames */
    uint16_t checksum;          /**< Simple checksum for validation */
} can_nvs_storage_header_t;

/**
 * @brief NVS handle (0 if not initialized)
 */
static nvs_handle_t nvs_handle = 0;

/**
 * @brief Calculate a simple checksum for validation
 */
static uint16_t calculate_checksum(const uint8_t *data, size_t len) {
    uint16_t checksum = 0;
    for (size_t i = 0; i < len; i++) {
        checksum += data[i];
    }
    return checksum;
}

esp_err_t can_nvs_init(const char *partition_name) {
    esp_err_t ret;

    // Initialize NVS if not already done
    ret = nvs_flash_init_partition(partition_name ? partition_name : "nvs");
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition was truncated and needs to be erased
        ESP_LOGW(TAG, "NVS partition needs erasing, erasing...");
        ESP_ERROR_CHECK(nvs_flash_erase_partition(partition_name ? partition_name : "nvs"));
        ret = nvs_flash_init_partition(partition_name ? partition_name : "nvs");
    }

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize NVS partition: %s", esp_err_to_name(ret));
        return ret;
    }

    // Open NVS handle
    ret = nvs_open_from_partition(
        partition_name ? partition_name : "nvs",
        CAN_NVS_NAMESPACE,
        NVS_READWRITE,
        &nvs_handle
    );

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "CAN NVS storage initialized successfully");
    return ESP_OK;
}

esp_err_t can_nvs_deinit(void) {
    if (nvs_handle != 0) {
        nvs_close(nvs_handle);
        nvs_handle = 0;
    }
    return ESP_OK;
}

esp_err_t can_nvs_store_sequence(const char *key, const can_nvs_sequence_t *sequence) {
    if (key == NULL || sequence == NULL || sequence->frames == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (strlen(key) > CAN_NVS_MAX_KEY_LEN) {
        ESP_LOGE(TAG, "Key too long (max %d characters)", CAN_NVS_MAX_KEY_LEN);
        return ESP_ERR_INVALID_ARG;
    }

    if (sequence->count == 0 || sequence->count > CAN_NVS_MAX_FRAMES_PER_SEQUENCE) {
        ESP_LOGE(TAG, "Invalid frame count: %d", sequence->count);
        return ESP_ERR_INVALID_ARG;
    }

    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    // Calculate total size needed
    size_t header_size = sizeof(can_nvs_storage_header_t);
    size_t frames_size = sequence->count * sizeof(can_nvs_packed_frame_t);
    size_t total_size = header_size + frames_size;

    // Allocate buffer for serialization
    uint8_t *buffer = (uint8_t *)malloc(total_size);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate serialization buffer");
        return ESP_ERR_NO_MEM;
    }

    // Pack header
    can_nvs_storage_header_t *header = (can_nvs_storage_header_t *)buffer;
    header->frame_count = sequence->count;

    // Pack frames
    can_nvs_packed_frame_t *packed_frames = (can_nvs_packed_frame_t *)(buffer + header_size);
    for (uint16_t i = 0; i < sequence->count; i++) {
        packed_frames[i].identifier = sequence->frames[i].identifier;
        packed_frames[i].data_length_code = sequence->frames[i].data_length_code;
        packed_frames[i].flags = sequence->frames[i].flags;
        memcpy(packed_frames[i].data, sequence->frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    // Calculate checksum (excluding the checksum field itself)
    header->checksum = calculate_checksum(buffer + sizeof(uint16_t), total_size - sizeof(uint16_t));

    // Write to NVS
    esp_err_t ret = nvs_set_blob(nvs_handle, key, buffer, total_size);
    free(buffer);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write sequence to NVS: %s", esp_err_to_name(ret));
        return ret;
    }

    // Commit changes
    ret = nvs_commit(nvs_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Stored %d CAN frames with key '%s'", sequence->count, key);
    return ESP_OK;
}

esp_err_t can_nvs_load_sequence(const char *key, can_nvs_sequence_t *sequence) {
    if (key == NULL || sequence == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    // Get the size of the stored blob
    size_t required_size = 0;
    esp_err_t ret = nvs_get_blob(nvs_handle, key, NULL, &required_size);
    if (ret != ESP_OK) {
        if (ret != ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to get blob size: %s", esp_err_to_name(ret));
        }
        return ret;
    }

    // Validate size
    if (required_size < sizeof(can_nvs_storage_header_t)) {
        ESP_LOGE(TAG, "Invalid stored data size");
        return ESP_ERR_INVALID_SIZE;
    }

    // Allocate buffer
    uint8_t *buffer = (uint8_t *)malloc(required_size);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate load buffer");
        return ESP_ERR_NO_MEM;
    }

    // Read from NVS
    ret = nvs_get_blob(nvs_handle, key, buffer, &required_size);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read blob: %s", esp_err_to_name(ret));
        free(buffer);
        return ret;
    }

    // Parse header
    can_nvs_storage_header_t *header = (can_nvs_storage_header_t *)buffer;
    uint16_t stored_checksum = header->checksum;
    uint16_t calculated_checksum = calculate_checksum(
        buffer + sizeof(uint16_t),
        required_size - sizeof(uint16_t)
    );

    if (stored_checksum != calculated_checksum) {
        ESP_LOGE(TAG, "Checksum mismatch (stored: 0x%04X, calculated: 0x%04X)",
                 stored_checksum, calculated_checksum);
        free(buffer);
        return ESP_ERR_INVALID_SIZE;
    }

    // Validate frame count
    if (header->frame_count == 0 || header->frame_count > CAN_NVS_MAX_FRAMES_PER_SEQUENCE) {
        ESP_LOGE(TAG, "Invalid frame count in stored data: %d", header->frame_count);
        free(buffer);
        return ESP_ERR_INVALID_SIZE;
    }

    // Allocate frames array
    sequence->frames = (can_nvs_frame_t *)malloc(header->frame_count * sizeof(can_nvs_frame_t));
    if (sequence->frames == NULL) {
        ESP_LOGE(TAG, "Failed to allocate frames array");
        free(buffer);
        return ESP_ERR_NO_MEM;
    }

    // Unpack frames
    can_nvs_packed_frame_t *packed_frames =
        (can_nvs_packed_frame_t *)(buffer + sizeof(can_nvs_storage_header_t));

    for (uint16_t i = 0; i < header->frame_count; i++) {
        sequence->frames[i].identifier = packed_frames[i].identifier;
        sequence->frames[i].data_length_code = packed_frames[i].data_length_code;
        sequence->frames[i].flags = packed_frames[i].flags;
        memcpy(sequence->frames[i].data, packed_frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    sequence->count = header->frame_count;

    free(buffer);
    ESP_LOGI(TAG, "Loaded %d CAN frames with key '%s'", sequence->count, key);
    return ESP_OK;
}

esp_err_t can_nvs_delete_sequence(const char *key) {
    if (key == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    esp_err_t ret = nvs_erase_key(nvs_handle, key);
    if (ret != ESP_OK) {
        if (ret != ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to delete sequence: %s", esp_err_to_name(ret));
        }
        return ret;
    }

    ret = nvs_commit(nvs_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Deleted sequence with key '%s'", key);
    return ESP_OK;
}

void can_nvs_free_sequence(can_nvs_sequence_t *sequence) {
    if (sequence != NULL && sequence->frames != NULL) {
        free(sequence->frames);
        sequence->frames = NULL;
        sequence->count = 0;
    }
}

esp_err_t can_nvs_key_exists(const char *key, bool *exists) {
    if (key == NULL || exists == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    size_t required_size = 0;
    esp_err_t ret = nvs_get_blob(nvs_handle, key, NULL, &required_size);

    if (ret == ESP_OK) {
        *exists = true;
    } else if (ret == ESP_ERR_NVS_NOT_FOUND) {
        *exists = false;
        ret = ESP_OK;  // Not an error, just doesn't exist
    } else {
        ESP_LOGE(TAG, "Error checking key existence: %s", esp_err_to_name(ret));
    }

    return ret;
}

esp_err_t can_nvs_get_sequence_size(const char *key, uint16_t *frame_count) {
    if (key == NULL || frame_count == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    // We only need to read the header to get the frame count
    can_nvs_storage_header_t header;
    size_t header_size = sizeof(header);

    // Get the full size first
    size_t required_size = 0;
    esp_err_t ret = nvs_get_blob(nvs_handle, key, NULL, &required_size);
    if (ret != ESP_OK) {
        return ret;
    }

    if (required_size < sizeof(header)) {
        return ESP_ERR_INVALID_SIZE;
    }

    // Allocate and read just enough to get the header
    uint8_t *buffer = (uint8_t *)malloc(required_size);
    if (buffer == NULL) {
        return ESP_ERR_NO_MEM;
    }

    ret = nvs_get_blob(nvs_handle, key, buffer, &required_size);
    if (ret == ESP_OK) {
        memcpy(&header, buffer, sizeof(header));
        *frame_count = header.frame_count;
    }

    free(buffer);
    return ret;
}

esp_err_t can_nvs_erase_all(void) {
    if (nvs_handle == 0) {
        ESP_LOGE(TAG, "NVS not initialized");
        return ESP_ERR_NVS_INVALID_HANDLE;
    }

    esp_err_t ret = nvs_erase_all(nvs_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to erase all: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = nvs_commit(nvs_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to commit NVS changes: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Erased all CAN sequences");
    return ESP_OK;
}
