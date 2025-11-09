/**
 * @file can_nvs.c
 * @brief CAN Frame NVS Storage Component Implementation (Optimized)
 *
 * This implementation uses two optimization techniques:
 * 1. Frame ID Dictionary: Stores unique IDs once, references by 1-byte index
 * 2. Frame Deduplication: Stores unique frames once, sequence is array of indices
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
 * @brief Version byte for storage format
 */
#define CAN_NVS_FORMAT_VERSION 2

/**
 * @brief Optimized storage header
 */
typedef struct __attribute__((packed)) {
    uint8_t version;                /**< Storage format version */
    uint16_t sequence_length;       /**< Original sequence length */
    uint8_t unique_id_count;        /**< Number of unique CAN IDs (max 256) */
    uint8_t unique_frame_count;     /**< Number of unique frames (max 256) */
    uint16_t checksum;              /**< Simple checksum for validation */
} optimized_header_t;

/**
 * @brief Optimized frame storage (without ID, using index)
 */
typedef struct __attribute__((packed)) {
    uint8_t id_index;               /**< Index into ID dictionary */
    uint8_t data_length_code;       /**< DLC */
    uint8_t flags;                  /**< Flags */
    uint8_t data[CAN_NVS_MAX_DATA_LEN];  /**< Data payload */
} optimized_frame_t;

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

/**
 * @brief Calculate hash for a frame (for deduplication)
 */
static uint32_t hash_frame(const can_nvs_frame_t *frame) {
    // Simple FNV-1a hash
    uint32_t hash = 2166136261u;
    const uint8_t *data = (const uint8_t *)frame;

    // Hash identifier
    for (int i = 0; i < 4; i++) {
        hash ^= ((uint8_t *)&frame->identifier)[i];
        hash *= 16777619u;
    }

    // Hash DLC and flags
    hash ^= frame->data_length_code;
    hash *= 16777619u;
    hash ^= frame->flags;
    hash *= 16777619u;

    // Hash data bytes (only up to DLC)
    for (int i = 0; i < frame->data_length_code && i < CAN_NVS_MAX_DATA_LEN; i++) {
        hash ^= frame->data[i];
        hash *= 16777619u;
    }

    return hash;
}

/**
 * @brief Compare two frames for equality
 */
static bool frames_equal(const can_nvs_frame_t *a, const can_nvs_frame_t *b) {
    if (a->identifier != b->identifier ||
        a->data_length_code != b->data_length_code ||
        a->flags != b->flags) {
        return false;
    }

    return memcmp(a->data, b->data, CAN_NVS_MAX_DATA_LEN) == 0;
}

/**
 * @brief Find index of ID in dictionary, or add if not present
 */
static int find_or_add_id(uint32_t id, uint32_t *id_dict, uint8_t *id_count) {
    // Search for existing ID
    for (uint8_t i = 0; i < *id_count; i++) {
        if (id_dict[i] == id) {
            return i;
        }
    }

    // Add new ID if space available
    if (*id_count < 256) {
        id_dict[*id_count] = id;
        return (*id_count)++;
    }

    return -1; // Dictionary full
}

/**
 * @brief Find index of frame in unique frames array, or add if not present
 */
static int find_or_add_frame(const can_nvs_frame_t *frame,
                              can_nvs_frame_t *unique_frames,
                              uint8_t *unique_count,
                              uint32_t *id_dict,
                              uint8_t *id_count) {
    uint32_t frame_hash = hash_frame(frame);

    // Search for existing frame
    for (uint8_t i = 0; i < *unique_count; i++) {
        if (hash_frame(&unique_frames[i]) == frame_hash &&
            frames_equal(&unique_frames[i], frame)) {
            return i;
        }
    }

    // Add new frame if space available
    if (*unique_count < 256) {
        // Ensure ID is in dictionary
        int id_idx = find_or_add_id(frame->identifier, id_dict, id_count);
        if (id_idx < 0) {
            return -1; // ID dictionary full
        }

        unique_frames[*unique_count] = *frame;
        return (*unique_count)++;
    }

    return -1; // Unique frames array full
}

esp_err_t can_nvs_init(const char *partition_name) {
    esp_err_t ret;

    // Initialize NVS if not already done
    ret = nvs_flash_init_partition(partition_name ? partition_name : "nvs");
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
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

    ESP_LOGI(TAG, "CAN NVS storage initialized (optimized format v%d)", CAN_NVS_FORMAT_VERSION);
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

    // Build ID dictionary and unique frames
    uint32_t *id_dict = (uint32_t *)malloc(256 * sizeof(uint32_t));
    can_nvs_frame_t *unique_frames = (can_nvs_frame_t *)malloc(256 * sizeof(can_nvs_frame_t));
    uint8_t *frame_indices = (uint8_t *)malloc(sequence->count * sizeof(uint8_t));

    if (!id_dict || !unique_frames || !frame_indices) {
        ESP_LOGE(TAG, "Failed to allocate temporary buffers");
        free(id_dict);
        free(unique_frames);
        free(frame_indices);
        return ESP_ERR_NO_MEM;
    }

    uint8_t id_count = 0;
    uint8_t unique_count = 0;

    // Process each frame
    for (uint16_t i = 0; i < sequence->count; i++) {
        int frame_idx = find_or_add_frame(&sequence->frames[i], unique_frames,
                                          &unique_count, id_dict, &id_count);
        if (frame_idx < 0) {
            ESP_LOGE(TAG, "Too many unique frames or IDs (max 256 each)");
            free(id_dict);
            free(unique_frames);
            free(frame_indices);
            return ESP_ERR_INVALID_ARG;
        }
        frame_indices[i] = (uint8_t)frame_idx;
    }

    // Calculate storage size
    size_t header_size = sizeof(optimized_header_t);
    size_t id_dict_size = id_count * sizeof(uint32_t);
    size_t unique_frames_size = unique_count * sizeof(optimized_frame_t);
    size_t indices_size = sequence->count * sizeof(uint8_t);
    size_t total_size = header_size + id_dict_size + unique_frames_size + indices_size;

    ESP_LOGI(TAG, "Compression: %d frames -> %d unique IDs, %d unique frames",
             sequence->count, id_count, unique_count);

    // Allocate serialization buffer
    uint8_t *buffer = (uint8_t *)malloc(total_size);
    if (buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate serialization buffer (%d bytes)", total_size);
        free(id_dict);
        free(unique_frames);
        free(frame_indices);
        return ESP_ERR_NO_MEM;
    }

    uint8_t *ptr = buffer;

    // Write header
    optimized_header_t *header = (optimized_header_t *)ptr;
    header->version = CAN_NVS_FORMAT_VERSION;
    header->sequence_length = sequence->count;
    header->unique_id_count = id_count;
    header->unique_frame_count = unique_count;
    ptr += sizeof(optimized_header_t);

    // Write ID dictionary
    memcpy(ptr, id_dict, id_dict_size);
    ptr += id_dict_size;

    // Write unique frames (with ID indices)
    for (uint8_t i = 0; i < unique_count; i++) {
        optimized_frame_t *opt_frame = (optimized_frame_t *)ptr;

        // Find ID index for this frame
        uint8_t id_idx = 0;
        for (uint8_t j = 0; j < id_count; j++) {
            if (id_dict[j] == unique_frames[i].identifier) {
                id_idx = j;
                break;
            }
        }

        opt_frame->id_index = id_idx;
        opt_frame->data_length_code = unique_frames[i].data_length_code;
        opt_frame->flags = unique_frames[i].flags;
        memcpy(opt_frame->data, unique_frames[i].data, CAN_NVS_MAX_DATA_LEN);

        ptr += sizeof(optimized_frame_t);
    }

    // Write frame indices
    memcpy(ptr, frame_indices, indices_size);
    ptr += indices_size;

    // Calculate and store checksum
    header->checksum = calculate_checksum(buffer + offsetof(optimized_header_t, sequence_length),
                                         total_size - offsetof(optimized_header_t, sequence_length));

    // Write to NVS
    esp_err_t ret = nvs_set_blob(nvs_handle, key, buffer, total_size);

    // Calculate compression ratio
    size_t uncompressed_size = 4 + sequence->count * 14; // Old format
    float compression_ratio = 100.0f * (1.0f - (float)total_size / uncompressed_size);

    ESP_LOGI(TAG, "Storage: %d bytes (was %d bytes, %.1f%% reduction)",
             total_size, uncompressed_size, compression_ratio);

    // Cleanup
    free(buffer);
    free(id_dict);
    free(unique_frames);
    free(frame_indices);

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
    if (required_size < sizeof(optimized_header_t)) {
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
    optimized_header_t *header = (optimized_header_t *)buffer;

    // Check version
    if (header->version != CAN_NVS_FORMAT_VERSION) {
        ESP_LOGE(TAG, "Unsupported format version: %d", header->version);
        free(buffer);
        return ESP_ERR_INVALID_SIZE;
    }

    // Verify checksum
    uint16_t stored_checksum = header->checksum;
    uint16_t calculated_checksum = calculate_checksum(
        buffer + offsetof(optimized_header_t, sequence_length),
        required_size - offsetof(optimized_header_t, sequence_length)
    );

    if (stored_checksum != calculated_checksum) {
        ESP_LOGE(TAG, "Checksum mismatch (stored: 0x%04X, calculated: 0x%04X)",
                 stored_checksum, calculated_checksum);
        free(buffer);
        return ESP_ERR_INVALID_SIZE;
    }

    // Validate counts
    if (header->sequence_length == 0 || header->sequence_length > CAN_NVS_MAX_FRAMES_PER_SEQUENCE) {
        ESP_LOGE(TAG, "Invalid sequence length: %d", header->sequence_length);
        free(buffer);
        return ESP_ERR_INVALID_SIZE;
    }

    uint8_t *ptr = buffer + sizeof(optimized_header_t);

    // Read ID dictionary
    uint32_t *id_dict = (uint32_t *)ptr;
    ptr += header->unique_id_count * sizeof(uint32_t);

    // Read unique frames
    optimized_frame_t *unique_frames = (optimized_frame_t *)ptr;
    ptr += header->unique_frame_count * sizeof(optimized_frame_t);

    // Read frame indices
    uint8_t *frame_indices = ptr;

    // Allocate output frames array
    sequence->frames = (can_nvs_frame_t *)malloc(header->sequence_length * sizeof(can_nvs_frame_t));
    if (sequence->frames == NULL) {
        ESP_LOGE(TAG, "Failed to allocate frames array");
        free(buffer);
        return ESP_ERR_NO_MEM;
    }

    // Reconstruct frames
    for (uint16_t i = 0; i < header->sequence_length; i++) {
        uint8_t frame_idx = frame_indices[i];

        if (frame_idx >= header->unique_frame_count) {
            ESP_LOGE(TAG, "Invalid frame index: %d", frame_idx);
            free(sequence->frames);
            free(buffer);
            return ESP_ERR_INVALID_SIZE;
        }

        optimized_frame_t *opt_frame = &unique_frames[frame_idx];

        if (opt_frame->id_index >= header->unique_id_count) {
            ESP_LOGE(TAG, "Invalid ID index: %d", opt_frame->id_index);
            free(sequence->frames);
            free(buffer);
            return ESP_ERR_INVALID_SIZE;
        }

        // Reconstruct frame
        sequence->frames[i].identifier = id_dict[opt_frame->id_index];
        sequence->frames[i].data_length_code = opt_frame->data_length_code;
        sequence->frames[i].flags = opt_frame->flags;
        memcpy(sequence->frames[i].data, opt_frame->data, CAN_NVS_MAX_DATA_LEN);
    }

    sequence->count = header->sequence_length;

    free(buffer);
    ESP_LOGI(TAG, "Loaded %d CAN frames with key '%s' (%d unique IDs, %d unique frames)",
             sequence->count, key, header->unique_id_count, header->unique_frame_count);
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
        ret = ESP_OK;
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

    size_t required_size = 0;
    esp_err_t ret = nvs_get_blob(nvs_handle, key, NULL, &required_size);
    if (ret != ESP_OK) {
        return ret;
    }

    if (required_size < sizeof(optimized_header_t)) {
        return ESP_ERR_INVALID_SIZE;
    }

    optimized_header_t header;
    uint8_t *buffer = (uint8_t *)malloc(required_size);
    if (buffer == NULL) {
        return ESP_ERR_NO_MEM;
    }

    ret = nvs_get_blob(nvs_handle, key, buffer, &required_size);
    if (ret == ESP_OK) {
        memcpy(&header, buffer, sizeof(header));
        *frame_count = header.sequence_length;
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
