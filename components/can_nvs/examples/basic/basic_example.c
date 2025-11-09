/**
 * @file basic_example.c
 * @brief Basic example demonstrating CAN NVS component usage
 */

#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "can_nvs.h"
#include "nvs_flash.h"

static const char *TAG = "can_nvs_example";

void app_main(void) {
    esp_err_t ret;

    ESP_LOGI(TAG, "=== CAN NVS Basic Example ===");

    // Initialize NVS flash (required for can_nvs)
    ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize CAN NVS component
    ret = can_nvs_init(NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize CAN NVS: %s", esp_err_to_name(ret));
        return;
    }
    ESP_LOGI(TAG, "CAN NVS initialized successfully");

    // Example 1: Store a single CAN frame
    ESP_LOGI(TAG, "\n--- Example 1: Single Frame ---");
    can_nvs_frame_t single_frame = {
        .identifier = 0x123,
        .data_length_code = 8,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
    };

    can_nvs_sequence_t single_seq = {
        .frames = &single_frame,
        .count = 1
    };

    ret = can_nvs_store_sequence("single_frame", &single_seq);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Stored single frame successfully");
    }

    // Example 2: Store multiple CAN frames
    ESP_LOGI(TAG, "\n--- Example 2: Multiple Frames ---");
    can_nvs_frame_t frames[5];

    // Frame 0: Standard ID
    frames[0].identifier = 0x100;
    frames[0].data_length_code = 8;
    frames[0].flags = CAN_NVS_FLAG_NONE;
    for (int i = 0; i < 8; i++) {
        frames[0].data[i] = i;
    }

    // Frame 1: Extended ID
    frames[1].identifier = 0x1FFFFFFF;
    frames[1].data_length_code = 4;
    frames[1].flags = CAN_NVS_FLAG_EXTENDED;
    frames[1].data[0] = 0xAA;
    frames[1].data[1] = 0xBB;
    frames[1].data[2] = 0xCC;
    frames[1].data[3] = 0xDD;

    // Frame 2: RTR (Remote Transmission Request)
    frames[2].identifier = 0x456;
    frames[2].data_length_code = 0;
    frames[2].flags = CAN_NVS_FLAG_RTR;

    // Frame 3: Another standard frame
    frames[3].identifier = 0x789;
    frames[3].data_length_code = 2;
    frames[3].flags = CAN_NVS_FLAG_NONE;
    frames[3].data[0] = 0x12;
    frames[3].data[1] = 0x34;

    // Frame 4: Extended with data
    frames[4].identifier = 0x1234ABCD;
    frames[4].data_length_code = 8;
    frames[4].flags = CAN_NVS_FLAG_EXTENDED;
    memset(frames[4].data, 0xFF, 8);

    can_nvs_sequence_t multi_seq = {
        .frames = frames,
        .count = 5
    };

    ret = can_nvs_store_sequence("multi_frames", &multi_seq);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Stored %d frames successfully", multi_seq.count);
    }

    // Example 3: Load and display frames
    ESP_LOGI(TAG, "\n--- Example 3: Loading Frames ---");

    // Check if key exists
    bool exists = false;
    can_nvs_key_exists("multi_frames", &exists);
    if (exists) {
        ESP_LOGI(TAG, "Key 'multi_frames' exists");

        // Get sequence size
        uint16_t frame_count = 0;
        can_nvs_get_sequence_size("multi_frames", &frame_count);
        ESP_LOGI(TAG, "Sequence contains %d frames", frame_count);

        // Load the sequence
        can_nvs_sequence_t loaded_seq = {0};
        ret = can_nvs_load_sequence("multi_frames", &loaded_seq);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Successfully loaded %d frames", loaded_seq.count);

            // Display each frame
            for (int i = 0; i < loaded_seq.count; i++) {
                ESP_LOGI(TAG, "Frame %d:", i);
                ESP_LOGI(TAG, "  ID: 0x%08X %s",
                         loaded_seq.frames[i].identifier,
                         (loaded_seq.frames[i].flags & CAN_NVS_FLAG_EXTENDED) ? "(Extended)" : "(Standard)");
                ESP_LOGI(TAG, "  DLC: %d", loaded_seq.frames[i].data_length_code);

                if (loaded_seq.frames[i].flags & CAN_NVS_FLAG_RTR) {
                    ESP_LOGI(TAG, "  Type: RTR");
                } else {
                    // Print data bytes
                    char data_str[32] = {0};
                    for (int j = 0; j < loaded_seq.frames[i].data_length_code; j++) {
                        sprintf(data_str + strlen(data_str), "%02X ", loaded_seq.frames[i].data[j]);
                    }
                    ESP_LOGI(TAG, "  Data: %s", data_str);
                }
            }

            // Free the loaded sequence
            can_nvs_free_sequence(&loaded_seq);
        }
    }

    // Example 4: Overwrite existing sequence
    ESP_LOGI(TAG, "\n--- Example 4: Overwrite Sequence ---");
    can_nvs_frame_t new_frame = {
        .identifier = 0x999,
        .data_length_code = 1,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x42}
    };

    can_nvs_sequence_t new_seq = {
        .frames = &new_frame,
        .count = 1
    };

    ret = can_nvs_store_sequence("single_frame", &new_seq);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Overwrote 'single_frame' with new data");
    }

    // Example 5: Delete a sequence
    ESP_LOGI(TAG, "\n--- Example 5: Delete Sequence ---");
    ret = can_nvs_delete_sequence("single_frame");
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Deleted 'single_frame' successfully");

        // Verify it's deleted
        can_nvs_key_exists("single_frame", &exists);
        ESP_LOGI(TAG, "Key exists after delete: %s", exists ? "yes" : "no");
    }

    // Cleanup
    ESP_LOGI(TAG, "\n--- Cleanup ---");
    can_nvs_deinit();
    ESP_LOGI(TAG, "CAN NVS deinitialized");

    ESP_LOGI(TAG, "\n=== Example Complete ===");
}
