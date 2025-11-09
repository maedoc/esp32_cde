/**
 * @file test_can_nvs.c
 * @brief Unit tests for CAN NVS storage component
 */

#include <string.h>
#include "unity.h"
#include "can_nvs.h"
#include "nvs_flash.h"

static const char *TEST_PARTITION = "nvs";

void setUp(void) {
    // Initialize NVS flash
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // Initialize CAN NVS
    ret = can_nvs_init(NULL);
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // Clean up any existing test data
    can_nvs_erase_all();
}

void tearDown(void) {
    can_nvs_deinit();
}

/**
 * Test initialization and deinitialization
 */
void test_can_nvs_init_deinit(void) {
    // Already initialized in setUp
    esp_err_t ret = can_nvs_deinit();
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // Re-initialize
    ret = can_nvs_init(NULL);
    TEST_ASSERT_EQUAL(ESP_OK, ret);
}

/**
 * Test storing and loading a single CAN frame
 */
void test_can_nvs_store_load_single_frame(void) {
    // Create a test frame
    can_nvs_frame_t test_frame = {
        .identifier = 0x123,
        .data_length_code = 8,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
    };

    // Create sequence with single frame
    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    // Store the sequence
    esp_err_t ret = can_nvs_store_sequence("test_key", &store_seq);
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // Load the sequence
    can_nvs_sequence_t load_seq = {0};
    ret = can_nvs_load_sequence("test_key", &load_seq);
    TEST_ASSERT_EQUAL(ESP_OK, ret);
    TEST_ASSERT_EQUAL(1, load_seq.count);
    TEST_ASSERT_NOT_NULL(load_seq.frames);

    // Verify the loaded frame
    TEST_ASSERT_EQUAL(test_frame.identifier, load_seq.frames[0].identifier);
    TEST_ASSERT_EQUAL(test_frame.data_length_code, load_seq.frames[0].data_length_code);
    TEST_ASSERT_EQUAL(test_frame.flags, load_seq.frames[0].flags);
    TEST_ASSERT_EQUAL_MEMORY(test_frame.data, load_seq.frames[0].data, CAN_NVS_MAX_DATA_LEN);

    // Clean up
    can_nvs_free_sequence(&load_seq);
}

/**
 * Test storing and loading multiple CAN frames
 */
void test_can_nvs_store_load_multiple_frames(void) {
    const uint16_t num_frames = 10;
    can_nvs_frame_t test_frames[num_frames];

    // Create test frames
    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i].identifier = 0x100 + i;
        test_frames[i].data_length_code = (i % 8) + 1;
        test_frames[i].flags = (i % 2) ? CAN_NVS_FLAG_EXTENDED : CAN_NVS_FLAG_NONE;
        for (uint8_t j = 0; j < CAN_NVS_MAX_DATA_LEN; j++) {
            test_frames[i].data[j] = i * 10 + j;
        }
    }

    // Create sequence
    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    // Store the sequence
    esp_err_t ret = can_nvs_store_sequence("multi_test", &store_seq);
    TEST_ASSERT_EQUAL(ESP_OK, ret);

    // Load the sequence
    can_nvs_sequence_t load_seq = {0};
    ret = can_nvs_load_sequence("multi_test", &load_seq);
    TEST_ASSERT_EQUAL(ESP_OK, ret);
    TEST_ASSERT_EQUAL(num_frames, load_seq.count);
    TEST_ASSERT_NOT_NULL(load_seq.frames);

    // Verify all loaded frames
    for (uint16_t i = 0; i < num_frames; i++) {
        TEST_ASSERT_EQUAL(test_frames[i].identifier, load_seq.frames[i].identifier);
        TEST_ASSERT_EQUAL(test_frames[i].data_length_code, load_seq.frames[i].data_length_code);
        TEST_ASSERT_EQUAL(test_frames[i].flags, load_seq.frames[i].flags);
        TEST_ASSERT_EQUAL_MEMORY(test_frames[i].data, load_seq.frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    // Clean up
    can_nvs_free_sequence(&load_seq);
}

/**
 * Test extended CAN ID (29-bit)
 */
void test_can_nvs_extended_id(void) {
    // Create a frame with extended ID
    can_nvs_frame_t test_frame = {
        .identifier = 0x1FFFFFFF,  // Maximum 29-bit ID
        .data_length_code = 4,
        .flags = CAN_NVS_FLAG_EXTENDED,
        .data = {0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x00, 0x00, 0x00}
    };

    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    // Store and load
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("ext_id", &store_seq));

    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("ext_id", &load_seq));
    TEST_ASSERT_EQUAL(1, load_seq.count);

    // Verify extended ID
    TEST_ASSERT_EQUAL(0x1FFFFFFF, load_seq.frames[0].identifier);
    TEST_ASSERT_EQUAL(CAN_NVS_FLAG_EXTENDED, load_seq.frames[0].flags & CAN_NVS_FLAG_EXTENDED);

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test RTR frame flag
 */
void test_can_nvs_rtr_flag(void) {
    can_nvs_frame_t test_frame = {
        .identifier = 0x456,
        .data_length_code = 0,  // RTR frames typically have no data
        .flags = CAN_NVS_FLAG_RTR,
        .data = {0}
    };

    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("rtr_test", &store_seq));

    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("rtr_test", &load_seq));

    TEST_ASSERT_EQUAL(CAN_NVS_FLAG_RTR, load_seq.frames[0].flags & CAN_NVS_FLAG_RTR);

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test deleting a sequence
 */
void test_can_nvs_delete_sequence(void) {
    can_nvs_frame_t test_frame = {
        .identifier = 0x789,
        .data_length_code = 2,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x12, 0x34, 0, 0, 0, 0, 0, 0}
    };

    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    // Store
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("delete_test", &store_seq));

    // Verify it exists
    bool exists = false;
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("delete_test", &exists));
    TEST_ASSERT_TRUE(exists);

    // Delete
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_delete_sequence("delete_test"));

    // Verify it's gone
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("delete_test", &exists));
    TEST_ASSERT_FALSE(exists);

    // Try to load deleted sequence
    can_nvs_sequence_t load_seq = {0};
    esp_err_t ret = can_nvs_load_sequence("delete_test", &load_seq);
    TEST_ASSERT_EQUAL(ESP_ERR_NVS_NOT_FOUND, ret);
}

/**
 * Test key existence check
 */
void test_can_nvs_key_exists(void) {
    bool exists = false;

    // Check non-existent key
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("nonexistent", &exists));
    TEST_ASSERT_FALSE(exists);

    // Store a sequence
    can_nvs_frame_t test_frame = {
        .identifier = 0xABC,
        .data_length_code = 1,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0xFF, 0, 0, 0, 0, 0, 0, 0}
    };

    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("exists_test", &store_seq));

    // Check existing key
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("exists_test", &exists));
    TEST_ASSERT_TRUE(exists);
}

/**
 * Test getting sequence size
 */
void test_can_nvs_get_sequence_size(void) {
    const uint16_t num_frames = 25;
    can_nvs_frame_t test_frames[num_frames];

    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i].identifier = i;
        test_frames[i].data_length_code = 8;
        test_frames[i].flags = CAN_NVS_FLAG_NONE;
        memset(test_frames[i].data, i, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("size_test", &store_seq));

    // Get size without loading
    uint16_t frame_count = 0;
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_get_sequence_size("size_test", &frame_count));
    TEST_ASSERT_EQUAL(num_frames, frame_count);
}

/**
 * Test overwriting existing sequence
 */
void test_can_nvs_overwrite_sequence(void) {
    // First sequence
    can_nvs_frame_t frame1 = {
        .identifier = 0x111,
        .data_length_code = 3,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {1, 2, 3, 0, 0, 0, 0, 0}
    };

    can_nvs_sequence_t seq1 = {
        .frames = &frame1,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("overwrite", &seq1));

    // Second sequence (different data)
    can_nvs_frame_t frame2 = {
        .identifier = 0x222,
        .data_length_code = 5,
        .flags = CAN_NVS_FLAG_EXTENDED,
        .data = {5, 6, 7, 8, 9, 0, 0, 0}
    };

    can_nvs_sequence_t seq2 = {
        .frames = &frame2,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("overwrite", &seq2));

    // Load and verify it's the second sequence
    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("overwrite", &load_seq));
    TEST_ASSERT_EQUAL(0x222, load_seq.frames[0].identifier);
    TEST_ASSERT_EQUAL(5, load_seq.frames[0].data_length_code);
    TEST_ASSERT_EQUAL(CAN_NVS_FLAG_EXTENDED, load_seq.frames[0].flags);

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test invalid arguments
 */
void test_can_nvs_invalid_args(void) {
    can_nvs_frame_t test_frame = {0};
    can_nvs_sequence_t test_seq = {
        .frames = &test_frame,
        .count = 1
    };

    // NULL key
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, can_nvs_store_sequence(NULL, &test_seq));

    // NULL sequence
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, can_nvs_store_sequence("test", NULL));

    // Invalid frame count (0)
    test_seq.count = 0;
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, can_nvs_store_sequence("test", &test_seq));

    // Key too long
    char long_key[20] = "this_key_is_too_long";
    test_seq.count = 1;
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, can_nvs_store_sequence(long_key, &test_seq));
}

/**
 * Test loading non-existent sequence
 */
void test_can_nvs_load_nonexistent(void) {
    can_nvs_sequence_t load_seq = {0};
    esp_err_t ret = can_nvs_load_sequence("does_not_exist", &load_seq);
    TEST_ASSERT_EQUAL(ESP_ERR_NVS_NOT_FOUND, ret);
}

/**
 * Test erase all
 */
void test_can_nvs_erase_all(void) {
    // Store multiple sequences
    can_nvs_frame_t test_frame = {
        .identifier = 0x100,
        .data_length_code = 1,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x42, 0, 0, 0, 0, 0, 0, 0}
    };

    can_nvs_sequence_t test_seq = {
        .frames = &test_frame,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("key1", &test_seq));
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("key2", &test_seq));
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("key3", &test_seq));

    // Erase all
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_erase_all());

    // Verify all are gone
    bool exists = false;
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("key1", &exists));
    TEST_ASSERT_FALSE(exists);
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("key2", &exists));
    TEST_ASSERT_FALSE(exists);
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_key_exists("key3", &exists));
    TEST_ASSERT_FALSE(exists);
}

/**
 * Test zero-length data frames
 */
void test_can_nvs_zero_length_data(void) {
    can_nvs_frame_t test_frame = {
        .identifier = 0x200,
        .data_length_code = 0,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0}
    };

    can_nvs_sequence_t store_seq = {
        .frames = &test_frame,
        .count = 1
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("zero_len", &store_seq));

    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("zero_len", &load_seq));
    TEST_ASSERT_EQUAL(0, load_seq.frames[0].data_length_code);

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test compression with many repeated frames (best case)
 */
void test_can_nvs_compression_best_case(void) {
    const uint16_t num_frames = 256;
    can_nvs_frame_t test_frames[num_frames];

    // All frames are identical - best compression
    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i].identifier = 0x100;
        test_frames[i].data_length_code = 8;
        test_frames[i].flags = CAN_NVS_FLAG_NONE;
        memset(test_frames[i].data, 0xAA, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("compress_best", &store_seq));

    // Load and verify
    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("compress_best", &load_seq));
    TEST_ASSERT_EQUAL(num_frames, load_seq.count);

    // Verify all frames match
    for (uint16_t i = 0; i < num_frames; i++) {
        TEST_ASSERT_EQUAL(test_frames[i].identifier, load_seq.frames[i].identifier);
        TEST_ASSERT_EQUAL_MEMORY(test_frames[i].data, load_seq.frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test compression with typical CAN traffic (few unique IDs, some repeats)
 */
void test_can_nvs_compression_typical_case(void) {
    const uint16_t num_frames = 200;
    const uint8_t num_unique_ids = 10;
    can_nvs_frame_t test_frames[num_frames];

    // Simulate typical CAN traffic: 10 different IDs cycling, with some data variation
    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i].identifier = 0x100 + (i % num_unique_ids);
        test_frames[i].data_length_code = 8;
        test_frames[i].flags = (i % 3 == 0) ? CAN_NVS_FLAG_EXTENDED : CAN_NVS_FLAG_NONE;

        // Data changes slowly to create some repeated frames
        uint8_t pattern = (i / 5) & 0xFF;
        for (uint8_t j = 0; j < CAN_NVS_MAX_DATA_LEN; j++) {
            test_frames[i].data[j] = pattern + j;
        }
    }

    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("compress_typical", &store_seq));

    // Load and verify
    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("compress_typical", &load_seq));
    TEST_ASSERT_EQUAL(num_frames, load_seq.count);

    // Verify all frames match
    for (uint16_t i = 0; i < num_frames; i++) {
        TEST_ASSERT_EQUAL(test_frames[i].identifier, load_seq.frames[i].identifier);
        TEST_ASSERT_EQUAL(test_frames[i].data_length_code, load_seq.frames[i].data_length_code);
        TEST_ASSERT_EQUAL(test_frames[i].flags, load_seq.frames[i].flags);
        TEST_ASSERT_EQUAL_MEMORY(test_frames[i].data, load_seq.frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test compression with worst case (all unique frames)
 */
void test_can_nvs_compression_worst_case(void) {
    const uint16_t num_frames = 100;
    can_nvs_frame_t test_frames[num_frames];

    // All frames completely unique - worst compression (still better than original due to ID dict)
    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i].identifier = 0x100 + i;
        test_frames[i].data_length_code = 8;
        test_frames[i].flags = (i % 2) ? CAN_NVS_FLAG_EXTENDED : CAN_NVS_FLAG_NONE;

        // Unique data for each frame
        for (uint8_t j = 0; j < CAN_NVS_MAX_DATA_LEN; j++) {
            test_frames[i].data[j] = (i + j) & 0xFF;
        }
    }

    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("compress_worst", &store_seq));

    // Load and verify
    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("compress_worst", &load_seq));
    TEST_ASSERT_EQUAL(num_frames, load_seq.count);

    // Verify all frames match
    for (uint16_t i = 0; i < num_frames; i++) {
        TEST_ASSERT_EQUAL(test_frames[i].identifier, load_seq.frames[i].identifier);
        TEST_ASSERT_EQUAL(test_frames[i].data_length_code, load_seq.frames[i].data_length_code);
        TEST_ASSERT_EQUAL(test_frames[i].flags, load_seq.frames[i].flags);
        TEST_ASSERT_EQUAL_MEMORY(test_frames[i].data, load_seq.frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_free_sequence(&load_seq);
}

/**
 * Test deduplication with alternating frames
 */
void test_can_nvs_deduplication(void) {
    const uint16_t num_frames = 100;
    can_nvs_frame_t test_frames[num_frames];

    // Create pattern: A, B, A, B, A, B... (only 2 unique frames)
    can_nvs_frame_t frame_a = {
        .identifier = 0x111,
        .data_length_code = 4,
        .flags = CAN_NVS_FLAG_NONE,
        .data = {0x11, 0x22, 0x33, 0x44, 0, 0, 0, 0}
    };

    can_nvs_frame_t frame_b = {
        .identifier = 0x222,
        .data_length_code = 4,
        .flags = CAN_NVS_FLAG_EXTENDED,
        .data = {0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0}
    };

    for (uint16_t i = 0; i < num_frames; i++) {
        test_frames[i] = (i % 2 == 0) ? frame_a : frame_b;
    }

    can_nvs_sequence_t store_seq = {
        .frames = test_frames,
        .count = num_frames
    };

    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_store_sequence("dedup_test", &store_seq));

    // Load and verify
    can_nvs_sequence_t load_seq = {0};
    TEST_ASSERT_EQUAL(ESP_OK, can_nvs_load_sequence("dedup_test", &load_seq));
    TEST_ASSERT_EQUAL(num_frames, load_seq.count);

    // Verify alternating pattern preserved
    for (uint16_t i = 0; i < num_frames; i++) {
        can_nvs_frame_t *expected = (i % 2 == 0) ? &frame_a : &frame_b;
        TEST_ASSERT_EQUAL(expected->identifier, load_seq.frames[i].identifier);
        TEST_ASSERT_EQUAL(expected->data_length_code, load_seq.frames[i].data_length_code);
        TEST_ASSERT_EQUAL(expected->flags, load_seq.frames[i].flags);
        TEST_ASSERT_EQUAL_MEMORY(expected->data, load_seq.frames[i].data, CAN_NVS_MAX_DATA_LEN);
    }

    can_nvs_free_sequence(&load_seq);
}

// Test runner
void app_main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_can_nvs_init_deinit);
    RUN_TEST(test_can_nvs_store_load_single_frame);
    RUN_TEST(test_can_nvs_store_load_multiple_frames);
    RUN_TEST(test_can_nvs_extended_id);
    RUN_TEST(test_can_nvs_rtr_flag);
    RUN_TEST(test_can_nvs_delete_sequence);
    RUN_TEST(test_can_nvs_key_exists);
    RUN_TEST(test_can_nvs_get_sequence_size);
    RUN_TEST(test_can_nvs_overwrite_sequence);
    RUN_TEST(test_can_nvs_invalid_args);
    RUN_TEST(test_can_nvs_load_nonexistent);
    RUN_TEST(test_can_nvs_erase_all);
    RUN_TEST(test_can_nvs_zero_length_data);

    // Compression and deduplication tests
    RUN_TEST(test_can_nvs_compression_best_case);
    RUN_TEST(test_can_nvs_compression_typical_case);
    RUN_TEST(test_can_nvs_compression_worst_case);
    RUN_TEST(test_can_nvs_deduplication);

    UNITY_END();
}
