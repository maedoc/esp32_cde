# CAN NVS Storage Component

A reusable ESP-IDF component for efficiently storing CAN frame sequences in Non-Volatile Storage (NVS).

## Features

- **Compact Binary Serialization**: Efficient storage format minimizes NVS space usage
- **CAN 2.0 Support**: Handles both standard (11-bit) and extended (29-bit) CAN identifiers
- **Frame Flags**: Support for RTR, extended frames, and custom flags
- **Sequence Management**: Store, load, and delete sequences of up to 256 CAN frames
- **Data Integrity**: Built-in checksum validation for stored data
- **Memory Efficient**: Dynamic memory allocation only when loading sequences
- **Well Tested**: Comprehensive unit test suite included
- **Reusable**: Designed as a standalone component for easy integration

## Installation

### As a Component in Your ESP-IDF Project

1. Copy the `can_nvs` directory to your project's `components/` folder:
   ```bash
   cp -r components/can_nvs /path/to/your/project/components/
   ```

2. The component will be automatically discovered by ESP-IDF's build system.

3. Include the header in your code:
   ```c
   #include "can_nvs.h"
   ```

### As a Git Submodule

```bash
cd /path/to/your/project/components
git submodule add https://github.com/yourusername/can_nvs.git
```

## Quick Start

### Basic Usage

```c
#include "can_nvs.h"

void app_main(void) {
    // Initialize the component
    esp_err_t ret = can_nvs_init(NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize CAN NVS");
        return;
    }

    // Create some CAN frames
    can_nvs_frame_t frames[3] = {
        {
            .identifier = 0x123,
            .data_length_code = 8,
            .flags = CAN_NVS_FLAG_NONE,
            .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
        },
        {
            .identifier = 0x456,
            .data_length_code = 4,
            .flags = CAN_NVS_FLAG_EXTENDED,
            .data = {0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x00, 0x00, 0x00}
        },
        {
            .identifier = 0x789,
            .data_length_code = 0,
            .flags = CAN_NVS_FLAG_RTR,
            .data = {0}
        }
    };

    // Create a sequence
    can_nvs_sequence_t sequence = {
        .frames = frames,
        .count = 3
    };

    // Store the sequence
    ret = can_nvs_store_sequence("my_can_data", &sequence);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Sequence stored successfully");
    }

    // Load the sequence later
    can_nvs_sequence_t loaded_seq = {0};
    ret = can_nvs_load_sequence("my_can_data", &loaded_seq);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Loaded %d frames", loaded_seq.count);

        // Process the frames
        for (int i = 0; i < loaded_seq.count; i++) {
            ESP_LOGI(TAG, "Frame %d: ID=0x%X, DLC=%d",
                     i, loaded_seq.frames[i].identifier,
                     loaded_seq.frames[i].data_length_code);
        }

        // Free the loaded sequence
        can_nvs_free_sequence(&loaded_seq);
    }

    // Clean up
    can_nvs_deinit();
}
```

## API Reference

### Initialization

#### `can_nvs_init()`
```c
esp_err_t can_nvs_init(const char *partition_name);
```
Initialize the component. Call this before any other functions.
- **Parameters**:
  - `partition_name`: NVS partition name (use `NULL` for default "nvs")
- **Returns**: `ESP_OK` on success, error code otherwise

#### `can_nvs_deinit()`
```c
esp_err_t can_nvs_deinit(void);
```
Deinitialize and free resources.

### Storage Operations

#### `can_nvs_store_sequence()`
```c
esp_err_t can_nvs_store_sequence(const char *key, const can_nvs_sequence_t *sequence);
```
Store a CAN frame sequence in NVS.
- **Parameters**:
  - `key`: Storage key (max 15 characters)
  - `sequence`: Pointer to sequence to store
- **Returns**: `ESP_OK` on success

#### `can_nvs_load_sequence()`
```c
esp_err_t can_nvs_load_sequence(const char *key, can_nvs_sequence_t *sequence);
```
Load a CAN frame sequence from NVS.
- **Parameters**:
  - `key`: Storage key
  - `sequence`: Pointer to sequence structure (frames will be allocated)
- **Returns**: `ESP_OK` on success
- **Note**: Call `can_nvs_free_sequence()` after use to prevent memory leaks

#### `can_nvs_delete_sequence()`
```c
esp_err_t can_nvs_delete_sequence(const char *key);
```
Delete a stored sequence.

#### `can_nvs_free_sequence()`
```c
void can_nvs_free_sequence(can_nvs_sequence_t *sequence);
```
Free memory allocated for a loaded sequence.

### Utility Functions

#### `can_nvs_key_exists()`
```c
esp_err_t can_nvs_key_exists(const char *key, bool *exists);
```
Check if a key exists in storage.

#### `can_nvs_get_sequence_size()`
```c
esp_err_t can_nvs_get_sequence_size(const char *key, uint16_t *frame_count);
```
Get the number of frames in a stored sequence without loading it.

#### `can_nvs_erase_all()`
```c
esp_err_t can_nvs_erase_all(void);
```
Erase all CAN sequences from the NVS namespace.

## Data Structures

### `can_nvs_frame_t`
```c
typedef struct {
    uint32_t identifier;              // CAN ID (11 or 29 bits)
    uint8_t data_length_code;         // Number of data bytes (0-8)
    uint8_t data[8];                  // Frame data
    uint8_t flags;                    // Frame flags
} can_nvs_frame_t;
```

### `can_nvs_sequence_t`
```c
typedef struct {
    can_nvs_frame_t *frames;  // Array of frames
    uint16_t count;           // Number of frames
} can_nvs_sequence_t;
```

### Frame Flags
```c
typedef enum {
    CAN_NVS_FLAG_NONE = 0x00,
    CAN_NVS_FLAG_EXTENDED = 0x01,        // 29-bit extended ID
    CAN_NVS_FLAG_RTR = 0x02,             // Remote Transmission Request
    CAN_NVS_FLAG_DLC_NON_COMPLIANT = 0x04  // DLC > 8 (CAN-FD)
} can_nvs_flags_t;
```

## Storage Format

The component uses a compact binary format:

```
[Header: 4 bytes]
  - Frame Count: 2 bytes
  - Checksum: 2 bytes

[Frame 0: 14 bytes]
  - Identifier: 4 bytes
  - DLC: 1 byte
  - Flags: 1 byte
  - Data: 8 bytes

[Frame 1: 14 bytes]
  ...

[Frame N: 14 bytes]
```

**Storage Efficiency**: Each frame uses exactly 14 bytes, plus a 4-byte header per sequence.

**Examples**:
- 1 frame: 18 bytes
- 10 frames: 144 bytes
- 100 frames: 1,404 bytes
- 256 frames (max): 3,588 bytes

## Memory Usage

- **Heap Usage**: Only when loading sequences
  - Header parsing: ~100 bytes temporary
  - Loaded sequence: `count × 16 bytes` (slightly more than storage due to alignment)
- **Stack Usage**: Minimal (~200 bytes)
- **Static Memory**: ~8 bytes (NVS handle)

## Configuration Limits

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CAN_NVS_MAX_DATA_LEN` | 8 | Maximum CAN frame data length |
| `CAN_NVS_MAX_FRAMES_PER_SEQUENCE` | 256 | Maximum frames per sequence |
| `CAN_NVS_MAX_KEY_LEN` | 15 | Maximum NVS key length |

## Testing

### Running Unit Tests

The component includes comprehensive unit tests using ESP-IDF's Unity framework.

1. Navigate to the test directory:
   ```bash
   cd components/can_nvs/test
   ```

2. Build the tests:
   ```bash
   idf.py build
   ```

3. Flash to hardware:
   ```bash
   idf.py flash monitor
   ```

4. Or run in QEMU (if available):
   ```bash
   idf.py qemu monitor
   ```

### Test Coverage

The test suite includes:
- ✅ Single frame storage and retrieval
- ✅ Multiple frame sequences
- ✅ Extended CAN IDs (29-bit)
- ✅ RTR frames
- ✅ Sequence deletion
- ✅ Key existence checks
- ✅ Sequence size queries
- ✅ Overwrite operations
- ✅ Invalid argument handling
- ✅ Non-existent key handling
- ✅ Erase all functionality
- ✅ Zero-length data frames
- ✅ Checksum validation

### CI/CD

GitHub Actions workflow (`.github/workflows/can_nvs_test.yml`) automatically:
- Builds the component and tests
- Runs tests in QEMU
- Validates component structure
- Checks code quality

## Integration Examples

### With ESP-IDF CAN Driver

```c
#include "driver/twai.h"
#include "can_nvs.h"

void record_can_traffic(void) {
    can_nvs_frame_t recorded_frames[100];
    uint16_t count = 0;

    // Initialize TWAI (CAN) driver
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(
        GPIO_NUM_21, GPIO_NUM_22, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    twai_driver_install(&g_config, &t_config, &f_config);
    twai_start();

    // Record CAN frames
    twai_message_t rx_msg;
    while (count < 100 && twai_receive(&rx_msg, pdMS_TO_TICKS(1000)) == ESP_OK) {
        recorded_frames[count].identifier = rx_msg.identifier;
        recorded_frames[count].data_length_code = rx_msg.data_length_code;
        recorded_frames[count].flags =
            (rx_msg.extd ? CAN_NVS_FLAG_EXTENDED : 0) |
            (rx_msg.rtr ? CAN_NVS_FLAG_RTR : 0);
        memcpy(recorded_frames[count].data, rx_msg.data, 8);
        count++;
    }

    // Store to NVS
    can_nvs_sequence_t seq = {
        .frames = recorded_frames,
        .count = count
    };
    can_nvs_store_sequence("traffic_log", &seq);

    ESP_LOGI(TAG, "Recorded %d CAN frames", count);
}
```

### Replay Stored Frames

```c
void replay_can_traffic(void) {
    can_nvs_sequence_t seq = {0};

    if (can_nvs_load_sequence("traffic_log", &seq) == ESP_OK) {
        for (int i = 0; i < seq.count; i++) {
            twai_message_t tx_msg = {
                .identifier = seq.frames[i].identifier,
                .data_length_code = seq.frames[i].data_length_code,
                .extd = (seq.frames[i].flags & CAN_NVS_FLAG_EXTENDED) != 0,
                .rtr = (seq.frames[i].flags & CAN_NVS_FLAG_RTR) != 0
            };
            memcpy(tx_msg.data, seq.frames[i].data, 8);

            twai_transmit(&tx_msg, pdMS_TO_TICKS(1000));
            vTaskDelay(pdMS_TO_TICKS(10));  // 10ms delay between frames
        }

        can_nvs_free_sequence(&seq);
    }
}
```

## Error Handling

All API functions return `esp_err_t` error codes:

| Error Code | Description |
|------------|-------------|
| `ESP_OK` | Operation successful |
| `ESP_ERR_INVALID_ARG` | Invalid argument (NULL pointer, invalid count, etc.) |
| `ESP_ERR_NVS_NOT_FOUND` | Key not found in NVS |
| `ESP_ERR_NVS_NOT_ENOUGH_SPACE` | NVS is full |
| `ESP_ERR_NO_MEM` | Memory allocation failed |
| `ESP_ERR_INVALID_SIZE` | Stored data is corrupted |
| `ESP_ERR_NVS_INVALID_HANDLE` | NVS not initialized |

## Best Practices

1. **Always initialize**: Call `can_nvs_init()` before using any other functions
2. **Free memory**: Always call `can_nvs_free_sequence()` after loading
3. **Check return values**: Always check `esp_err_t` return codes
4. **Key naming**: Use descriptive keys (max 15 characters)
5. **Sequence size**: Keep sequences under 100 frames for best performance
6. **Error recovery**: Handle `ESP_ERR_NVS_NOT_ENOUGH_SPACE` by erasing old data

## Troubleshooting

### "NVS not initialized" error
- Ensure you call `can_nvs_init()` before other functions
- Check that NVS partition is defined in your partition table

### "Not enough space" error
- Reduce the number of frames per sequence
- Delete old sequences with `can_nvs_delete_sequence()`
- Use `can_nvs_erase_all()` to clear all data (warning: destructive)
- Increase NVS partition size in `partitions.csv`

### Memory allocation failures
- Check available heap with `esp_get_free_heap_size()`
- Reduce sequence sizes or number of concurrent loaded sequences
- Ensure you're calling `can_nvs_free_sequence()` to prevent leaks

## License

This component is open source. Please check the repository for license details.

## Contributing

Contributions are welcome! Please:
1. Follow ESP-IDF coding standards
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## See Also

- [ESP-IDF NVS Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/storage/nvs_flash.html)
- [ESP-IDF TWAI (CAN) Driver](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/twai.html)
- [CAN Bus Protocol](https://en.wikipedia.org/wiki/CAN_bus)
