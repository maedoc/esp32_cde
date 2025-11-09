/**
 * @file can_nvs.h
 * @brief CAN Frame NVS Storage Component
 *
 * This component provides functionality to store and retrieve CAN frame sequences
 * in ESP32's Non-Volatile Storage (NVS). It uses compact binary serialization
 * for efficient storage.
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CAN_NVS_H
#define CAN_NVS_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum data length for a CAN frame
 */
#define CAN_NVS_MAX_DATA_LEN 8

/**
 * @brief Maximum number of frames that can be stored in a single sequence
 */
#define CAN_NVS_MAX_FRAMES_PER_SEQUENCE 256

/**
 * @brief Maximum key length for NVS storage
 */
#define CAN_NVS_MAX_KEY_LEN 15

/**
 * @brief CAN frame flags
 */
typedef enum {
    CAN_NVS_FLAG_NONE = 0x00,       /**< No flags */
    CAN_NVS_FLAG_EXTENDED = 0x01,   /**< Extended 29-bit CAN ID */
    CAN_NVS_FLAG_RTR = 0x02,        /**< Remote Transmission Request */
    CAN_NVS_FLAG_DLC_NON_COMPLIANT = 0x04  /**< DLC > 8 for CAN-FD */
} can_nvs_flags_t;

/**
 * @brief CAN frame structure
 */
typedef struct {
    uint32_t identifier;                    /**< CAN identifier (11-bit or 29-bit) */
    uint8_t data_length_code;               /**< Number of data bytes (0-8) */
    uint8_t data[CAN_NVS_MAX_DATA_LEN];     /**< Frame data payload */
    uint8_t flags;                          /**< Frame flags (see can_nvs_flags_t) */
} can_nvs_frame_t;

/**
 * @brief CAN frame sequence structure
 */
typedef struct {
    can_nvs_frame_t *frames;    /**< Array of CAN frames */
    uint16_t count;             /**< Number of frames in the sequence */
} can_nvs_sequence_t;

/**
 * @brief Initialize the CAN NVS storage component
 *
 * This function must be called before any other CAN NVS functions.
 * It initializes the NVS partition and prepares it for use.
 *
 * @param[in] partition_name NVS partition name (NULL for default "nvs")
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_NVS_NOT_INITIALIZED if NVS is not initialized
 *      - ESP_ERR_NO_MEM if memory allocation fails
 */
esp_err_t can_nvs_init(const char *partition_name);

/**
 * @brief Deinitialize the CAN NVS storage component
 *
 * Closes the NVS handle and frees resources.
 *
 * @return
 *      - ESP_OK on success
 */
esp_err_t can_nvs_deinit(void);

/**
 * @brief Store a CAN frame sequence in NVS
 *
 * Serializes and stores a sequence of CAN frames in NVS using a compact
 * binary format. If a sequence with the same key already exists, it will
 * be overwritten.
 *
 * @param[in] key NVS key (max 15 characters)
 * @param[in] sequence Pointer to the CAN frame sequence
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_INVALID_ARG if key or sequence is NULL, or if parameters are invalid
 *      - ESP_ERR_NVS_NOT_ENOUGH_SPACE if NVS is full
 *      - ESP_ERR_NVS_INVALID_HANDLE if NVS is not initialized
 */
esp_err_t can_nvs_store_sequence(const char *key, const can_nvs_sequence_t *sequence);

/**
 * @brief Load a CAN frame sequence from NVS
 *
 * Deserializes and loads a sequence of CAN frames from NVS.
 * The caller is responsible for freeing the frames array after use.
 *
 * @param[in] key NVS key (max 15 characters)
 * @param[out] sequence Pointer to sequence structure (frames will be allocated)
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_INVALID_ARG if key or sequence is NULL
 *      - ESP_ERR_NVS_NOT_FOUND if key does not exist
 *      - ESP_ERR_NO_MEM if memory allocation fails
 *      - ESP_ERR_INVALID_SIZE if stored data is corrupted
 */
esp_err_t can_nvs_load_sequence(const char *key, can_nvs_sequence_t *sequence);

/**
 * @brief Delete a CAN frame sequence from NVS
 *
 * Removes the sequence associated with the given key from NVS.
 *
 * @param[in] key NVS key (max 15 characters)
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_INVALID_ARG if key is NULL
 *      - ESP_ERR_NVS_NOT_FOUND if key does not exist
 */
esp_err_t can_nvs_delete_sequence(const char *key);

/**
 * @brief Free a CAN frame sequence
 *
 * Frees the memory allocated for the frames array in a sequence.
 * This should be called after loading a sequence to prevent memory leaks.
 *
 * @param[in] sequence Pointer to the sequence to free
 */
void can_nvs_free_sequence(can_nvs_sequence_t *sequence);

/**
 * @brief Check if a key exists in NVS
 *
 * @param[in] key NVS key to check
 * @param[out] exists Pointer to boolean that will be set to true if key exists
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_INVALID_ARG if key or exists is NULL
 */
esp_err_t can_nvs_key_exists(const char *key, bool *exists);

/**
 * @brief Get the size of a stored sequence
 *
 * Returns the number of frames in a stored sequence without loading it.
 *
 * @param[in] key NVS key
 * @param[out] frame_count Pointer to store the frame count
 * @return
 *      - ESP_OK on success
 *      - ESP_ERR_INVALID_ARG if key or frame_count is NULL
 *      - ESP_ERR_NVS_NOT_FOUND if key does not exist
 */
esp_err_t can_nvs_get_sequence_size(const char *key, uint16_t *frame_count);

/**
 * @brief Erase all CAN sequences from NVS
 *
 * WARNING: This will erase the entire NVS namespace used by this component.
 *
 * @return
 *      - ESP_OK on success
 *      - ESP_FAIL on error
 */
esp_err_t can_nvs_erase_all(void);

#ifdef __cplusplus
}
#endif

#endif // CAN_NVS_H
