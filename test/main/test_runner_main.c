#include <stdio.h>
#include <string.h>
#include "unity.h"
#include "unity_test_runner.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "test_runner";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting Unit Tests...");
    
    // Allow time for QEMU monitor to hook up
    vTaskDelay(pdMS_TO_TICKS(1000));

    unity_run_menu();
}
