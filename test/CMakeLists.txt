cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS 
    "../components"
)

# Define components to test
set(TEST_COMPONENTS "esp32_cde")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_cde_test)
