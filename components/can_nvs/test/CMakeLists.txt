# Test application for CAN NVS component
cmake_minimum_required(VERSION 3.16)

# Set the components to include
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/..")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(can_nvs_test)
