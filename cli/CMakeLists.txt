cmake_minimum_required(VERSION 3.10)
project(maf_cli C)

set(CMAKE_C_STANDARD 99)

# Compilation flags for Release build
if(MSVC)
    add_compile_options(/O2 /fp:fast)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -ffast-math -Wall -Wextra)
endif()

# Include directories
include_directories(../components/esp32_cde/include)

# Source files
set(SOURCES
    main.c
    ../components/esp32_cde/src/maf.c
)

add_executable(maf_cli ${SOURCES})

# Link math library on Linux/Unix
if(UNIX OR MINGW)
    target_link_libraries(maf_cli m)
endif()
