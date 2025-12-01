cmake_minimum_required(VERSION 3.10)
project(maf_cli C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Compilation flags for Release build
if(MSVC)
    add_compile_options(/O2 /fp:fast)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES)
    # Suppress conversion warnings for now to keep build log clean
    add_compile_options(/wd4244 /wd4018)
else()
    add_compile_options(-O3 -ffast-math -march=native -mavx2 -Wall -Wextra)
endif()

# Include directories
include_directories(../components/esp32_cde/include)

# Source files
set(SOURCES
    main.c
    ../components/esp32_cde/src/maf.c
    ../components/esp32_cde/src/maf_batched.cpp
)

add_executable(maf_cli ${SOURCES})

# Link math library on Linux/Unix
if(UNIX OR MINGW)
    target_link_libraries(maf_cli m)
endif()

# Test Batched Implementation
add_executable(maf_test_batched
    test_batched.cpp
    ../components/esp32_cde/src/maf.c
    ../components/esp32_cde/src/maf_batched.cpp
)
if(UNIX OR MINGW)
    target_link_libraries(maf_test_batched m)
endif()

# Benchmark Training
add_executable(maf_benchmark
    benchmark_training.cpp
    ../components/esp32_cde/src/maf.c
    ../components/esp32_cde/src/maf_batched.cpp
)
if(UNIX OR MINGW)
    target_link_libraries(maf_benchmark m)
endif()

# Lorenz Stats Test
add_executable(maf_test_lorenz
    test_lorenz_stats.cpp
    ../components/esp32_cde/src/maf.c
    ../components/esp32_cde/src/maf_batched.cpp
)
if(UNIX OR MINGW)
    target_link_libraries(maf_test_lorenz m)
endif()
