idf_component_register(
    SRCS "test_can_nvs.c"
    INCLUDE_DIRS "."
    REQUIRES unity can_nvs nvs_flash
)
