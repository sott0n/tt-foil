# tt_metal_deps.cmake
# Pulls in only the necessary components from the tt-metal submodule.
# Does NOT trigger a full tt-metal build.

set(TT_METAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-metal")

if(NOT EXISTS "${TT_METAL_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "tt-metal submodule not found at ${TT_METAL_ROOT}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ---- UMD (User Mode Driver) ----
# PCIe access, TLB management, core resets. Cannot be avoided.
set(UMD_ROOT "${TT_METAL_ROOT}/tt_metal/third_party/umd")
add_subdirectory("${UMD_ROOT}" "${CMAKE_BINARY_DIR}/umd_build" EXCLUDE_FROM_ALL)

# ---- Blackhole HAL ----
# Memory map constants, RISC firmware addresses, core type info.
set(HAL_ROOT "${TT_METAL_ROOT}/tt_metal/llrt/hal")
add_subdirectory("${HAL_ROOT}" "${CMAKE_BINARY_DIR}/hal_build" EXCLUDE_FROM_ALL)

# ---- LLRT subset ----
# Only the files needed: ELF loader, memory abstraction, cluster wrapper.
# We intentionally skip the rest of llrt (profiler, watcher, dprint, etc.)
add_library(tt_foil_llrt STATIC
    "${TT_METAL_ROOT}/tt_metal/llrt/tt_elffile.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/tt_memory.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/tt_cluster.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/rtoptions.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/tlb_config.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/metal_soc_descriptor.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/core_descriptor.cpp"
)

target_include_directories(tt_foil_llrt PUBLIC
    "${TT_METAL_ROOT}"
    "${TT_METAL_ROOT}/tt_metal"
    "${TT_METAL_ROOT}/tt_metal/llrt"
    "${TT_METAL_ROOT}/tt_metal/hw/inc"
    "${TT_METAL_ROOT}/tt_metal/hw/inc/hostdev"
    "${TT_METAL_ROOT}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    "${TT_METAL_ROOT}/tt_metal/hostdevcommon"
    "${TT_METAL_ROOT}/tt_metal/common"
    "${TT_METAL_ROOT}/tt_metal/api"
    "${TT_METAL_ROOT}/tt_metal/api/tt-metalium"
)

target_compile_definitions(tt_foil_llrt PUBLIC
    ARCH_BLACKHOLE
)

target_link_libraries(tt_foil_llrt
    PUBLIC
        umd::tt-umd
    PRIVATE
        HAL::1xx
        fmt::fmt-header-only
        nlohmann_json::nlohmann_json
        yaml-cpp
)
