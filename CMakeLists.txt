cmake_minimum_required(VERSION 3.21)
project(tt_foil VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TT_FOIL_BUILD_TESTS    "Build tests"    ON)
option(TT_FOIL_BUILD_EXAMPLES "Build examples" ON)

# Pull in tt-metal submodule components (UMD, HAL, llrt subset)
include(cmake/tt_metal_deps.cmake)

# ---- tt_foil library ----
add_library(tt_foil STATIC
    src/device.cpp
    src/buffer.cpp
    src/kernel.cpp
    src/dispatch.cpp
    src/firmware_paths.cpp
    src/firmware_load.cpp
    src/mailbox_init.cpp
    src/core_info_init.cpp
    src/bank_tables_init.cpp
    src/reset.cpp
    src/umd_boot.cpp
    src/runtime_impl.cpp
    # Bundled tt-metal llrt sources (Phase B3-2 + B3-3): copied from
    # tt_metal/llrt/, namespace renamed ll_api → tt::foil::ll_api so they
    # don't conflict with libtt_metal.so's copies, and tt_memory.cpp's one
    # MetalContext::instance() call patched out (XIP-dump debug path).
    src/llrt_local/tt_memory.cpp
    src/llrt_local/tt_elffile.cpp
)
add_library(TT::Foil ALIAS tt_foil)

target_include_directories(tt_foil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(tt_foil
    PUBLIC
        umd::tt-umd
    PRIVATE
        tt_foil_llrt
        tt_foil_hal_local
)

if(TT_FOIL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TT_FOIL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
