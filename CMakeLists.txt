cmake_minimum_required(VERSION 3.21)
project(tt_foil VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TT_FOIL_BUILD_TESTS     "Build tests"                              ON)
option(TT_FOIL_BUILD_EXAMPLES  "Build examples"                           ON)
option(TT_FOIL_BUILD_FIRMWARE  "Self-build RISC firmware ELFs (Plan L)"   ON)

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
    src/noc_addr.cpp
    src/cb_config.cpp
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

# ---- tt_foil_weaken host CLI ----
# Used by scripts/build_firmware.sh to produce *_weakened.elf companions to
# the firmware ELFs.  Mirrors tt-metal's ll_api::ElfFile::WeakenDataSymbols
# without dragging in any device-side dependency.
add_executable(tt_foil_weaken tools/tt_foil_weaken.cpp)
target_include_directories(tt_foil_weaken PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llrt_local"
)
target_link_libraries(tt_foil_weaken PRIVATE tt_foil)
set_target_properties(tt_foil_weaken PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
)

# ---- Plan L: self-build firmware ELFs (no tt-metal runtime needed) ----
# Builds the 5 firmware ELFs + *_weakened.elf companions directly from the
# tt-metal source tree under ${CMAKE_BINARY_DIR}/firmware/.  tt-foil's
# firmware_paths resolver prefers this tree over the JIT cache, removing the
# requirement that tt-metal has been executed at least once on this host.
if(TT_FOIL_BUILD_FIRMWARE)
    target_compile_definitions(tt_foil PRIVATE
        TT_FOIL_FIRMWARE_BUILD_DIR="${CMAKE_BINARY_DIR}/firmware"
    )
    set(_tt_foil_fw_outputs
        "${CMAKE_BINARY_DIR}/firmware/brisc/brisc.elf"
        "${CMAKE_BINARY_DIR}/firmware/brisc/brisc_weakened.elf"
        "${CMAKE_BINARY_DIR}/firmware/ncrisc/ncrisc.elf"
        "${CMAKE_BINARY_DIR}/firmware/ncrisc/ncrisc_weakened.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc0/trisc0.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc0/trisc0_weakened.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc1/trisc1.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc1/trisc1_weakened.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc2/trisc2.elf"
        "${CMAKE_BINARY_DIR}/firmware/trisc2/trisc2_weakened.elf"
    )
    add_custom_command(
        OUTPUT ${_tt_foil_fw_outputs}
        COMMAND ${CMAKE_COMMAND} -E env
            TT_METAL_ROOT=${TT_METAL_ROOT}
            TT_FOIL_BUILD=${CMAKE_BINARY_DIR}
            OUT_DIR=${CMAKE_BINARY_DIR}/firmware
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_firmware.sh
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_firmware.sh
            tt_foil_weaken
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Building tt-foil firmware ELFs from tt-metal source"
        VERBATIM
    )
    add_custom_target(tt_foil_firmware ALL DEPENDS ${_tt_foil_fw_outputs})
endif()
