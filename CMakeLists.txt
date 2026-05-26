cmake_minimum_required(VERSION 3.21)
project(tt_foil VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TT_FOIL_BUILD_TESTS     "Build tests"                              ON)
option(TT_FOIL_BUILD_EXAMPLES  "Build examples"                           ON)
option(TT_FOIL_BUILD_FIRMWARE  "Self-build RISC firmware ELFs (Plan L)"   ON)
option(TT_FOIL_ENABLE_TRACY
    "Enable Tracy profiling for offline Performance/Memory Report generation" OFF)
# Device-side cycle profiler (firmware + kernel instrumented with
# DeviceZoneScopedN). Implies TT_FOIL_ENABLE_TRACY since the per-RISC
# zones flow through the same report pipeline. See
# docs/device_profiler_plan.md for the design.
option(TT_FOIL_DEVICE_PROFILER
    "Enable device-side cycle profiling — rebuilds firmware + kernels with PROFILE_KERNEL" OFF)
if(TT_FOIL_DEVICE_PROFILER AND NOT TT_FOIL_ENABLE_TRACY)
    message(FATAL_ERROR
        "TT_FOIL_DEVICE_PROFILER requires TT_FOIL_ENABLE_TRACY=ON "
        "(device zones share the host profiler's CSV pipeline).")
endif()
# Bitmask passed as -DPROFILE_KERNEL=<value> to firmware + kernel builds.
# Bit 0 (1) = base profiling (zone start/end markers).
# Bit 1 (2) = also instrument dispatch cores (we have none, no effect).
# Bit 2 (4) = trace-only mode (no zones, only trace markers — incompatible).
# Bit 3 (8) = SUM mode — required for DeviceZoneScopedSumN1/N2 accumulator
#             zones to actually accumulate. Used by conv_3x3/reader.cpp
#             and any kernel that runs more zones per call than fit in
#             the 512-word L1 profiler buffer (e.g. inner-loop zones).
# Default 9 = base + SUM, no measurable cost over plain base.
set(TT_FOIL_PROFILE_KERNEL_BITMASK "9" CACHE STRING
    "PROFILE_KERNEL bitmask when TT_FOIL_DEVICE_PROFILER=ON")

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
    src/fast_dispatch.cpp
    src/profiling.cpp
    src/device_profile.cpp
    # Bundled tt-metal llrt sources (Phase B3-2 + B3-3): copied from
    # tt_metal/llrt/, namespace renamed ll_api → tt::foil::ll_api so they
    # don't conflict with libtt_metal.so's copies, and tt_memory.cpp's one
    # MetalContext::instance() call patched out (XIP-dump debug path).
    src/llrt_local/tt_memory.cpp
    src/llrt_local/tt_elffile.cpp
    # op_lib: host-side primitive op wrappers (see include/tt_foil/ops.hpp)
    src/op_lib/silu.cpp
    src/op_lib/eltwise_mul.cpp
    src/op_lib/silu_mul.cpp
    src/op_lib/add_rmsnorm.cpp
    src/op_lib/rmsnorm_rope.cpp
    src/op_lib/eltwise_add.cpp
    src/op_lib/softmax.cpp
    src/op_lib/embedding.cpp
    src/op_lib/rmsnorm.cpp
    src/op_lib/mha.cpp
    src/op_lib/kv_cache.cpp
    src/op_lib/matmul.cpp
    src/op_lib/rope.cpp
    src/op_lib/transpose_2d.cpp
    src/op_lib/gqa_fused.cpp
    src/op_lib/gqa_decode.cpp
    src/op_lib/argmax_row0.cpp
    src/op_lib/kv_append.cpp
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

if(TT_FOIL_ENABLE_TRACY)
    # Public so dependents of tt_foil see TRACY_ENABLE and link libtracy.so.
    # ZoneScoped uses RAII objects, so any TU that calls into the tt_foil API
    # may emit Tracy symbols.
    target_link_libraries(tt_foil PUBLIC tracy::TracyClient)
    target_compile_options(tt_foil PRIVATE -fno-omit-frame-pointer)
endif()
if(TT_FOIL_DEVICE_PROFILER)
    # Host-side compile gate. Activates the device_profile.cpp code path
    # that reads the profiler L1 buffer back after each dispatch. The
    # kernel-side counterpart is PROFILE_KERNEL in the firmware/kernel
    # SFPI build (see scripts/build_firmware.sh).
    target_compile_definitions(tt_foil PRIVATE TT_FOIL_DEVICE_PROFILER_ENABLED=1)
endif()

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

# ---- qwen3_run device CLI ----
# Device-only Qwen3-VL-2B inference. Reads token IDs from a binary file,
# runs prefill+decode on the chip, prints generated token IDs to stdout.
# Pair with models/qwen3_vl_2b/qwen3_chat.py for prompt → text round-tripping.
add_executable(qwen3_run models/qwen3_vl_2b/qwen3_run.cpp)
target_include_directories(qwen3_run PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(qwen3_run PRIVATE TT::Foil tt_foil_llrt)
set_target_properties(qwen3_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/qwen3_vl_2b")

# ---- qwen3vl_run device CLI ----
# Qwen3-VL-2B with variable-length prefill and optional visual embeddings.
# Pair with models/qwen3_vl_2b/qwen3vl_chat.py for image+text → text.
add_executable(qwen3vl_run models/qwen3_vl_2b/qwen3vl_run.cpp)
target_include_directories(qwen3vl_run PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(qwen3vl_run PRIVATE TT::Foil tt_foil_llrt)
set_target_properties(qwen3vl_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/qwen3_vl_2b")

# ---- tt_foil_inspect host CLI ----
# Reads soft-reset/L1 state of every Tensix core without doing init.
# Used to diagnose chip-state degradation after heavy workloads.
add_executable(tt_foil_inspect tools/tt_foil_inspect.cpp)
target_link_libraries(tt_foil_inspect PRIVATE tt_foil)
set_target_properties(tt_foil_inspect PROPERTIES
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
    # When the device profiler is enabled, pass the bitmask through to
    # build_firmware.sh so SFPI g++ compiles brisc.cc / ncrisc.cc / trisc.cc
    # with -DPROFILE_KERNEL=<bitmask>.
    set(_tt_foil_fw_profile_env "")
    if(TT_FOIL_DEVICE_PROFILER)
        set(_tt_foil_fw_profile_env
            "TT_FOIL_PROFILE_KERNEL=${TT_FOIL_PROFILE_KERNEL_BITMASK}")
    endif()
    add_custom_command(
        OUTPUT ${_tt_foil_fw_outputs}
        COMMAND ${CMAKE_COMMAND} -E env
            TT_METAL_ROOT=${TT_METAL_ROOT}
            TT_FOIL_BUILD=${CMAKE_BINARY_DIR}
            OUT_DIR=${CMAKE_BINARY_DIR}/firmware
            ${_tt_foil_fw_profile_env}
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
