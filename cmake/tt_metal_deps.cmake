# tt_metal_deps.cmake
# Links tt-foil against a pre-built tt-metal installation.
#
# Required: set TT_METAL_BUILD_DIR (cmake variable or env) to the tt-metal
# build directory, e.g. /path/to/tt-metal/build_Release
#
# The build dir must contain:
#   lib/libtt_metal.so    -- llrt + HAL symbols
#   lib/libtt-umd.so      -- UMD PCIe driver
#   lib/cmake/umd/        -- UMD cmake config
#   include/              -- installed headers (tt_stl, tt-logger, enchantum...)
#
# Build tt-metal once to produce these artifacts:
#   cd /path/to/tt-metal
#   cmake -B build_Release -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_Release -j$(nproc) --target tt_metal

# ---- Locate the tt-metal build directory ----
if(NOT DEFINED TT_METAL_BUILD_DIR)
    if(DEFINED ENV{TT_METAL_BUILD_DIR})
        set(TT_METAL_BUILD_DIR "$ENV{TT_METAL_BUILD_DIR}")
    else()
        # Fallback: build_Release next to the submodule checkout
        set(TT_METAL_BUILD_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-metal/build_Release")
    endif()
endif()

if(NOT EXISTS "${TT_METAL_BUILD_DIR}/lib/libtt_metal.so")
    message(FATAL_ERROR
        "tt-metal build not found at TT_METAL_BUILD_DIR=${TT_METAL_BUILD_DIR}.\n"
        "Set TT_METAL_BUILD_DIR to a tt-metal build directory that contains "
        "lib/libtt_metal.so.  Build tt-metal first:\n"
        "  cmake -B build_Release && cmake --build build_Release --target tt_metal")
endif()

message(STATUS "tt-foil: using tt-metal build at ${TT_METAL_BUILD_DIR}")

# Derive the tt-metal source root.
# If TT_METAL_SOURCE_DIR is explicitly provided, use it; otherwise infer from
# the build directory (the build dir is typically build_Release/ inside the
# source root, so parent is the source root).
if(DEFINED TT_METAL_SOURCE_DIR)
    set(TT_METAL_ROOT "${TT_METAL_SOURCE_DIR}")
elseif(DEFINED ENV{TT_METAL_SOURCE_DIR})
    set(TT_METAL_ROOT "$ENV{TT_METAL_SOURCE_DIR}")
else()
    # Infer: parent of the build dir is the source root (e.g. .../tt-metal/build_Release -> .../tt-metal)
    cmake_path(GET TT_METAL_BUILD_DIR PARENT_PATH TT_METAL_ROOT_INFERRED)
    if(EXISTS "${TT_METAL_ROOT_INFERRED}/tt_metal/llrt/hal.hpp")
        set(TT_METAL_ROOT "${TT_METAL_ROOT_INFERRED}")
        message(STATUS "tt-foil: inferred tt-metal source at ${TT_METAL_ROOT}")
    else()
        # Fall back to submodule
        set(TT_METAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-metal")
        message(STATUS "tt-foil: using submodule at ${TT_METAL_ROOT}")
    endif()
endif()

# ---- UMD (from tt-metal cmake package) ----
find_package(umd REQUIRED
    PATHS "${TT_METAL_BUILD_DIR}/lib/cmake/umd"
    NO_DEFAULT_PATH)

# ---- tt_foil_llrt: includes-only interface ----
# Phase B3-4: libtt_metal.so is no longer linked. HAL sources are compiled
# directly into tt_foil (see top-level CMakeLists.txt); ll_api::memory is
# bundled under src/llrt_local/. The only runtime dependency now is UMD.
add_library(tt_foil_llrt INTERFACE)
target_link_libraries(tt_foil_llrt INTERFACE
    umd::tt-umd
)
target_include_directories(tt_foil_llrt INTERFACE
    # tt-metal source tree (for llrt/*.hpp, hw/inc/*, hostdevcommon/*,
    # hal/*.hpp + generated dev_msgs)
    "${TT_METAL_ROOT}"
    "${TT_METAL_ROOT}/tt_metal"
    "${TT_METAL_ROOT}/tt_metal/llrt"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx"
    "${TT_METAL_ROOT}/tt_metal/hw/inc"
    "${TT_METAL_ROOT}/tt_metal/hw/inc/hostdev"
    "${TT_METAL_ROOT}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
    "${TT_METAL_ROOT}/tt_metal/hostdevcommon"
    "${TT_METAL_ROOT}/tt_metal/api"
    "${TT_METAL_ROOT}/tt_metal/api/tt-metalium"
    "${TT_METAL_ROOT}/tt_metal/impl"
    # tt-metal build tree (for installed headers: tt_stl, tt-logger,
    # enchantum, fmt, spdlog)
    "${TT_METAL_BUILD_DIR}/include"
    # tt_stl source tree (enum.hpp and others not installed to build/include)
    "${TT_METAL_ROOT}/tt_stl"
)
target_compile_definitions(tt_foil_llrt INTERFACE
    ARCH_BLACKHOLE
    SPDLOG_FMT_EXTERNAL
)

# tt_foil_hal_local: static lib with HAL sources compiled from the tt-metal
# tree. Lives separately from tt_foil so include dirs and properties stay
# focused; tt_foil PRIVATE-links it.
add_library(tt_foil_hal_local STATIC
    "${TT_METAL_ROOT}/tt_metal/llrt/hal.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_active_eth.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_idle_eth.cpp"
    "${TT_METAL_ROOT}/tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_dram.cpp"
    # Blackhole-only build: stub out the WH/QA initialisers that hal.cpp
    # still references via its arch switch.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llrt_local/hal_stubs.cpp"
)
target_link_libraries(tt_foil_hal_local PUBLIC tt_foil_llrt)
set_target_properties(tt_foil_hal_local PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Convenience alias to match existing references.
add_library(HAL::1xx ALIAS tt_foil_hal_local)
