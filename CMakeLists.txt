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
    src/runtime_impl.cpp
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
)

if(TT_FOIL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TT_FOIL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
