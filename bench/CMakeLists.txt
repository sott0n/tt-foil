# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone performance benchmarks (NOT ctest). Built unconditionally as
# plain executables linking the runtime — driven by bench/run.sh, never part
# of the regression sweep.
add_executable(matmul_bench matmul_bench.cpp)
target_include_directories(matmul_bench PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(matmul_bench PRIVATE TT::Foil tt_foil_llrt)

# Weight-stationary matmul validation + WS-vs-stock timing. Needs tile_utils.
add_executable(matmul_ws_check matmul_ws_check.cpp)
target_include_directories(matmul_ws_check PRIVATE
    "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tests")
target_link_libraries(matmul_ws_check PRIVATE TT::Foil tt_foil_llrt)
