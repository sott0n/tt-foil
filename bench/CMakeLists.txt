# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone performance benchmarks (NOT ctest). Built unconditionally as
# plain executables linking the runtime — driven by bench/run.sh, never part
# of the regression sweep.
add_executable(matmul_bench matmul_bench.cpp)
target_include_directories(matmul_bench PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(matmul_bench PRIVATE TT::Foil tt_foil_llrt)
