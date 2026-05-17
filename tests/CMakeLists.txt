cmake_minimum_required(VERSION 3.21)

# TT_FOIL_DEVICE picks the PCIe board index used by HW tests. Resolution
# order: CMake -DTT_FOIL_DEVICE=N > env TT_FOIL_DEVICE > 0.
if(NOT DEFINED TT_FOIL_DEVICE)
    if(DEFINED ENV{TT_FOIL_DEVICE})
        set(TT_FOIL_DEVICE "$ENV{TT_FOIL_DEVICE}")
    else()
        set(TT_FOIL_DEVICE 0)
    endif()
endif()
message(STATUS "tt-foil tests: TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")

# ---------------------------------------------------------------------------
# tt_foil_unit_test(name src ...)
#   Adds a host-side unit test (no HW). Optional remaining args are passed as
#   target_compile_definitions PRIVATE values.
# ---------------------------------------------------------------------------
function(tt_foil_unit_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE TT::Foil tt_foil_llrt)
    if(ARGN)
        target_compile_definitions(${name} PRIVATE ${ARGN})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "unit" TIMEOUT 30)
endfunction()

# ---------------------------------------------------------------------------
# tt_foil_hw_test(name src kernel_dir ...)
#   Adds a hardware integration test. The test binary gets:
#     TT_FOIL_KERNEL_DIR  pointed at `kernel_dir`
#     TT_FOIL_DEVICE      pointed at the CMake variable above
#   All HW tests share `RESOURCE_LOCK chip` so ctest serialises them
#   without running them concurrently.  Remaining args go to
#   target_compile_definitions PRIVATE.
# ---------------------------------------------------------------------------
function(tt_foil_hw_test name src kernel_dir)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(${name} PRIVATE TT::Foil tt_foil_llrt)
    if(ARGN)
        target_compile_definitions(${name} PRIVATE ${ARGN})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    # set_tests_properties splits the ENVIRONMENT value on semicolons
    # for its own list parsing; pass it as a single quoted string and
    # set each property in its own call so list-ish values don't get
    # split into argument-count errors.
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "TT_FOIL_KERNEL_DIR=${kernel_dir};TT_FOIL_DEVICE=${TT_FOIL_DEVICE}")
    set_tests_properties(${name} PROPERTIES RESOURCE_LOCK chip)
    set_tests_properties(${name} PROPERTIES LABELS "hw")
    set_tests_properties(${name} PROPERTIES TIMEOUT 90)
endfunction()

# =============================================================================
# Host-only unit tests
# =============================================================================
tt_foil_unit_test(test_elf_load        test_elf_load.cpp)
tt_foil_unit_test(test_bump_alloc      test_bump_alloc.cpp)
tt_foil_unit_test(test_risc_ids        test_risc_ids.cpp)
tt_foil_unit_test(test_hal_standalone  test_hal_standalone.cpp)
tt_foil_unit_test(test_firmware_paths  test_firmware_paths.cpp)

# =============================================================================
# Hardware integration tests
# Enable with -DTT_FOIL_HW_TESTS=ON
# =============================================================================
option(TT_FOIL_HW_TESTS "Build hardware integration tests" OFF)
if(NOT TT_FOIL_HW_TESTS)
    return()
endif()

set(EX "${CMAKE_SOURCE_DIR}/examples")

# NOTE: HW tests assume the board is in a healthy state at ctest
# invocation time. If you've been crashing kernels, run a one-shot
# `tt-smi -r <TT_FOIL_DEVICE>` before `ctest`. RESOURCE_LOCK chip
# serialises HW tests within one ctest run, but ctest cannot revive a
# chip that a prior test hung.

# Per-config tile-count defaults. Override via CMake -DMM_XX=N.
macro(_tt_default var val)
    if(NOT DEFINED ${var})
        set(${var} ${val})
    endif()
endmacro()
_tt_default(MM_MT      2)
_tt_default(MM_KT      4)
_tt_default(MM_NT      2)
_tt_default(MM_MT_2C   1)
_tt_default(MM_MT_2x2  1)
_tt_default(MM_NT_2x2  1)
_tt_default(MM_MT_2CMC 1)
_tt_default(MM_MT_4x2  1)
_tt_default(MM_NT_4x2  1)
_tt_default(MM_MT_C1   2)
_tt_default(MM_KT_C1   2)
_tt_default(MM_NT_C1   2)
_tt_default(MM_MT_C3   1)
_tt_default(MM_KT_C3   9)
_tt_default(MM_NT_C3   2)
_tt_default(MM_MT_C32  1)
_tt_default(MM_KT_C32  9)
_tt_default(MM_NT_C32  2)
_tt_default(MM_MT_CS2  1)
_tt_default(MM_KT_CS2  9)
_tt_default(MM_NT_CS2  2)

# ---- Boot / IO smoke (no kernel ELFs needed beyond firmware) ----
tt_foil_hw_test(test_umd_open         test_umd_open.cpp         "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_multi_core_boot  test_multi_core_boot.cpp  "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_dram_roundtrip   test_dram_roundtrip.cpp   "${EX}/add_two_numbers/prebuilt")

# ---- v1..v4: foundational dispatch + 5-RISC + CBs ----
tt_foil_hw_test(test_add_kernel       test_add_kernel.cpp       "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_brisc_ncrisc     test_brisc_ncrisc.cpp     "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_multi_kernel     test_multi_kernel.cpp     "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_noc_passthrough  test_noc_passthrough.cpp  "${EX}/noc_passthrough/prebuilt")
tt_foil_hw_test(test_cb_config        test_cb_config.cpp        "${EX}/add_two_numbers/prebuilt")
tt_foil_hw_test(test_tile_copy        test_tile_copy.cpp        "${EX}/tile_copy/prebuilt")

# ---- v5: matmul + ReLU + add + bias ----
tt_foil_hw_test(test_matmul_1tile     test_matmul_1tile.cpp     "${EX}/matmul_1tile/prebuilt")
tt_foil_hw_test(test_matmul_kt        test_matmul_kt.cpp        "${EX}/matmul_kt/prebuilt"
    "MM_KT=${MM_KT}")
tt_foil_hw_test(test_matmul_mnk       test_matmul_mnk.cpp       "${EX}/matmul_mnk/prebuilt"
    "MM_MT=${MM_MT}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_dram      test_matmul_dram.cpp      "${EX}/matmul_dram/prebuilt"
    "MM_MT=${MM_MT}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_relu             test_relu.cpp             "${EX}/relu/prebuilt")
tt_foil_hw_test(test_add_tiles        test_add_tiles.cpp        "${EX}/add_tiles/prebuilt")
tt_foil_hw_test(test_bias_add         test_bias_add.cpp         "${EX}/bias_add/prebuilt")

# ---- v6: multi-core sharding + NOC mcast ----
tt_foil_hw_test(test_matmul_2core         test_matmul_2core.cpp        "${EX}/matmul_2core/prebuilt"
    "MM_MT=${MM_MT_2C}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_2x2           test_matmul_2x2.cpp          "${EX}/matmul_2x2/prebuilt"
    "MM_MT=${MM_MT_2x2}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT_2x2}")
tt_foil_hw_test(test_matmul_2core_mcast   test_matmul_2core_mcast.cpp  "${EX}/matmul_2core_mcast/prebuilt"
    "MM_MT=${MM_MT_2CMC}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT}")
tt_foil_hw_test(test_matmul_4x2           test_matmul_4x2.cpp          "${EX}/matmul_4x2/prebuilt"
    "MM_MT=${MM_MT_4x2}" "MM_KT=${MM_KT}" "MM_NT=${MM_NT_4x2}")

# ---- v7: conv (1×1, 3×3, sharded, stride=2) ----
tt_foil_hw_test(test_conv_1x1         test_conv_1x1.cpp         "${EX}/conv_1x1/prebuilt"
    "MM_MT=${MM_MT_C1}"  "MM_KT=${MM_KT_C1}"  "MM_NT=${MM_NT_C1}")
tt_foil_hw_test(test_conv_3x3         test_conv_3x3.cpp         "${EX}/conv_3x3/prebuilt"
    "MM_MT=${MM_MT_C3}"  "MM_KT=${MM_KT_C3}"  "MM_NT=${MM_NT_C3}")
tt_foil_hw_test(test_conv_3x3_2core   test_conv_3x3_2core.cpp   "${EX}/conv_3x3_2core/prebuilt"
    "MM_MT=${MM_MT_C32}" "MM_KT=${MM_KT_C32}" "MM_NT=${MM_NT_C32}")
tt_foil_hw_test(test_conv_3x3_s2      test_conv_3x3_s2.cpp      "${EX}/conv_3x3_s2/prebuilt"
    "MM_MT=${MM_MT_CS2}" "MM_KT=${MM_KT_CS2}" "MM_NT=${MM_NT_CS2}")

# NOTE: The Phase B2 step-by-step tests (test_cold_boot, test_firmware_load,
# test_mailbox_init, test_core_info_init, test_bank_tables_init) were
# development scaffolding that linked against tt-metal's CreateDevice +
# MetalContext. They were removed when libtt_metal.so was unlinked (B3-4).
# Their production-equivalent coverage lives in test_add_kernel +
# test_brisc_ncrisc.
