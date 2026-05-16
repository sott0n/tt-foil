cmake_minimum_required(VERSION 3.21)

# Unit test: ELF loading (no hardware required)
add_executable(test_elf_load test_elf_load.cpp)
target_include_directories(test_elf_load PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(test_elf_load PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_elf_load COMMAND test_elf_load)

# Unit test: bump allocator (no hardware required)
add_executable(test_bump_alloc test_bump_alloc.cpp)
target_link_libraries(test_bump_alloc PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_bump_alloc COMMAND test_bump_alloc)

# Unit test: RiscId → HAL index mapping (v4-1)
add_executable(test_risc_ids test_risc_ids.cpp)
target_include_directories(test_risc_ids PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(test_risc_ids PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_risc_ids COMMAND test_risc_ids)

# Unit test: standalone HAL instantiation (Phase B3 probe)
add_executable(test_hal_standalone test_hal_standalone.cpp)
target_link_libraries(test_hal_standalone PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_hal_standalone COMMAND test_hal_standalone)

# Unit test: firmware ELF discovery (no hardware required)
add_executable(test_firmware_paths test_firmware_paths.cpp)
target_include_directories(test_firmware_paths PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(test_firmware_paths PRIVATE TT::Foil)
add_test(NAME test_firmware_paths COMMAND test_firmware_paths)

# Integration test: add two numbers (requires Blackhole hardware)
# Disabled by default; enable with -DTT_FOIL_HW_TESTS=ON
option(TT_FOIL_HW_TESTS "Build hardware integration tests" OFF)
if(TT_FOIL_HW_TESTS)
    add_executable(test_add_kernel test_add_kernel.cpp)
    target_link_libraries(test_add_kernel PRIVATE TT::Foil)
    add_test(NAME test_add_kernel COMMAND test_add_kernel)

    # BRISC + NCRISC concurrent dispatch (Phase A)
    add_executable(test_brisc_ncrisc test_brisc_ncrisc.cpp)
    target_link_libraries(test_brisc_ncrisc PRIVATE TT::Foil)
    add_test(NAME test_brisc_ncrisc COMMAND test_brisc_ncrisc)

    # NOC inter-core passthrough end-to-end (v3-3..v3-5)
    add_executable(test_noc_passthrough test_noc_passthrough.cpp)
    target_link_libraries(test_noc_passthrough PRIVATE TT::Foil)
    add_test(NAME test_noc_passthrough COMMAND test_noc_passthrough)

    # CB config blob layout (v4-3) — uses HW to validate L1 round-trip but
    # doesn't run any TRISC firmware
    add_executable(test_cb_config test_cb_config.cpp)
    target_include_directories(test_cb_config PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_cb_config PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_cb_config COMMAND test_cb_config)

    # Multi-kernel dispatch over multiple cores (v3-2)
    add_executable(test_multi_kernel test_multi_kernel.cpp)
    target_link_libraries(test_multi_kernel PRIVATE TT::Foil)
    add_test(NAME test_multi_kernel COMMAND test_multi_kernel)

    # End-to-end tile copy via all 5 Tensix RISCs (v4-6)
    add_executable(test_tile_copy test_tile_copy.cpp)
    target_include_directories(test_tile_copy PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_tile_copy PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_tile_copy COMMAND test_tile_copy)

    # bf16 ReLU through the SFPU (v5.5) — 5-RISC pipeline, single tile.
    add_executable(test_relu test_relu.cpp)
    target_include_directories(test_relu PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_relu PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_relu COMMAND test_relu)

    # bf16 eltwise add_tiles (v5.6) — 2-input compute, single tile, FPU.
    add_executable(test_add_tiles test_add_tiles.cpp)
    target_include_directories(test_add_tiles PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_add_tiles PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_add_tiles COMMAND test_add_tiles)

    # Row-broadcast bias add (v5.7) — out[i,j] = in[i,j] + bias[0,j].
    # The Linear/FC layer bias-add primitive.
    add_executable(test_bias_add test_bias_add.cpp)
    target_include_directories(test_bias_add PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_bias_add PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_bias_add COMMAND test_bias_add)

    # Row-sharded DRAM matmul across 2 Tensix cores (v6-1).  Per-core
    # slice height is MM_MT_2C (default 1); the test sees a global M of
    # 2 × MM_MT_2C tiles.  Kernel ELFs and the test must agree on the
    # per-core triple (MM_MT_2C, MM_KT, MM_NT).
    if(NOT DEFINED MM_MT_2C)
        set(MM_MT_2C 1)
    endif()
    add_executable(test_matmul_2core test_matmul_2core.cpp)
    target_include_directories(test_matmul_2core PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(test_matmul_2core PRIVATE
        MM_MT=${MM_MT_2C} MM_KT=${MM_KT} MM_NT=${MM_NT})
    target_link_libraries(test_matmul_2core PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_2core COMMAND test_matmul_2core)

    # 2×2 grid block-sharded DRAM matmul (v6-2). Per-core slice is
    # MM_MT_2x2 × MM_NT_2x2 tiles; global problem is 2x larger in each
    # of M and N. Kernel ELFs and test must agree on the per-core triple.
    if(NOT DEFINED MM_MT_2x2)
        set(MM_MT_2x2 1)
    endif()
    if(NOT DEFINED MM_NT_2x2)
        set(MM_NT_2x2 1)
    endif()
    add_executable(test_matmul_2x2 test_matmul_2x2.cpp)
    target_include_directories(test_matmul_2x2 PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(test_matmul_2x2 PRIVATE
        MM_MT=${MM_MT_2x2} MM_KT=${MM_KT} MM_NT=${MM_NT_2x2})
    target_link_libraries(test_matmul_2x2 PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_2x2 COMMAND test_matmul_2x2)

    # Single-tile bf16 matmul (v5-1) — full 5-RISC + 2-input + matmul_tiles
    add_executable(test_matmul_1tile test_matmul_1tile.cpp)
    target_include_directories(test_matmul_1tile PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_matmul_1tile PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_1tile COMMAND test_matmul_1tile)

    # K-loop bf16 matmul (v5-2) — DST_ACCUM_MODE=true, Kt partial products
    # summed exactly in fp32 DST regs.  MM_KT must match the value the
    # kernel ELFs were compiled with (default 4).
    if(NOT DEFINED MM_KT)
        set(MM_KT 4)
    endif()
    add_executable(test_matmul_kt test_matmul_kt.cpp)
    target_include_directories(test_matmul_kt PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(test_matmul_kt PRIVATE MM_KT=${MM_KT})
    target_link_libraries(test_matmul_kt PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_kt COMMAND test_matmul_kt)

    # Host-side DRAM round trip (v5-4 smoke) — write_buffer + read_buffer
    # on BufferLocation::DRAM.  Runs before the kernel-side DRAM matmul so
    # we catch host I/O bugs separately.
    add_executable(test_dram_roundtrip test_dram_roundtrip.cpp)
    target_link_libraries(test_dram_roundtrip PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_dram_roundtrip COMMAND test_dram_roundtrip)

    # General Mt×Kt × Kt×Nt single-core matmul (v5-3) — outer-product
    # schedule with one DST-resident accumulator per output tile.  Tile
    # counts must match what the kernel ELFs were compiled with.
    if(NOT DEFINED MM_MT)
        set(MM_MT 2)
    endif()
    if(NOT DEFINED MM_NT)
        set(MM_NT 2)
    endif()
    add_executable(test_matmul_mnk test_matmul_mnk.cpp)
    target_include_directories(test_matmul_mnk PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(test_matmul_mnk PRIVATE
        MM_MT=${MM_MT} MM_KT=${MM_KT} MM_NT=${MM_NT})
    target_link_libraries(test_matmul_mnk PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_mnk COMMAND test_matmul_mnk)

    # DRAM-backed matmul (v5-4) — same Mt×Kt×Nt schedule, A/B/C in DRAM.
    # Lifts the L1 capacity wall from v5-3.
    add_executable(test_matmul_dram test_matmul_dram.cpp)
    target_include_directories(test_matmul_dram PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_compile_definitions(test_matmul_dram PRIVATE
        MM_MT=${MM_MT} MM_KT=${MM_KT} MM_NT=${MM_NT})
    target_link_libraries(test_matmul_dram PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_matmul_dram COMMAND test_matmul_dram)

    # Multi-core boot via device_open (v3-1)
    add_executable(test_multi_core_boot test_multi_core_boot.cpp)
    target_include_directories(test_multi_core_boot PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_multi_core_boot PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_multi_core_boot COMMAND test_multi_core_boot)

    # UMD direct open + L1 round trip — opens umd::Cluster directly, no
    # tt-metal symbols required.
    add_executable(test_umd_open test_umd_open.cpp)
    target_include_directories(test_umd_open PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_umd_open PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_umd_open COMMAND test_umd_open)

    # The Phase B2 step-by-step tests (test_cold_boot, test_firmware_load,
    # test_mailbox_init, test_core_info_init, test_bank_tables_init) were
    # development scaffolding that linked against tt-metal's CreateDevice +
    # MetalContext to byte-compare our writes against tt-metal's. With B3-4
    # those symbols are no longer linked. The production-equivalent
    # verification — full cold boot via device_open() + kernel launch — is
    # covered by test_add_kernel and test_brisc_ncrisc above. The .cpp files
    # remain on disk for reference but are no longer in the build.
endif()
