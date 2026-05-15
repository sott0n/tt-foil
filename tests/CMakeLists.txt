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
