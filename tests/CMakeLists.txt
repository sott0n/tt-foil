cmake_minimum_required(VERSION 3.21)

# Unit test: ELF loading (no hardware required)
add_executable(test_elf_load test_elf_load.cpp)
target_link_libraries(test_elf_load PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_elf_load COMMAND test_elf_load)

# Unit test: bump allocator (no hardware required)
add_executable(test_bump_alloc test_bump_alloc.cpp)
target_link_libraries(test_bump_alloc PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_bump_alloc COMMAND test_bump_alloc)

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

    # Tensix mailbox boot-time init (Phase B2 step 4)
    add_executable(test_mailbox_init test_mailbox_init.cpp)
    target_include_directories(test_mailbox_init PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_mailbox_init PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_mailbox_init COMMAND test_mailbox_init)

    # Firmware ELF load equivalence vs tt-metal (Phase B2 step 3)
    add_executable(test_firmware_load test_firmware_load.cpp)
    target_include_directories(test_firmware_load PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_firmware_load PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_firmware_load COMMAND test_firmware_load)

    # UMD direct open + L1 round trip (Phase B2 step 2)
    # No tt-metal CreateDevice: opens umd::Cluster directly. Must not run
    # alongside another process that has the same chip open.
    add_executable(test_umd_open test_umd_open.cpp)
    target_include_directories(test_umd_open PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(test_umd_open PRIVATE TT::Foil tt_foil_llrt)
    add_test(NAME test_umd_open COMMAND test_umd_open)
endif()
