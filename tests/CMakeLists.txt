cmake_minimum_required(VERSION 3.21)

# Unit test: ELF loading (no hardware required)
add_executable(test_elf_load test_elf_load.cpp)
target_link_libraries(test_elf_load PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_elf_load COMMAND test_elf_load)

# Unit test: bump allocator (no hardware required)
add_executable(test_bump_alloc test_bump_alloc.cpp)
target_link_libraries(test_bump_alloc PRIVATE TT::Foil tt_foil_llrt)
add_test(NAME test_bump_alloc COMMAND test_bump_alloc)

# Integration test: add two numbers (requires Blackhole hardware)
# Disabled by default; enable with -DTT_FOIL_HW_TESTS=ON
option(TT_FOIL_HW_TESTS "Build hardware integration tests" OFF)
if(TT_FOIL_HW_TESTS)
    add_executable(test_add_kernel test_add_kernel.cpp)
    target_link_libraries(test_add_kernel PRIVATE TT::Foil)
    add_test(NAME test_add_kernel COMMAND test_add_kernel)
endif()
