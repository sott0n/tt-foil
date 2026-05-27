cmake_minimum_required(VERSION 3.21)

add_executable(add_two_numbers main.cpp)
target_link_libraries(add_two_numbers PRIVATE TT::Foil)

tt_foil_add_kernels(
    NAME    add_two_numbers
    SCRIPT  ${CMAKE_CURRENT_SOURCE_DIR}/build_kernels.sh
    SOURCES kernels/add_brisc.cpp
            kernels/add_ncrisc.cpp
    OUTPUTS prebuilt/add_brisc.elf
            prebuilt/add_ncrisc.elf
            prebuilt/manifest.txt
)
add_dependencies(add_two_numbers kernels_add_two_numbers)
