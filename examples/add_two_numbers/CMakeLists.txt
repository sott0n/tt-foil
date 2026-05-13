cmake_minimum_required(VERSION 3.21)

add_executable(add_two_numbers main.cpp)
target_link_libraries(add_two_numbers PRIVATE TT::Foil)
