cmake_minimum_required(VERSION 3.8)
project(simulator)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp Barrier.hpp)
add_executable(simulator ${SOURCE_FILES})