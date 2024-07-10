cmake_minimum_required(VERSION 3.10)
project(libtorch_trace)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
add_executable(libtorch_trace libtorch_trace.cpp)
target_link_libraries(libtorch_trace "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_trace PROPERTY CXX_STANDARD 17)