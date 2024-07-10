cmake_minimum_required(VERSION 3.10)
project(libtorch_tensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(libtorch_tensor libtorch_tensor.cpp)
target_link_libraries(libtorch_tensor "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_tensor PROPERTY CXX_STANDARD 17)