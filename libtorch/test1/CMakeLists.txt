# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# project(libtorch_trace)

# find_package(Torch REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(${PROJECT_NAME} libtorch_trace.cpp)
# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# # Libtorch是基于C++14来实现的
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)


cmake_minimum_required(VERSION 3.10)
project(libtorch_trace)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(libtorch_trace libtorch_trace.cpp)
target_link_libraries(libtorch_trace "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_trace PROPERTY CXX_STANDARD 17)