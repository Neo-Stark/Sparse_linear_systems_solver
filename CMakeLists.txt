cmake_minimum_required(VERSION 3.16)
project(sistemas_lineales CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA)

cuda_add_library(jacobi jacobi.cu include/jacobi.h)
cuda_add_executable(${PROJECT_NAME} main.cpp CSR.cpp)

target_link_libraries(${PROJECT_NAME} jacobi ${CUDA_cusparse_LIBRARY})

include_directories(include/ .)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
