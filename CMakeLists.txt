cmake_minimum_required(VERSION 3.16)
project(practica5 CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA)

cuda_add_executable(practica5 main.cu)
cuda_add_library(jacobi jacobi.cu jacobi.cuh)

include_directories(include .)

target_link_libraries(${PROJECT_NAME} jacobi ${CUDA_cusparse_LIBRARY})

set_target_properties(
        practica5
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
