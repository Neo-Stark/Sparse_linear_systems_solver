cmake_minimum_required(VERSION 3.16)
project(practica5 CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(practica5 main.cu jacobi.cu)

set_target_properties(
        practica5
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
include_directories(include)
#add_subdirectory(external/matrix_reader)
#target_link_libraries(${CMAKE_PROJECT_NAME} MatrixMarketReader)
