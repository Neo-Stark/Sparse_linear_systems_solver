cmake_minimum_required(VERSION 3.16)
project(sistemas_lineales CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

add_library(jacobi jacobi.cu SRJ.cu include/SRJ.h)
target_link_libraries(jacobi OpenMP::OpenMP_CXX)
IF (OPENMP_FOUND)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

add_executable(${PROJECT_NAME} main.cpp CSR.cpp)
target_link_libraries(${PROJECT_NAME} jacobi ${CUDA_cusparse_LIBRARY})

add_executable(test tests/tests_jacobi.cu CSR.cpp tests/matrices_test.h tests/tests_cuda.cu tests/tests_omp.cpp tests/tests_csr.cpp tests/matrices_test.cpp tests/tests.h tests/main.cpp)
target_link_libraries(test jacobi ${CUDA_cusparse_LIBRARY})

include_directories(include/ .)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
