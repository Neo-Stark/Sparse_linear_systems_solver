cmake_minimum_required(VERSION 3.16)
project(sistemas_lineales CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

cuda_add_library(jacobi src/jacobiCuda.cu include/jacobiCuda.h src/jacobi.cpp src/SRJ_CUDA.cu include/SRJ_CUDA.h src/utilidades.cu include/utilidades.h src/jacobiOmp.cpp include/jacobiOmp.h src/SRJ.cpp include/SRJ.h src/SRJ_OMP.cpp include/SRJ_OMP.h)
target_link_libraries(jacobi OpenMP::OpenMP_CXX)
IF (OPENMP_FOUND)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF ()

add_executable(${PROJECT_NAME} src/main.cpp src/CSR.cpp)
target_link_libraries(${PROJECT_NAME} jacobi ${CUDA_cusparse_LIBRARY})

add_executable(test tests/algoritmos.cu src/CSR.cpp tests/matrices_test.h tests/tests_cuda.cu tests/tests_omp.cpp tests/tests_csr.cpp tests/matrices_test.cpp tests/tests.h tests/main.cu)
target_link_libraries(test jacobi ${CUDA_cusparse_LIBRARY})

include_directories(include/ .)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
