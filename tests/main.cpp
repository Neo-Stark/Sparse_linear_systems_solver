//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//
#include <iostream>
#include <vector>
#include "matrices_test.h"
#include "tests.h"
#include <omp.h>

void mul_matriz_vector_sec() {
    int filas = 8;
    auto matriz = m.m4;
    auto _x = v_x._4_5;
    for (int i = 0; i < filas; ++i) {
        double sum = 0;
        for (unsigned int j = 0; j < filas; j++) {
            sum += matriz[i][j] * _x[j];
        }
        std::cout << sum << "  ";
    }
}



int main(int argc, char **argv) {
    if (argc < 5) {
        cout << "Uso: test [mtx] [rhs] [x0] [srjSch]";
        abort();
    }
    string matriz = argv[1];
    string rhs = argv[2];
    string x0 = argv[3];
    string srjSch = argv[4];
//    test_jacobi_CUDA(matriz, rhs, x0);
    test_jacobi_OMP(matriz, rhs, x0);
//    jacobi_secuencial(matriz, rhs, x0);
//    srj_secuencial(matriz, rhs, x0, srjSch);
//    SOR_CSR(matriz, rhs, x0, 1);
//    SOR();
//    jacobi_clasico();
//    test_multiplicacion_cuda();
//    test_reduce_max();
//    test_constructor_csr();
}
