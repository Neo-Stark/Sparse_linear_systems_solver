#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
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

void reduction_omp() {
    vector<int> v(12);
    for (int i = 0; i < v.size(); i++) v[i] = i;
    int maximo = 0;
#pragma omp parallel
    {
#pragma omp for reduction (max : maximo)
        for (int i = 0; i < v.size(); i++) {
            maximo = max(maximo, v[i]);
        }
#pragma omp master
        cout << "maximo : " << maximo << " hebra: " << omp_get_thread_num() << endl;
    }
}

int main(int argc, char **argv) {
//    jacobi_secuencial();
    test_jacobi_OMP();
//    test_jacobi_CUDA();
//    SOR_CSR();
//    SOR();
//    jacobi_clasico();
//    srj_secuencial();
//    test_multiplicacion_cuda();
//    test_reduce_max();
//    test_constructor_csr();
}
