//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_TESTS_H
#define SISTEMAS_LINEALES_TESTS_H

// tests_jacobi
void test_jacobi_CUDA();

void test_jacobi_OMP();

void jacobi_secuencial();

void jacobi_clasico();

void SOR_CSR();

void SOR();

void srj_secuencial();

void printResult(int iter, const int *x);

//tests_cuda
void test_multiplicacion_cuda();

void test_reduce_max();

//test CSR
void test_constructor_csr();

#endif //SISTEMAS_LINEALES_TESTS_H
