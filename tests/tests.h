//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_TESTS_H
#define SISTEMAS_LINEALES_TESTS_H

#include "CSR.h"

// tests_jacobi
double test_jacobi_CUDA(CSR matriz, double *_b, string x0);

double test_jacobi_OMP(CSR matriz, double *_b, string x0);

double jacobi_secuencial(CSR matriz, double *_b, string x0);

double SOR(CSR matriz, double *_b, string x0, double omega);

double SOR_CUDA(CSR matriz, double *_b, string x0, double omega);

double SOR_OMP(CSR matriz, double *_b, string x0, double omega);

double srj_secuencial(CSR matriz, double *_b, string x0, string srjSch);

double srj_CUDA(CSR matriz, double *_b, string x0, string srjSch);

double srj_OMP(CSR matriz, double *_b, string x0, string srjSch);

double printResult(int iter, const int *x);

//tests_cuda
void test_multiplicacion_cuda();

void test_reduce_max();

//test CSR
void test_constructor_csr();

#endif //SISTEMAS_LINEALES_TESTS_H
