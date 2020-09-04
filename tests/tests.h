//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_TESTS_H
#define SISTEMAS_LINEALES_TESTS_H

// tests_jacobi
void test_jacobi_CUDA(string _matriz, string _rhs, string _x0);

void test_jacobi_OMP(string _matriz, string _rhs, string _x0);

void jacobi_secuencial(string _matriz, string _rhs, string _x0);

void jacobi_clasico();

void SOR_CSR(string _matriz, string _rhs, string _x0, double omega);

void SOR();

void srj_secuencial(string _matriz, string _rhs, string _x0, string srjSch);

void printResult(int iter, const int *x);

//tests_cuda
void test_multiplicacion_cuda();

void test_reduce_max();

//test CSR
void test_constructor_csr();

#endif //SISTEMAS_LINEALES_TESTS_H
