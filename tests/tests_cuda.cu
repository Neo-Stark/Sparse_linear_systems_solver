//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "matrices_test.h"
#include <kernels.cuh>
#include <readers.h>
#include <CSR.h>
#include <iostream>
#include <jacobi.h>
#include "cpu_seconds.h"

using namespace std;
#define BLOCKSIZE 64


void test_multiplicacion_cuda() {

    CSR matriz(m._4);
    double *x, *x_d, *y, *y_d, *A;
    unsigned int *col_ind, *row_ptr;
    x = v_x._4_5;
    y = new double[matriz.getFilas()];
    cudaMalloc(&A, sizeof(double) * matriz.getVal().size());
    cudaMalloc(&col_ind, sizeof(int) * matriz.getColInd().size());
    cudaMalloc(&row_ptr, sizeof(int) * matriz.getRowPtr().size());
    cudaMalloc(&x_d, sizeof(double) * matriz.getColumnas());
    cudaMalloc(&y_d, sizeof(double) * matriz.getFilas());
    cudaMemcpy(A, matriz.getVal().data(), matriz.getVal().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.getColInd().data(), matriz.getColInd().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.getRowPtr().data(), matriz.getRowPtr().size() * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Row ptr: ";
    for (auto row : matriz.getRowPtr()) cout << row << " ";
    cout << " size: " << matriz.getRowPtr().size();
    cout << endl;

    dim3 block_size(BLOCKSIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * matriz.getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(x_d, x, matriz.getColumnas() * sizeof(double), cudaMemcpyHostToDevice);

    matrix_vector_multiplication<double><<<grid_size, block_size>>>(A, col_ind, row_ptr, x_d,
                                                                    y_d, matriz.getFilas());

    cudaMemcpy(y, y_d, matriz.getFilas() * sizeof(double), cudaMemcpyDeviceToHost);
    matriz.printMatrix();
    cout << "Solución multiplicación: " << endl;
    for (auto i = 0; i < matriz.getFilas(); i++) cout << y[i] << ", ";
    cout << "\b\b\n";
}

void test_reduce_max_wrapper(double *h_v, int n) {
    // test_reduce_max
    double *d_vi, *dva;
    double x[4] = {-0.5, 4002, -2.1, 0.625};
    cudaMalloc(&d_vi, sizeof(double) * n);
    cudaMalloc(&dva, sizeof(double) * 4);
    cudaMemcpy(d_vi, h_v, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dva, x, sizeof(double) * 4, cudaMemcpyHostToDevice);
    double t1, t2;
    cout.precision(20);
    jacobi test32(CSR(m._1), 32);
    t1 = cpuSecond();
    auto maximo = test32.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test32   - Máximo: " << maximo << " tiempo " << t2 << endl;

    jacobi test64(CSR(m._1), 64);
    t1 = cpuSecond();
    maximo = test64.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test64   - Máximo: " << maximo << " tiempo " << t2 << endl;

    jacobi test128(CSR(m._1), 128);
    t1 = cpuSecond();
    maximo = test128.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test128  - Máximo: " << maximo << " tiempo " << t2 << endl;

    jacobi test256(CSR(m._1), 256);
    t1 = cpuSecond();
    maximo = test256.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test256  - Máximo: " << maximo << " tiempo " << t2 << endl;

    jacobi test512(CSR(m._1), 512);
    t1 = cpuSecond();
    maximo = test512.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test512  - Máximo: " << maximo << " tiempo " << t2 << endl;

    jacobi test1024(CSR(m._1), 1024);
    t1 = cpuSecond();
    maximo = test1024.reduce_max_CUDA(d_vi, n);
    t2 = cpuSecond() - t1;
    cout << "test1024 - Máximo: " << maximo << " tiempo " << t2 << endl;

    cudaFree(d_vi);
}

void test_reduce_max() {
    int select = 0;
    switch (select) {
        case 0: {
            auto n = 675;
            auto h_v = new double[n];
            for (int i = 0; i < n; i++) {
                h_v[i] = 2e2 + i + 1.5;
            }
            test_reduce_max_wrapper(h_v, n);
            free(h_v);
            break;
        }
        case 1: {
            double h_v[4] = {0.5, 2.18182, -1.1, 1.625};
            test_reduce_max_wrapper(h_v, 4);
            break;
        }
        case 2: {
            double x[4] = {-0.5, 1.18182, -2.1, 0.625};
            test_reduce_max_wrapper(x, 4);
            break;
        }
        case 3: {
            ifstream fv("../vector675");
            double *x = new double[675];
            if (fv.is_open()) for (int i = 0; i < 675; i++) fv >> x[i];
            test_reduce_max_wrapper(x, 675);
            break;
        }
    }
}