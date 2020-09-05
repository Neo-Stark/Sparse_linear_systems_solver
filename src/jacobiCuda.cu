//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include <iostream>
#include "jacobiCuda.h"
#include "kernels.cuh"
#include "utilidades.h"

double *jacobi_CUDA::multiplicacionMV() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(x_d, x.data(), getColumnas() * sizeof(double), cudaMemcpyHostToDevice);

    matrix_vector_multiplication<double><<<grid_size, block_size>>>(A, col_ind, row_ptr, x_d, y_d, getFilas());

    cudaMemcpy(y, y_d, getFilas() * sizeof(double), cudaMemcpyDeviceToHost);

    return y;
}

double jacobi_CUDA::norma() {
    double r_max = utilidades::reduce_max_CUDA(r_d, getFilas(), BLOCK_SIZE);
    double x_max = utilidades::reduce_max_CUDA(x_d, getFilas(), BLOCK_SIZE);
    double norma = r_max / x_max;
//    cout << "r_max: " << r_max;
//    cout << "  x_max: " << x_max << endl;
//    cout << " norma: " << norma << endl;
    return norma;
}

void jacobi_CUDA::obtenerNuevaX() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    grid_size.x = (getFilas() + block_size.x - 1) / block_size.x;
    cudaMemcpy(r_d, r, getFilas() * sizeof(double), cudaMemcpyHostToDevice);
    nuevaX<double><<<grid_size, block_size>>>(x.size(), x_d, r_d);
}

void jacobi_CUDA::actualizaX() {
    cudaMemcpy(x.data(), x_d, getColumnas() * sizeof(double), cudaMemcpyDeviceToHost);
}

jacobi_CUDA::jacobi_CUDA(const CSR &m, const vector<double> &aprox_inicial, const int &block_size_arg)
        : jacobi(m, aprox_inicial),
          BLOCK_SIZE(block_size_arg) {
    cudaMalloc(&r_d, sizeof(double) * getFilas());
    cudaMalloc(&A, sizeof(double) * matriz.getVal().size());
    cudaMalloc(&col_ind, sizeof(int) * matriz.getColInd().size());
    cudaMalloc(&row_ptr, sizeof(int) * matriz.getRowPtr().size());
    cudaMalloc(&x_d, sizeof(double) * getFilas());
    cudaMalloc(&y_d, sizeof(double) * getFilas());
    cudaMalloc(&inversa_d, sizeof(double) * getFilas());

    cudaMemcpy(A, matriz.getVal().data(), matriz.getVal().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.getColInd().data(), matriz.getColInd().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.getRowPtr().data(), matriz.getRowPtr().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(inversa_d, inversa, getFilas() * sizeof(double), cudaMemcpyHostToDevice);
}

jacobi_CUDA::jacobi_CUDA(const CSR &m, const int &block_size_arg) :
        jacobi_CUDA(m, vector<double>(m.getFilas(), 1), block_size_arg) {}

jacobi_CUDA::~jacobi_CUDA() {
    cudaFree(A);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(r_d);
    cudaFree(col_ind);
    cudaFree(row_ptr);
    cudaFree(inversa_d);
};
