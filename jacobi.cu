//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "jacobi.h"
#include <kernels.cuh>
#include <omp.h>


jacobi::jacobi(const CSR &m, const int &block_size_arg) : x(m.getColumnas(), 1),
                                                          BLOCK_SIZE(block_size_arg),
                                                          matriz(m) {
    if (m.getFilas() == 0)
        throw std::invalid_argument("la matriz no está inicializada");

    y = new double[getFilas()];
    diagonal = calculaDiagonal();
    inversa = inversaDiagonal();
    cudaMalloc(&A, sizeof(double) * matriz.getVal().size());
    cudaMalloc(&col_ind, sizeof(double) * matriz.getColInd().size());
    cudaMalloc(&row_ptr, sizeof(double) * matriz.getRowPtr().size());
    cudaMalloc(&x_d, sizeof(double) * getColumnas());
    cudaMalloc(&y_d, sizeof(double) * getFilas());
    cudaMalloc(&y_d, sizeof(double) * getFilas());
    cudaMalloc(&inversa_diag, sizeof(double) * getFilas());

    cudaMemcpy(A, matriz.getVal().data(), matriz.getVal().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.getColInd().data(), matriz.getColInd().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.getRowPtr().data(), matriz.getRowPtr().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(inversa_diag, inversa, getFilas() * sizeof(double), cudaMemcpyHostToDevice);
}

double jacobi::norma() {
    double sum_cuadrados = 0;
    for (size_t i = 0; i < getFilas(); i++) {
        sum_cuadrados += pow(y[i], 2);
    }
    return sqrt(sum_cuadrados);
}

double *jacobi::calculaDiagonal() {
    auto *D = new double[matriz.getFilas()];
    int pos = 0;
    int lim = min(getFilas(), getColumnas());
    int i = 0;
    while (pos < lim) {
        if (matriz.getRowPtr()[pos] == matriz.getRowPtr()[pos + 1] || i >= matriz.getRowPtr()[pos + 1]) {
            D[pos] = 0;
            pos++;
        } else if (matriz.getColInd()[i] == pos) {
            D[pos] = matriz.getVal()[i];
            while (i < matriz.getRowPtr()[pos + 1]) i++;
            pos++;
        } else i++;
    }
    return D;
}

int jacobi::getFilas() {
    return matriz.getFilas();
}

double *jacobi::getDiagonal() const {
    return diagonal;
}

double *jacobi::inversaDiagonal() {
    auto inver_diag = new double[getFilas()];
    for (size_t i = 0; i < getFilas(); i++) {
        if (diagonal[i] != 0)
            inver_diag[i] = 1.0 / diagonal[i];
        else
            inver_diag[i] = 0;
    }
    return inver_diag;
}

double *jacobi::getInversa() const {
    return inversa;
}

void jacobi::diferencia(const double *b, const double *q, double *r, int n) {
    for (size_t i = 0; i < n; i++) {
        r[i] = b[i] - q[i];
    }
}

int jacobi::getColumnas() {
    return matriz.getColumnas();
}

double *jacobi::multiplicacionMV_CUDA() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(x_d, x.data(), getColumnas() * sizeof(double), cudaMemcpyHostToDevice);

    csr_spmv_vector_kernel<double><<<grid_size, block_size>>>(getFilas(), col_ind, row_ptr, A, x_d, y_d);

    cudaMemcpy(y, y_d, getFilas() * sizeof(double), cudaMemcpyDeviceToHost);

    return y;
}

double *jacobi::getY() {
    return y;
}

double &jacobi::getX(int i) {
    return x[i];
}

const vector<double> &jacobi::getX() const {
    return x;
}

double jacobi::getY(int i) {
    return y[i];
}

double jacobi::getInversa(int i) {
    return inversa[i];
}

double *jacobi::multiplicacionMV_OMP() {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < getFilas(); i++) {
            const unsigned int row_start = matriz.getRowPtr()[i];
            const unsigned int row_end = matriz.getRowPtr()[i + 1];
            y[i] = 0;
            for (auto j = row_start; j < row_end; j++) {
                y[i] += matriz.getVal()[j] * x[matriz.getColInd()[j]];
            }
        }
    };
    return y;
}

void jacobi::obtenerNuevaX() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    grid_size.x = (getColumnas() + block_size.x - 1) / block_size.x;
    cudaMemcpy(y_d, y, getFilas() * sizeof(double), cudaMemcpyHostToDevice);

    kernelNuevaX<double><<<grid_size, block_size>>>(x.size(), x_d, y_d, inversa_diag);

    cudaMemcpy(x.data(), x_d, getColumnas() * sizeof(double), cudaMemcpyDeviceToHost);
}

jacobi::~jacobi() {
    free(y);
    free(diagonal);
    free(inversa);
    cudaFree(A);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(col_ind);
    cudaFree(row_ptr);
    cudaFree(inversa_diag);
}
