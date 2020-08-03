//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "jacobi.h"
#include <kernels.cuh>
#include <iostream>
#include <omp.h>


jacobi::jacobi(const CSR &m, const vector<double> &aprox_inicial, const int &block_size_arg)
        : x(aprox_inicial), BLOCK_SIZE(block_size_arg), matriz(m) {
    if (m.getFilas() == 0)
        throw std::invalid_argument("la matriz no está inicializada");

    y = new double[getFilas()];
    r = new double[getFilas()];
    diagonal = calculaDiagonal();
    inversa = inversaDiagonal();
    cudaMalloc(&r_d, sizeof(double) * getFilas());
    cudaMalloc(&A, sizeof(double) * matriz.getVal().size());
    cudaMalloc(&col_ind, sizeof(int) * matriz.getColInd().size());
    cudaMalloc(&row_ptr, sizeof(int) * matriz.getRowPtr().size());
    cudaMalloc(&x_d, sizeof(double) * getFilas());
    cudaMalloc(&y_d, sizeof(double) * getFilas());
    cudaMalloc(&inversa_diag, sizeof(double) * getFilas());

    cudaMemcpy(A, matriz.getVal().data(), matriz.getVal().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.getColInd().data(), matriz.getColInd().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.getRowPtr().data(), matriz.getRowPtr().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(inversa_diag, inversa, getFilas() * sizeof(double), cudaMemcpyHostToDevice);
}

jacobi::jacobi(const CSR &m, const int &block_size_arg) :
        jacobi(m, vector<double>(m.getFilas(), 1), block_size_arg) {}


double jacobi::norma() {
    double sum_cuadrados = 0;
    for (size_t i = 0; i < getFilas(); i++) {
        sum_cuadrados += pow(r[i], 2);
    }
    auto raiz = sqrt(sum_cuadrados);
    return raiz;
}

double *jacobi::calculaDiagonal() {
    auto *D = new double[matriz.getFilas()];
    for (int i = 0; i < getFilas(); i++) {
        D[i] = 0;
        for (int j = matriz.getRowPtr()[i]; j < matriz.getRowPtr()[i + 1]; ++j) {
            if (matriz.getColInd()[j] == i) D[i] = matriz.getVal()[j];
        }
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
    for (int i = 0; i < getFilas(); i++) {
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

void jacobi::calculaResiduo(const double *b) {
    for (int i = 0; i < getFilas(); i++) {
        r[i] = b[i] - y[i];
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
#pragma omp parallel num_threads(4)
    cout << "multiplicando - Hebra " << omp_get_thread_num() << endl;
#pragma omp for
    for (int i = 0; i < getFilas(); i++) {
        const unsigned int row_start = matriz.getRowPtr()[i];
        const unsigned int row_end = matriz.getRowPtr()[i + 1];
        y[i] = 0;
        for (auto j = row_start; j < row_end; j++) {
            y[i] += matriz.getVal()[j] * x[matriz.getColInd()[j]];
        }
    }
    return y;
}

void jacobi::obtenerNuevaX() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    grid_size.x = (getFilas() + block_size.x - 1) / block_size.x;
//    cout << "r: ";
//    for(int i = 0; i < getFilas(); i++) cout << r[i] << " ";
//    cout << endl;
    cudaMemcpy(r_d, r, getFilas() * sizeof(double), cudaMemcpyHostToDevice);

    kernelNuevaX<double><<<grid_size, block_size>>>(x.size(), x_d, r_d, inversa_diag);
}

void jacobi::actualizaX() {
    cudaMemcpy(x.data(), x_d, getColumnas() * sizeof(double), cudaMemcpyDeviceToHost);
}

double jacobi::normaInfinito_r() {
    double r_max = reduce_max_CUDA(r_d, getFilas());
    cout << "r_max: " << r_max << endl;
    double x_max = reduce_max_CUDA(x_d, getFilas());
    cout << "x_max: " << x_max << endl;
    double norma = r_max / x_max;
    cout << " norma: " << norma << endl;
    return norma;
}

jacobi::~jacobi() {
    free(y);
    free(r);
    free(diagonal);
    free(inversa);
    cudaFree(A);
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(r_d);
    cudaFree(col_ind);
    cudaFree(row_ptr);
    cudaFree(inversa_diag);
}

double jacobi::reduce_max_CUDA(const double *d_vi, const int n) const {
    dim3 block(BLOCK_SIZE);
    dim3 grid = (n / 2 + block.x - 1) / block.x;
    auto smemSize = block.x * sizeof(double);
    double *d_vo, *h_vo = new double[grid.x];
    cudaMalloc(&d_vo, sizeof(double) * grid.x);
    switch (BLOCK_SIZE) {
        case 1024:
            reduce_max<double, 1024><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 512:
            reduce_max<double, 512><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 256:
            reduce_max<double, 256><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 128:
            reduce_max<double, 128><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 64:
            reduce_max<double, 64><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 32:
            reduce_max<double, 32><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
    }
    cudaMemcpy(h_vo, d_vo, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double maximo = 0.0;
    for (int i = 0; i < grid.x; i++) maximo = max(maximo, h_vo[i]);

    cudaFree(d_vo);
    free(h_vo);
    return maximo;
}

double *jacobi::getR() const {
    return r;
}

double jacobi::getR(int i) {
    return r[i];
}



