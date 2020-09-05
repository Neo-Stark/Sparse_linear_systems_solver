#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "utilidades.h"
#include "algorithm"
#include "kernels.cuh"
#include <sys/time.h>


using namespace std;

double utilidades::reduce_max_OMP(const double *v, int n) {
    double maximo = -1e36;
#pragma omp parallel for reduction (max : maximo)
    for (int i = 0; i < n; i++) {
        maximo = max(maximo, abs(v[i]));
    }
    return maximo;
}

double utilidades::reduce_max_CUDA(const double *d_vi, int n, const int BLOCK_SIZE) {
    dim3 block(BLOCK_SIZE);
    dim3 grid = (n / 2 + block.x) / block.x;
    auto smemSize = block.x * sizeof(double);
    double *d_vo, *h_vo = new double[grid.x];
    cudaMalloc(&d_vo, sizeof(double) * grid.x);
    switch (BLOCK_SIZE) {
        case 1024:
            reduction_max<double, 1024><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 512:
            reduction_max<double, 512><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 256:
            reduction_max<double, 256><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 128:
            reduction_max<double, 128><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 64:
            reduction_max<double, 64><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
        case 32:
            reduction_max<double, 32><<< grid, block, smemSize >>>(d_vi, d_vo, n);
            break;
    }
    cudaMemcpy(h_vo, d_vo, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double maximo = 0.0;
    for (int i = 0; i < grid.x; i++) maximo = max(maximo, h_vo[i]);

    cudaFree(d_vo);
    free(h_vo);
    return maximo;
}

double utilidades::reduce_max_sec(const double *v, int n) {
    double maximo = 0;
    for (int line = 0; line < n; ++line) {
        maximo = (fabs(v[line]) > maximo ? fabs(v[line]) : maximo);
    }
    return maximo;
}

double utilidades::cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}


#pragma clang diagnostic pop