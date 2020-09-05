#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include <iostream>
#include "jacobiOmp.h"
#include "utilidades.h"

double *jacobi_OMP::multiplicacionMV() {
#pragma omp parallel
#pragma omp for
    for (int line = 0; line < matriz.getFilas(); ++line) {
        double sum = 0;
        const unsigned int row_start = matriz.getRowPtr()[line];
        const unsigned int row_end = matriz.getRowPtr()[line + 1];
        for (unsigned int element = row_start; element < row_end; element++) {
            sum += matriz.getVal()[element] * x[matriz.getColInd()[element]];
        }
        y[line] = sum;
    }
    return y;
}

double jacobi_OMP::norma() {
    double r_max = utilidades::reduce_max_OMP(r, getFilas());
    double x_max = utilidades::reduce_max_OMP(x_kp1.data(), getFilas());
    double norma = r_max / x_max;
//    cout << "r_max: " << r_max;
//    cout << "  x_max: " << x_max << endl;
//    cout << " norma: " << norma << endl;
    return norma;
}

void jacobi_OMP::obtenerNuevaX() {
#pragma omp parallel for
    for (int i = 0; i < getFilas(); i++) {
        x_kp1[i] = x[i] + r[i];
    }
}

#pragma clang diagnostic pop