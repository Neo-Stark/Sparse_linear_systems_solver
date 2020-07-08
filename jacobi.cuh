//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//
#ifndef JACOBI_H
#define JACOBI_H

#include <CSR.h>
#include <mv_multiplication.cuh>
#include <cmath>

using namespace std;

class jacobi {
public:
    jacobi(const CSR &m, const int &block_size_arg);

    ~jacobi();

    double *multiplicacionMV_CUDA();

    void obtenerNuevaX();

    double *multiplicacionMV_OMP();

    double norma();

    static void diferencia(const double *b, const double *q, double *r, int n);

    int getFilas();

    int getColumnas();

    const CSR &getMatriz() const { return matriz; }

    double *getInversa() const;

    double getInversa(int i);

    double *getDiagonal() const;

    double *getY();

    double getY(int i);

    const vector<double> &getX() const;

    double &getX(int i);

private:
    double *inversaDiagonal();

    double *calculaDiagonal();

    CSR matriz;
    double *diagonal;
    double *inversa;
    double *y;
    vector<double> x;

    // Punteros a memoria en GPU
    double *A{}, *x_d{}, *y_d{}, *inversa_diag{};
    unsigned int *col_ind{}, *row_ptr{};

    const int BLOCK_SIZE;
};

#endif //JACOBI_H