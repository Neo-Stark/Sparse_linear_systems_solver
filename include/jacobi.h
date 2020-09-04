//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//
#ifndef JACOBI_H
#define JACOBI_H

#include <CSR.h>
#include <cmath>

using namespace std;

class jacobi {
public:
    explicit jacobi(const CSR &m, const vector<double> &aprox_inicial, const int &block_size_arg = 256);

    explicit jacobi(const CSR &m, const int &block_size_arg = 256);

    ~jacobi();

    double *multiplicacionMV_CUDA();

    double *multiplicacionMV_OMP();

    virtual void obtenerNuevaX();

    void actualizaX();

    double norma();

    double norma_CUDA();

    void calculaResiduo(const double *b);

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

    double *getR() const;

    double getR(int i);

    double reduce_max_CUDA(const double *d_vi, int n) const;

    double reduce_max_OMP(const double *v, int n);

    double norma_OMP();

protected:
    double *inversaDiagonal();

    double *calculaDiagonal();


    CSR matriz;
    double *diagonal;
    double *inversa;
    double *y;
    double *r;
    vector<double> x;

    // Punteros a memoria en GPU
    double *A{}, *x_d{}, *y_d{}, *inversa_d{}, *r_d{};
    unsigned int *col_ind{}, *row_ptr{};

    const int BLOCK_SIZE;
};

#endif //JACOBI_H