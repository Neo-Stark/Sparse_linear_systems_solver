//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_JACOBICUDA_H
#define SISTEMAS_LINEALES_JACOBICUDA_H

#include "jacobi.h"


class jacobi_CUDA : public jacobi {
protected:
    // Punteros a memoria en GPU
    double *A{}, *x_d{}, *y_d{}, *inversa_d{}, *r_d{};
    unsigned int *col_ind{}, *row_ptr{};

    const int BLOCK_SIZE;
public:
    explicit jacobi_CUDA(const CSR &m, const vector<double> &aprox_inicial, const int &block_size_arg = 256);

    explicit jacobi_CUDA(const CSR &m, const int &block_size_arg = 256);

    ~jacobi_CUDA() override;

    double *multiplicacionMV() override;

    double norma() override;

    void obtenerNuevaX() override;

    void actualizaX() override;
};


#endif //SISTEMAS_LINEALES_JACOBICUDA_H
