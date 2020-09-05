//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_JACOBIOMP_H
#define SISTEMAS_LINEALES_JACOBIOMP_H

#include "jacobi.h"

class jacobi_OMP : public jacobi {
public:

    explicit jacobi_OMP(const CSR &m, const vector<double> &aprox_inicial) : jacobi(m, aprox_inicial) {};

    explicit jacobi_OMP(const CSR &m) : jacobi(m) {};

    double *multiplicacionMV() override;

    double norma() override;

    void obtenerNuevaX() override;
};


#endif //SISTEMAS_LINEALES_JACOBIOMP_H
