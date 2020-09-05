//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_SRJ_OMP_H
#define SISTEMAS_LINEALES_SRJ_OMP_H

#include "jacobiOmp.h"

class SRJ_OMP : public jacobi_OMP {
    int k = 0;
    const vector<double> srjSch;
public:
    explicit SRJ_OMP(const CSR &m, const vector<double> &_srjSch, const vector<double> &aprox_inicial)
            : jacobi_OMP(m, aprox_inicial),
              srjSch(_srjSch) {};

    explicit SRJ_OMP(const CSR &m, const vector<double> &_srjSch)
            : SRJ_OMP(m, _srjSch, vector<double>(m.getFilas(), 1)) {};

    void obtenerNuevaX() override;
};


#endif //SISTEMAS_LINEALES_SRJ_OMP_H
