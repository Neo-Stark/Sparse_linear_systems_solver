//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_SRJ_H
#define SISTEMAS_LINEALES_SRJ_H
#include "jacobi.h"

class SRJ : public jacobi {
    int k = 0;
    const vector<double> srjSch;
public:
    explicit SRJ(const CSR &m, const vector<double> &_srjSch, const vector<double> &aprox_inicial)
            : jacobi(m, aprox_inicial),
              srjSch(_srjSch) {};

    explicit SRJ(const CSR &m, const vector<double> &_srjSch)
            : SRJ(m, _srjSch, vector<double>(m.getFilas(), 1)) {};

    void obtenerNuevaX() override;

};


#endif //SISTEMAS_LINEALES_SRJ_H
