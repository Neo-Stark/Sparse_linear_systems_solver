//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "SRJ_OMP.h"

void SRJ_OMP::obtenerNuevaX() {
#pragma omp parallel for
    for (int i = 0; i < getFilas(); i++) {
        x_kp1[i] = x[i] + r[i] * srjSch[k % srjSch.size()];
    }
    k++;
}
