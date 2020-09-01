//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "matrices_test.h"
#include <CSR.h>
#include <iostream>
#include <jacobi.h>

void test_multiplacion_omp() {
    CSR matriz(m._2);
    jacobi test(matriz, {3, -2, 2, 1});

    auto y = test.multiplicacionMV_OMP();
    cout << "y(): ";
    for (int i = 0; i < test.getFilas(); i++) cout << y[i] << " ";
    cout << endl;
}
