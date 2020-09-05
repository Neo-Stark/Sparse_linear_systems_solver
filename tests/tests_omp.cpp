//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "matrices_test.h"
#include <CSR.h>
#include <iostream>
#include <jacobi.h>
#include <jacobiOmp.h>
#include "omp.h"

void test_multiplacion_omp() {
    CSR matriz(m._2);
    jacobi_OMP test(matriz, {3, -2, 2, 1});

    auto y = test.multiplicacionMV();
    cout << "y(): ";
    for (int i = 0; i < test.getFilas(); i++) cout << y[i] << " ";
    cout << endl;
}

void reduction_omp() {
    vector<int> v(12);
    for (int i = 0; i < v.size(); i++) v[i] = i;
    int maximo = 0;
#pragma omp parallel
    {
#pragma omp for reduction (max : maximo)
        for (int i = 0; i < v.size(); i++) {
            maximo = max(maximo, v[i]);
        }
#pragma omp master
        cout << "maximo : " << maximo << " hebra: " << omp_get_thread_num() << endl;
    }
}
