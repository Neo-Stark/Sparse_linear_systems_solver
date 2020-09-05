//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//
#include <iostream>
#include <vector>
#include "matrices_test.h"
#include "tests.h"
#include <omp.h>
#include <CSR.h>
#include "readers.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        cout << "Uso: test [mtx] [rhs] [x0] [srjSch]";
        abort();
    }
    string m = argv[1];
    string rhs = argv[2];
    string x0 = argv[3];
    string srjSch = argv[4];

    cout << "Leyendo matriz...." << endl;
    fstream fs(m);
    CSR matriz(fs);
    auto _b = rhsVector_reader<double>(rhs, matriz.getFilas());

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    int max_threads = omp_get_max_threads();
    const int n = 1;
    vector<vector<double>> ejecuciones(n);

    for (int iter = 0; iter < n; iter++) {
// Tests JACOBI ***************
//        ejecuciones[iter].push_back(jacobi_secuencial(matriz, _b, x0));
//        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
//            ejecuciones[iter].push_back(test_jacobi_CUDA(matriz, _b, x0));
//        }
//        for (int i = 2; i <= max_threads; i = i << 1) {
//            omp_set_num_threads(i);
//            cout << "Nucleos CPU: " << omp_get_max_threads() << endl;
//            ejecuciones[iter].push_back(test_jacobi_OMP(matriz, _b, x0));
//        }
// Tests SRJ *****************
        ejecuciones[iter].push_back(srj_secuencial(matriz, _b, x0, srjSch));
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            ejecuciones[iter].push_back(srj_CUDA(matriz, _b, x0, srjSch));
        }
        for (int i = 2; i <= max_threads; i = i << 1) {
            omp_set_num_threads(i);
            ejecuciones[iter].push_back(srj_OMP(matriz, _b, x0, srjSch));
        }
// Tests SOR ******************
//        double omega = 1.1;
//        ejecuciones[iter].push_back(SOR(matriz, _b, x0, omega));
//        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
//            ejecuciones[iter].push_back(SOR_CUDA(matriz, _b, x0, omega));
//        }
//        for (int i = 2; i <= max_threads; i = i << 1) {
//            omp_set_num_threads(i);
//            ejecuciones[iter].push_back(SOR_OMP(matriz, _b, x0, omega));
//        }
//// Tests Gauss Seidel ******************
//        omega = 1;
//        ejecuciones[iter].push_back(SOR(matriz, _b, x0, omega));
//        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
//            ejecuciones[iter].push_back(SOR_CUDA(matriz, _b, x0, omega));
//        }
//        for (int i = 2; i <= max_threads; i = i << 1) {
//            omp_set_num_threads(i);
//            ejecuciones[iter].push_back(SOR_OMP(matriz, _b, x0, omega));
//        }
    }
    cout << "Tiempos: ";
    vector<double> tiempo_medio(ejecuciones[0].size());
    for (int i = 0; i < tiempo_medio.size(); i++) {
        for (int j = 0; j < n; j++) {
            tiempo_medio[i] += ejecuciones[j][i];
//            cout << ejecuciones[j][i] << " ";
        }
        tiempo_medio[i] /= n;
        cout << tiempo_medio[i] << " ";
    }
}
