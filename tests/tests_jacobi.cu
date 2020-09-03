#include <jacobi.h>
#include <iostream>
#include "matrices_test.h"
#include <readers.h>
#include "tests.h"


using namespace std;
#define BLOCKSIZE 32

void printResult(int k, const double *_x, const double *_y, const double *_b, int n) {
    if (n > 1000) n = 100;
    cout << "Vector x: ";
    for (auto i = 0; i < n; i++) cout << _x[i] << " ";
    cout << endl;
    if (_y != nullptr) {
        cout << "Vector y: ";
        for (auto i = 0; i < n; i++) cout << _y[i] << " ";
        cout << endl;
    }
    cout << "Vector b: ";
    for (auto i = 0; i < n; i++) cout << _b[i] << " ";
    cout << endl;
    cout << "Iteraciones K: " << k << endl << endl;
}

void test_jacobi_CUDA() {
    cout << "Test Jacobi CUDA *****///\n"
         << "BLOCKSIZE: " << BLOCKSIZE << endl;
    fstream fs("../data/matrix-193x65.mtx");
    CSR matriz(fs);
    cout << "Es diagonal dominante: " << (matriz.isDiagonallyDominant() ? "true" : "false") << endl;
    auto _b = rhsVector_reader<double>("../data/rhs-193x65.dat", matriz.getFilas());
    double *_x_k = rhsVector_reader<double>("../x0/x0-193x65.dat", matriz.getFilas());

//    double *_b = new double[matriz.getFilas()];
//    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 1;
//    CSR matriz(m._4);
//    double *_b = b._4_5;
    for (int i = 2; i < 64; i = i << 1) {
        cout << "BLOCK SIZE: " << BLOCKSIZE * i << endl;
        jacobi test(matriz, vector<double>(_x_k, _x_k + matriz.getFilas()), BLOCKSIZE * i);
        int iter = 0;
        double residual = 1e36;
        while (residual > 1e-8) {
            test.multiplicacionMV_CUDA();
            test.calculaResiduo(_b);
            test.obtenerNuevaX();
            residual = test.norma_CUDA();
            test.actualizaX();
            if (iter % 1000 == 0) cout << "residual: " << residual << endl;
            iter++;
        }
        cout << "iteraciones: " << iter << endl;
        printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    }
}

void test_jacobi_OMP() {
    cout << "Test Jacobi CUDA *****///\n"
         << "BLOCKSIZE: " << BLOCKSIZE << endl;
    fstream fs("../data/matrix-65x65.mtx");
    CSR matriz(fs);
    cout << "Es diagonal dominante: " << (matriz.isDiagonallyDominant() ? "true" : "false") << endl;
    auto _b = rhsVector_reader<double>("../data/rhs-65x65.dat", matriz.getFilas());
    double *_x_k = rhsVector_reader<double>("../x0/x0-65x65.dat", matriz.getFilas());

//    double *_b = new double[matriz.getFilas()];
//    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 1;
//    CSR matriz(m._4);
//    double *_b = b._4_5;
    jacobi test(matriz, vector<double>(_x_k, _x_k + matriz.getFilas()), BLOCKSIZE);
    int iter = 0;
    double residual = 1e36;
    while (residual > 1e-8) {
        test.multiplicacionMV_OMP();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma_OMP();
        test.actualizaX();
        if (iter % 1000 == 0) cout << "residual: " << residual << endl;
        iter++;
    }
    cout << "iteraciones: " << iter << endl;
    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
}


void jacobi_secuencial() {
    cout << "Jacobi secuencial*****//" << endl;
    fstream fs("../data/matriz.mtx");
    ofstream res("../residuo-jacobi-secuencial-pores_1.dat");
    CSR matriz(fs);

    cout << "Es diagonal dominante: " << (matriz.isDiagonallyDominant() ? "true" : "false") << endl;
//    auto _b = rhsVector_reader<double>("../data/rhs-65x65.dat", matriz.getFilas());
//    double *_x_k = rhsVector_reader<double>("../x0/x0-65x65.dat", matriz.getFilas());

//    CSR matriz(m._4, false);
//    auto _b = b._4;
//    if (matriz.isPrecondicionada())
//        for (int i = 0; i < matriz.getFilas(); i++) _b[i] = _b[i] / matriz.getDiagonal()[i];
    double *_x_kp1 = new double[matriz.getFilas()];         // Vector de salida iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo
    double residual = 1.e36;
    const double tolerance = 1.e-8;

// initial seed
    double *_b = new double[matriz.getFilas()];
    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 1;
    double *_x_k = new double[matriz.getFilas()];
    for (int i = 0; i < matriz.getFilas(); i++) _x_k[i] = 0;
    int iter;

    for (iter = 0; residual > tolerance; ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        for (int line = 0; line < matriz.getFilas(); ++line) {
            double sum = 0;
            const unsigned int row_start = matriz.getRowPtr()[line];
            const unsigned int row_end = matriz.getRowPtr()[line + 1];
            for (unsigned int element = row_start; element < row_end; element++) {
                sum += matriz.getVal()[element] * _x_k[matriz.getColInd()[element]];
            }
            _residual_vec[line] = (_b[line] - sum) / matriz.getDiagonal()[line];
            // vector for residual
            _x_kp1[line] = _x_k[line] + _residual_vec[line];
        }
        // compute residual
        residual = 0;
        double max_x_k = 0;
        for (int line = 0; line < matriz.getFilas(); ++line) {
            residual = (fabs(_residual_vec[line]) > residual ? fabs(_residual_vec[line]) : residual);
            max_x_k = (fabs(_x_kp1[line]) > max_x_k ? fabs(_x_kp1[line]) : max_x_k);
        }
        res << "residual: " << residual << "  max_x: " << max_x_k << "\n";
        residual = residual / max_x_k;
        // transfer _x_kp1 into _x_k
        double *tmp;
        tmp = _x_k;
        _x_k = _x_kp1;
        _x_kp1 = tmp;

    }
    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    res.close();
}

void srj_secuencial() {
    cout << "SRJ secuencial*****//" << endl;
    fstream fs("../data/matrix-129x65.mtx");
    ofstream res("../residuo-srj-secuencial-129x65.dat");
    CSR matriz(fs);
    auto _b = rhsVector_reader<double>("../data/rhs-129x65.dat", matriz.getFilas());
//    double *_b = new double[matriz.getFilas()];
//    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 1;
//    CSR matriz(m._4);
//    auto _b = b._4;
    double *_x_k = rhsVector_reader<double>("../x0/x0-129x65.dat", matriz.getFilas()); // Vector de entrada iteración
    double *_x_kp1 = new double[matriz.getFilas()];         // Vector de salida iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo
    auto srjSch = srjSch_reader<double>("../srjSch/P9_32.srj");
    double residual = 1.e36;
    const double tolerance = 1.e-8;

// initial seed
//    for (int i = 0; i < matriz.getFilas(); i++) _x_k[i] = 0;
    int iter;

    for (iter = 0; residual > tolerance; ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        for (int line = 0; line < matriz.getFilas(); ++line) {
            double sum = 0;
            const unsigned int row_start = matriz.getRowPtr()[line];
            const unsigned int row_end = matriz.getRowPtr()[line + 1];
            for (unsigned int element = row_start; element < row_end; element++) {
                sum += matriz.getVal()[element] * _x_k[matriz.getColInd()[element]];
            }
            _residual_vec[line] = (_b[line] - sum) / matriz.getDiagonal()[line];
            // vector for residual
            _x_kp1[line] = _x_k[line] + _residual_vec[line] * srjSch[iter % srjSch.size()];
        }
        // compute residual
        residual = 0;
        double max_x_k = 0;
        for (int line = 0; line < matriz.getFilas(); ++line) {
            residual = (fabs(_residual_vec[line]) > residual ? fabs(_residual_vec[line])
                                                             : residual);
            max_x_k = (fabs(_x_kp1[line]) > max_x_k ? fabs(_x_kp1[line]) : max_x_k);
        }
        res << "residual: " << residual << "  max_x: " << max_x_k << "\n";
        residual = residual / max_x_k;
        if (iter % 1000 == 0) cout << "iter: " << iter << " residuo " << residual << endl;
        // transfer _x_kp1 into _x_k
        double *tmp;
        tmp = _x_k;
        _x_k = _x_kp1;
        _x_kp1 = tmp;

    }
    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    res.close();
}

void SOR_CSR() {
    cout << "SOR CSR *****///" << endl;
    fstream fs("../data/matrix-65x65.mtx");
//    CSR matriz(fs);
//    cout << "Es diagonal dominante: " << (matriz.isDiagonallyDominant() ? "true" : "false") << endl;
//    auto _b = rhsVector_reader<double>("../data/rhs-65x65.dat", matriz.getFilas());
//        double *_b = new double[matriz.getFilas()];
//    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 0;
    CSR matriz(m._3);
    matriz.printMatrix();
    double *_b = b._3_5;
    double *_x_k = new double[matriz.getFilas()];           // Vector de entrada iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo

    double omega = 0.1;
    const double tolerance = 1.e-8;

    int iter;
    for (; omega < 1.4; omega += 0.05) {
        double residual = 1.e36;
        for (int i = 0; i < matriz.getFilas(); i++) _x_k[i] = 0;

        for (iter = 0; (residual > tolerance); ++iter) {
            // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
            for (int line = 0; line < matriz.getFilas(); ++line) {
                double sum = 0;
                const unsigned int row_start = matriz.getRowPtr()[line];
                const unsigned int row_end = matriz.getRowPtr()[line + 1];
                for (unsigned int element = row_start; element < row_end; element++) {
                    if (line != matriz.getColInd()[element])
                        sum += matriz.getVal()[element] * _x_k[matriz.getColInd()[element]];
                }
                _x_k[line] = (1 - omega) * _x_k[line] +
                             (omega * (_b[line] - sum) / matriz.getDiagonal()[line]);
            }

            for (int line = 0; line < matriz.getFilas(); ++line) {
                double sum = 0;
                const unsigned int row_start = matriz.getRowPtr()[line];
                const unsigned int row_end = matriz.getRowPtr()[line + 1];
                for (unsigned int element = row_start; element < row_end; element++) {
                    sum += matriz.getVal()[element] * _x_k[matriz.getColInd()[element]];
                }
                _residual_vec[line] = sum - _b[line];
            }
            double sum_cuadrados = 0;
            for (size_t i = 0; i < matriz.getFilas(); i++) {
                sum_cuadrados += pow(_residual_vec[i], 2);
            }
            residual = sqrt(sum_cuadrados);
        }
        cout << "Omega: " << omega << " Iter: " << iter << endl;
    }
    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
}

void SOR() {
    cout << "SOR *****///" << endl;
    const int filas = 4;
    auto matriz = m.m2;
    double *_b = b._2;
    double *_x_k = new double[filas];           // Vector de entrada iteración
    double *_residual_vec = new double[filas];  // Vector de residuo

    const double omega = 0.55;
    double residual = 1.e36;
    const double tolerance = 1.e-8;

    for (int i = 0; i < filas; i++) _x_k[i] = 0;

    int iter;

    for (iter = 0; (residual > tolerance); ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        for (int i = 0; i < filas; ++i) {
            double sigma = 0;
            for (int j = 0; j < filas; j++) {
                if (i != j)
                    sigma += matriz[i][j] * _x_k[j];
            }
            _x_k[i] = (1 - omega) * _x_k[i] + (omega * 1.0 / matriz[i][i]) * (_b[i] - sigma);
        }

        // Calculo residuo
        for (int i = 0; i < filas; ++i) {
            double sum = 0;
            for (unsigned int j = 0; j < filas; j++) {
                sum += matriz[i][j] * _x_k[j];
            }
            _residual_vec[i] = sum - _b[i];
        }
        double sum_cuadrados = 0;
        for (size_t i = 0; i < filas; i++) {
            sum_cuadrados += pow(_residual_vec[i], 2);
        }
        residual = sqrt(sum_cuadrados);

    }
    printResult(iter, _x_k, nullptr, _b, filas);
}

void jacobi_clasico() {
    cout << "Jacobi clasico *****///" << endl;
    const int filas = 8;
    auto matriz = m.m4;
    double *_b = b._4;
    double *_x_k = new double[filas];           // Vector de entrada iteración
    double *_x_kp1 = new double[filas];           // Vector de entrada iteración
    double *_residual_vec = new double[filas];  // Vector de residuo

    double residual = 1.e36;
    const double tolerance = 1.e-8;

    for (int i = 0; i < filas; i++) _x_k[i] = 1;

    int iter;

    for (iter = 0; residual > tolerance; iter++) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        for (int i = 0; i < filas; ++i) {
            double sigma = 0;
            for (int j = 0; j < filas; j++) {
                if (i != j)
                    sigma += matriz[i][j] * _x_k[j];
            }
            _x_kp1[i] = (_b[i] - sigma) / matriz[i][i];
        }

        // Calculo residuo
        for (int i = 0; i < filas; ++i) {
            double sum = 0;
            for (unsigned int j = 0; j < filas; j++) {
                sum += matriz[i][j] * _x_kp1[j];
            }
            _residual_vec[i] = sum - _b[i];
        }
        double sum_cuadrados = 0;
        for (size_t i = 0; i < filas; i++) {
            sum_cuadrados += pow(_residual_vec[i], 2);
        }
        residual = sqrt(sum_cuadrados);
        if (residual < tolerance) break;

        double *tmp = _x_k;
        _x_k = _x_kp1;
        _x_kp1 = tmp;
    }
    printResult(iter, _x_k, nullptr, _b, filas);
}