#include "utilidades.h"
#include <jacobi.h>
#include <iostream>
#include "matrices_test.h"
#include <readers.h>
#include <SRJ_CUDA.h>
#include <SRJ.h>
#include <SRJ_OMP.h>
#include "tests.h"
#include "jacobiOmp.h"
#include "jacobiCuda.h"


using namespace std;

const double tolerance = 1.e-8;

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

double test_jacobi_CUDA(CSR matriz, double *_b, string _x0) {
    int BLOCKSIZE = 256;
    cout << "Test Jacobi CUDA *****///\n"
         << "BLOCKSIZE: " << BLOCKSIZE << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());

    jacobi_CUDA test(matriz, vector<double>(_x_k, _x_k + matriz.getFilas()), BLOCKSIZE);
    int iter = 0;
    double residual = 1e36;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
//        if (iter % 1000 == 0) cout << "residual: " << residual << endl;
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}
//}

double test_jacobi_OMP(CSR matriz, double *_b, string _x0) {
    cout << "Test Jacobi OMP *****///\n";
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());

    jacobi_OMP test(matriz, vector<double>(_x_k, _x_k + matriz.getFilas()));
    int iter = 0;
    double residual = 1e36;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}


double jacobi_secuencial(CSR matriz, double *_b, string _x0) {
    cout << "Jacobi secuencial*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());

    jacobi test(matriz, vector<double>(_x_k, _x_k + matriz.getFilas()));
    double residual = 1.e36;
    int iter = 0;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}

double srj_CUDA(CSR matriz, double *_b, string _x0, string _srjSch) {
    cout << "SRJ CUDA*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());
    auto srjSch = srjSch_reader<double>(_srjSch);

    SRJ_CUDA test(matriz, srjSch, vector<double>(_x_k, _x_k + matriz.getFilas()));
    double residual = 1.e36;
    int iter = 0;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}

double srj_OMP(CSR matriz, double *_b, string _x0, string _srjSch) {
    cout << "SRJ OMP*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());
    auto srjSch = srjSch_reader<double>(_srjSch);

    SRJ_OMP test(matriz, srjSch, vector<double>(_x_k, _x_k + matriz.getFilas()));
    double residual = 1.e36;
    int iter = 0;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}

double srj_secuencial(CSR matriz, double *_b, string _x0, string _srjSch) {
    cout << "SRJ SECUENCIAL*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());
    auto srjSch = srjSch_reader<double>(_srjSch);

    SRJ test(matriz, srjSch, vector<double>(_x_k, _x_k + matriz.getFilas()));
    double residual = 1.e36;
    int iter = 0;
    double t1 = utilidades::cpuSecond();
    while (residual > tolerance) {
        test.multiplicacionMV();
        test.calculaResiduo(_b);
        test.obtenerNuevaX();
        residual = test.norma();
        test.actualizaX();
        iter++;
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, test.getX().data(), test.getY(), _b, matriz.getFilas());
    return t2;
}


double SOR_OMP(CSR matriz, double *_b, string _x0, double omega) {
    cout << "SOR OMP *****///" << endl;
    cout << "omega: " << omega << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());           // Vector de entrada iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo

    double residual = 1.e36;
    int iter;
    double t1 = utilidades::cpuSecond();
    for (iter = 0; (residual > tolerance); ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        vector<double> x_km1(_x_k, _x_k + matriz.getFilas());
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
            _residual_vec[line] = _x_k[line] - x_km1[line];
        }
        residual = utilidades::reduce_max_OMP(_residual_vec, matriz.getFilas());
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    return t2;
}

double SOR_CUDA(CSR matriz, double *_b, string _x0, double omega) {
    cout << "SOR CUDA *****///" << endl;
    cout << "omega: " << omega << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());           // Vector de entrada iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo
    double *_residual_vec_d;
    cudaMalloc(&_residual_vec_d, matriz.getFilas() * sizeof(double));

    double residual = 1.e36;
    int iter;
    double t1 = utilidades::cpuSecond();
    for (iter = 0; (residual > tolerance); ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        vector<double> x_km1(_x_k, _x_k + matriz.getFilas());
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
            _residual_vec[line] = _x_k[line] - x_km1[line];
        }
        cudaMemcpy(_residual_vec_d, _residual_vec, matriz.getFilas() * sizeof(double), cudaMemcpyHostToDevice);
        residual = utilidades::reduce_max_CUDA(_residual_vec_d, matriz.getFilas(), 256);
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    return t2;
}

double SOR(CSR matriz, double *_b, string _x0, double omega) {
    cout << "SOR secuencial *****///" << endl;
    cout << "omega: " << omega << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());           // Vector de entrada iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo

    double residual = 1.e36;
    int iter;
    double t1 = utilidades::cpuSecond();
    for (iter = 0; (residual > tolerance); ++iter) {
        // Matrix-Vector Product Matrix_2d *x_k and compute residual vector
        vector<double> x_km1(_x_k, _x_k + matriz.getFilas());
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
            _residual_vec[line] = _x_k[line] - x_km1[line];
        }
        residual = utilidades::reduce_max_OMP(_residual_vec, matriz.getFilas());
    }
    double t2 = utilidades::cpuSecond() - t1;
    cout << "iter: " << iter << endl;
//    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    return t2;
}

void SOR_clasico() {
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

void jacobi_secuencial_funcion(CSR matriz, double *_b, string _x0) {
    cout << "Jacobi secuencial*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas());

//    if (matriz.isPrecondicionada())
//        for (int i = 0; i < matriz.getFilas(); i++) _b[i] = _b[i] / matriz.getDiagonal()[i];
    double *_x_kp1 = new double[matriz.getFilas()];         // Vector de salida iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo
    double residual = 1.e36;
    const double tolerance = 1.e-8;

// initial seed
//    double *_b = new double[matriz.getFilas()];
//    for (int i = 0; i < matriz.getFilas(); i++) _b[i] = 1;
//    double *_x_k = new double[matriz.getFilas()];
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
            _x_kp1[line] = _x_k[line] + _residual_vec[line];
        }
        // compute residual
        residual = 0;
        double max_x_k = 0;
        for (int line = 0; line < matriz.getFilas(); ++line) {
            residual = (fabs(_residual_vec[line]) > residual ? fabs(_residual_vec[line]) : residual);
            max_x_k = (fabs(_x_kp1[line]) > max_x_k ? fabs(_x_kp1[line]) : max_x_k);
        }
        residual = residual / max_x_k;
        // transfer _x_kp1 into _x_k
        double *tmp;
        tmp = _x_k;
        _x_k = _x_kp1;
        _x_kp1 = tmp;

    }
    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
}

double srj_secuencial_funcion(CSR matriz, double *_b, string _x0, string _srjSch) {
    cout << "SRJ secuencial*****//" << endl;
    double *_x_k = rhsVector_reader<double>(_x0, matriz.getFilas()); // Vector de entrada iteración
    double *_x_kp1 = new double[matriz.getFilas()];         // Vector de salida iteración
    double *_residual_vec = new double[matriz.getFilas()];  // Vector de residuo
    auto srjSch = srjSch_reader<double>(_srjSch);
    double residual = 1.e36;
    const double tolerance = 1.e-8;
    int iter;

    double t1 = utilidades::cpuSecond();
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
        residual = residual / max_x_k;
        // transfer _x_kp1 into _x_k
        double *tmp;
        tmp = _x_k;
        _x_k = _x_kp1;
        _x_kp1 = tmp;

    }
    double t2 = utilidades::cpuSecond() - t1;
//    printResult(iter, _x_k, nullptr, _b, matriz.getFilas());
    return t2;
}

