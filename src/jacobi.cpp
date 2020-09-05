//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "utilidades.h"
#include "jacobi.h"
#include <iostream>


jacobi::jacobi(const CSR &m, const vector<double> &aprox_inicial)
        : x(aprox_inicial), x_kp1(m.getFilas()), matriz(m) {
    if (m.getFilas() == 0)
        throw std::invalid_argument("la matriz no está inicializada");

    y = new double[getFilas()];
    r = new double[getFilas()];
    inversa = inversaDiagonal();
}

jacobi::jacobi(const CSR &m) :
        jacobi(m, vector<double>(m.getFilas(), 1)) {}


double jacobi::norma() {
    double residual = utilidades::reduce_max_sec(r, getFilas());
    double max_x_k = utilidades::reduce_max_sec(x_kp1.data(), getFilas());
    double norma = residual / max_x_k;
    return norma;
}

int jacobi::getFilas() {
    return matriz.getFilas();
}

const vector<double> jacobi::getDiagonal() const {
    return matriz.getDiagonal();
}

double *jacobi::inversaDiagonal() {
    auto inver_diag = new double[getFilas()];
    for (int i = 0; i < getFilas(); i++) {
        if (getDiagonal()[i] != 0)
            inver_diag[i] = 1.0 / getDiagonal()[i];
        else
            inver_diag[i] = 0;
    }
    return inver_diag;
}

double *jacobi::getInversa() const {
    return inversa;
}

void jacobi::calculaResiduo(const double *b) {
    for (int i = 0; i < getFilas(); i++) {
        r[i] = (b[i] - y[i]) * getInversa(i);
    }
}

int jacobi::getColumnas() {
    return matriz.getColumnas();
}

double *jacobi::getY() {
    return y;
}

double &jacobi::getX(int i) {
    return x[i];
}

const vector<double> &jacobi::getX() const {
    return x;
}

double jacobi::getY(int i) {
    return y[i];
}

double jacobi::getInversa(int i) {
    return inversa[i];
}

void jacobi::obtenerNuevaX() {
    for (int i = 0; i < getFilas(); i++) {
        x_kp1[i] = x[i] + r[i];
    }
}

void jacobi::actualizaX() {
    x.swap(x_kp1);
}

jacobi::~jacobi() {
    free(y);
    free(r);
    free(inversa);
}

double *jacobi::getR() const {
    return r;
}

double jacobi::getR(int i) {
    return r[i];
}

double *jacobi::multiplicacionMV() {
    for (int line = 0; line < matriz.getFilas(); ++line) {
        double sum = 0;
        const unsigned int row_start = matriz.getRowPtr()[line];
        const unsigned int row_end = matriz.getRowPtr()[line + 1];
        for (unsigned int element = row_start; element < row_end; element++) {
            sum += matriz.getVal()[element] * x[matriz.getColInd()[element]];
        }
        y[line] = sum;
    }
    return y;
}



