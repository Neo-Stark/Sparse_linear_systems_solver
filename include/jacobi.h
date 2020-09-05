//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//
#ifndef JACOBI_H
#define JACOBI_H

#include <CSR.h>
#include <cmath>

using namespace std;

class jacobi {
public:
    explicit jacobi(const CSR &m, const vector<double> &aprox_inicial);

    explicit jacobi(const CSR &m);

    virtual ~jacobi();

    virtual double *multiplicacionMV();

    virtual void obtenerNuevaX();

    virtual void actualizaX();

    virtual double norma();

    void calculaResiduo(const double *b);

    int getFilas();

    int getColumnas();

    const CSR &getMatriz() const { return matriz; }

    double *getInversa() const;

    double getInversa(int i);

    const vector<double> getDiagonal() const;

    double *getY();

    double getY(int i);

    const vector<double> &getX() const;

    double &getX(int i);

    double *getR() const;

    double getR(int i);

protected:
    double *inversaDiagonal();

    double *calculaDiagonal();


    CSR matriz;
    double *inversa;
    double *y;
    double *r;
    vector<double> x, x_kp1;

};

#endif //JACOBI_H