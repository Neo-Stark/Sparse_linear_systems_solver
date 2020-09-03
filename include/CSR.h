//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_CSR_H
#define SISTEMAS_LINEALES_CSR_H

#include <vector>
#include <istream>

using namespace std;

class CSR {
    vector<double> val;
    vector<int> col_ind;
    vector<int> row_ptr;
    vector<double> diagonal;
    int filas = 0;
    int columnas = 0;
    bool precondicion = false;

    void calculaDiagonal();

    void precondicionar_con_diagonal();
public:
    explicit CSR(istream &fin, bool precondicionar = false);

    CSR(const CSR &m);

    const vector<double> &getDiagonal() const;

    void printMatrix();

    int getDegree(int vertex);

    vector<int> getAdjVertices(int vertex);

    int getBandwidth();

    bool isDiagonallyDominant();

    const vector<double> &getVal() const;

    const vector<int> &getColInd() const;

    const vector<int> &getRowPtr() const;

    int getFilas() const;

    int getColumnas() const;

    bool isPrecondicionada();

};


#endif //SISTEMAS_LINEALES_CSR_H
