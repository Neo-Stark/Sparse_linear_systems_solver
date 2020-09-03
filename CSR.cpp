//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "CSR.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "fstream"

CSR::CSR(istream &fin, bool precondicionar) {
    if (!fin.good()) {
        cout << "error al abrir fichero...Abortando\n";
        abort();
    }
    fin.seekg(0);
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;

    filas = M;
    columnas = N;

    int row, col;
    double data;
    vector<int> col_cord, row_cord;
    vector<double> val_cord;
    for (int l = 0; l < L; l++) {
        fin >> row >> col >> data;
        row_cord.push_back(row - 1);
        col_cord.push_back(col - 1);
        val_cord.push_back(data);
    }

    row_ptr.push_back(0);
    for (int fila = 0; fila < filas; fila++) {
        for (int l = 0; l < L; l++) {
            if (row_cord[l] == fila) {
                col_ind.push_back(col_cord[l]);
                val.push_back(val_cord[l]);
            }
        }
        row_ptr.push_back(col_ind.size());
    }
    calculaDiagonal();
    if (precondicionar) {
        precondicion = true;
        precondicionar_con_diagonal();
    }
//
//    ofstream ofile("matriz-4x4.mtx");
//    ofile << filas << ' ' << columnas << ' ' << L << endl;
//    for (auto columna = 0; columna < columnas; columna++) {
//        for (int l = 0; l < L; l++) {
//            if (col_cord[l] == columna) {
//                ofile << row_cord[l] + 1 << ' ' << col_cord[l] + 1 << ' ' << val_cord[l] << endl;
//            }
//        }
//    }
//    ofile.close();
}

void CSR::printMatrix() {
    int cont = 0;
    cout << "{";
    for (int i = 1; i < row_ptr.size(); i++) {
        cout << "{";
        int row_start = row_ptr[i - 1];
        int row_end = row_ptr[i];
        vector<int>::const_iterator first = col_ind.begin() + row_start;
        vector<int>::const_iterator last = col_ind.begin() + row_end;
        vector<int> row(first, last);
        for (int j = 0; j < row_ptr.size() - 1; j++) {
            if (count(row.begin(), row.end(), j) == 0)
                cout << ' ' << ", ";
            else {
                cout << "#" << ", ";
                cont++;
            }
        }
        std::cout << "\b\b}" << std::endl;
    }
    std::cout << "\b}" << std::endl;
}

int CSR::getDegree(int vertex) {
    return row_ptr[vertex] - row_ptr[vertex - 1];
}

vector<int> CSR::getAdjVertices(int vertex) {
    int row_start = row_ptr[vertex - 1];
    int row_end = row_ptr[vertex];
    vector<int>::const_iterator first = col_ind.begin() + row_start;
    vector<int>::const_iterator last = col_ind.begin() + row_end;
    vector<int> adjVertices(first, last);
    return adjVertices;
}

int CSR::getBandwidth() {
    int bandwidth = std::numeric_limits<int>::min();
    for (int i = 1; i < row_ptr.size() - 1; i++) { // i = current row id
        int row_start = row_ptr[i - 1];
        int row_end = row_ptr[i];
        if (row_end - row_start == 1)
            continue;
        for (int j = row_start; j < row_end; j++) {
            if (abs(col_ind[j] - i) > bandwidth) {
                bandwidth = abs(col_ind[j] - i);
            }

        }
    }
    return bandwidth;
}

CSR::CSR(const CSR &m) {
    this->val = m.val;
    this->col_ind = m.col_ind;
    this->row_ptr = m.row_ptr;
    this->columnas = m.columnas;
    this->filas = m.filas;
}

const vector<double> &CSR::getVal() const {
    return val;
}

const vector<int> &CSR::getColInd() const {
    return col_ind;
}

const vector<int> &CSR::getRowPtr() const {
    return row_ptr;
}

int CSR::getFilas() const {
    return filas;
}

int CSR::getColumnas() const {
    return columnas;
}

bool CSR::isDiagonallyDominant() {
    bool sol = true;
    double d, sum = 0;
    for (int i = 0; sol && i < filas; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_ind[j] == i) d = val[j];
            else sum += val[j];
        }
        if (sum > d) sol = false;
        sum = 0;
    };

    return sol;
}

void CSR::calculaDiagonal() {
    for (int i = 0; i < getFilas(); i++) {
        diagonal.push_back(0.0);
        for (int j = getRowPtr()[i]; j < getRowPtr()[i + 1]; ++j) {
            if (getColInd()[j] == i) diagonal[i] = getVal()[j];
        }
    }
}

const vector<double> &CSR::getDiagonal() const {
    return diagonal;
}

void CSR::precondicionar_con_diagonal() {
    for (auto fila = 0; fila < filas; fila++)
        for (auto i = row_ptr[fila]; i < row_ptr[fila + 1]; i++)
            val[i] = val[i] / getDiagonal()[fila];
}

bool CSR::isPrecondicionada() {
    return precondicion;
}
