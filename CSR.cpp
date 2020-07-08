//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "CSR.h"
#include <iostream>
#include <cmath>
#include <algorithm>

CSR::CSR(istream &fin) {
    int M, N, L;
    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    // Read defining parameters:
    fin >> M >> N >> L;

    filas = M;
    columnas = N;

    int last_row = 0;
    row_ptr.push_back(0);
    for (int l = 0; l < L; l++) {
        int row, col;
        double data;
        fin >> row >> col >> data;
        row = row - 1;
        col_ind.push_back(col - 1);
        val.push_back(data);
        if (row > last_row) {
            last_row = row;
            row_ptr.push_back(col_ind.size() - 1);
        }
    }
    row_ptr.push_back(col_ind.size());
}

void CSR::printMatrix() {
    int cont = 0;
    for (int i = 1; i < row_ptr.size(); i++) {
        int row_start = row_ptr[i - 1];
        int row_end = row_ptr[i];
        vector<int>::const_iterator first = col_ind.begin() + row_start;
        vector<int>::const_iterator last = col_ind.begin() + row_end;
        vector<int> row(first, last);
        for (int j = 0; j < row_ptr.size(); j++) {
            if (count(row.begin(), row.end(), j) == 0)
                cout << '0' << ' ';
            else {
                cout << val[cont] << ' ';
                cont++;
            }
        }
        std::cout << std::endl;
    }
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
