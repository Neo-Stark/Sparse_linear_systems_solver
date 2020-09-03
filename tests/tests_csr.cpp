//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include <jacobi.h>
#include <iostream>
#include "matrices_test.h"


using namespace std;
#define BLOCKSIZE 512

void test_diagonal() {
    CSR matriz(m._1);
    jacobi test(matriz, BLOCKSIZE);
    cout << "Diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getDiagonal()[i] << " ";
    cout << endl;
    cout << "Inversa diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getInversa(i) << " ";
    cout << endl;
    cout << "Inversa diagonal x Diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getInversa(i) * test.getDiagonal()[i] << " ";
    cout << endl;
}

void ordenar_mtx(const string &nombre) {
    ofstream of(nombre + "_copia", ios::trunc);
    ifstream fin(nombre);
    string banner;
    while (fin.peek() == '%') {
        getline(fin, banner);
        of << banner << endl;
    }
    int M, N, L;
    fin >> M >> N >> L;
    of << M << " " << N << " " << L << endl;
    for (int l = 0; l < L; l++) {
        int row, col;
        double data;
        fin >> row >> col >> data;
        of << col << " " << row << " " << data << endl;
    }
    of.close();
    fin.close();
}


void test_inversa() {
    CSR matriz(m._2);
    jacobi test(matriz);

    for (int i = 0; i < test.getFilas(); i++)
        if (test.getInversa(i) != 1 / test.getDiagonal()[i])
            cout << "error en i = " << i << endl;

    cout << "Inversa calculada correctamente" << endl;
}

void test_constructor_csr() {
    ifstream file("../data/cage5.mtx");
    CSR matriz(file);
    matriz.printMatrix();

    cout << "row_ptr: ";
    for (auto row_ptr : matriz.getRowPtr()) cout << row_ptr << " ";
    cout << endl;
    cout << "col_ind: ";
    for (auto col_ind : matriz.getColInd()) cout << col_ind << " ";
    cout << endl;
    cout << "val: ";
    for (auto val : matriz.getVal()) cout << val << " ";
    cout << endl;
}