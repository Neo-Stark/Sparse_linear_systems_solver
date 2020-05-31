//
// Created by fran.
//

#include "jacobi.h"

__global__ void hello(){
    printf("Hello!\n");
}

float jacobi::norma(const float *r) {
    float sum_cuadrados = 0;
    for (size_t i = 0; i < getN(); i++) {
        sum_cuadrados += pow(r[i], 2);
    }
    return sqrt(sum_cuadrados);
}

float *jacobi::calculaDiagonal() {
    float *D = new float[matriz.N];
    int fila = 0;
    int i = 0;
    while (fila < getN()) {
        if (matriz.row_ptr[fila] == matriz.row_ptr[fila + 1]) {
            D[fila] = 0;
            fila++;
        } else if (matriz.col_ind[i] == fila) {
            D[fila] = matriz.val[i];
            fila++;
        } else if (i >= matriz.row_ptr[fila + 1]) {
            D[fila] = 0;
            fila++;
        } else i++;
    }
    return D;
}

int jacobi::getN() {
    return matriz.N;
}

float *jacobi::getDiagonal() const {
    hello<<<1,3>>>();
    cudaDeviceSynchronize();
    return diagonal;
}

void jacobi::inversaDiagonal() {
    for (size_t i = 0; i < getN(); i++) {
        if (diagonal[i] != 0)
            inversa[i] = 1.0f / diagonal[i];
        else
            inversa[i] = 0;
    }
}

float *jacobi::getInversa() const {
    return inversa;
}

void jacobi::diferencia(float *b, float *q, float *r, int n) {
    for (size_t i = 0; i < n; i++) {
        r[i] = b[i] - q[i];
    }
}
