//
// Created by fran.
//

#include "jacobi.cuh"

double jacobi::norma(const double *r) {
    double sum_cuadrados = 0;
    for (size_t i = 0; i < getFilas(); i++) {
        sum_cuadrados += pow(r[i], 2);
    }
    return sqrt(sum_cuadrados);
}

double *jacobi::calculaDiagonal() {
    double *D = new double[matriz.filas];
    int pos = 0;
    int lim = min(getFilas(), getColumnas());
    int i = 0;
    while (pos < lim) {
        if (matriz.row_ptr[pos] == matriz.row_ptr[pos + 1] || i >= matriz.row_ptr[pos + 1]) {
            D[pos] = 0;
            pos++;
        } else if (matriz.col_ind[i] == pos) {
            D[pos] = matriz.val[i];
            while (i < matriz.row_ptr[pos + 1]) i++;
            pos++;
        } else i++;
    }
    return D;
}

int jacobi::getFilas() {
    return matriz.filas;
}

double *jacobi::getDiagonal() const {
    return diagonal;
}

void jacobi::inversaDiagonal() {
    for (size_t i = 0; i < getFilas(); i++) {
        if (diagonal[i] != 0)
            inversa[i] = 1.0f / diagonal[i];
        else
            inversa[i] = 0;
    }
}

double *jacobi::getInversa() const {
    return inversa;
}

void jacobi::diferencia(double *b, double *q, double *r, int n) {
    for (size_t i = 0; i < n; i++) {
        r[i] = b[i] - q[i];
    }
}

int jacobi::getColumnas() {
    return matriz.columnas;
}

double *jacobi::multiplicacionMV() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(x_d, x_h.data(), getColumnas() * sizeof(double), cudaMemcpyHostToDevice);

    csr_spmv_vector_kernel<double><<<grid_size, block_size>>>(getFilas(), col_ind, row_ptr, A, x_d, y_d);

    cudaMemcpy(y_h, y_d, getFilas() * sizeof(double), cudaMemcpyDeviceToHost);

    return y_h;
}

double *jacobi::getYH() {
    return y_h;
}

double &jacobi::getXH(int i) {
    return x_h[i];
}

const vector<double> &jacobi::getXH() const {
    return x_h;
}
