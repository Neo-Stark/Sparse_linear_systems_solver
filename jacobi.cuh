//
// Created by fran.
//
#ifndef JACOBI_H
#define JACOBI_H

#include <csr_formatter.h>
#include <reduce.cuh>
#include <cmath>

using namespace std;

template<typename data_type>
__global__ void csr_spmv_vector_kernel(
        unsigned int n_rows,
        const unsigned int *col_ids,
        const unsigned int *row_ptr,
        const data_type *data,
        const data_type *x,
        data_type *y);


template<typename data_type>
__global__ void csr_spmv_vector_kernel(
        unsigned int n_rows,
        const unsigned int *col_ids,
        const unsigned int *row_ptr,
        const data_type *data,
        const data_type *x,
        data_type *y) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = thread_id / 32;
    const unsigned int lane = thread_id % 32;

//    DEBUG //////////////
//    if(thread_id == 0){
//        printf("Vector datos: ");
//        for (int i = 0; i < row_ptr[n_rows]; i++) printf("%f ", data[i]);
//        printf("\n");
//        printf("Vector filas: ");
//        for (int i = 0; i < n_rows+1; i++) printf("%d ", row_ptr[i]);
//        printf("\n");
//        printf("Vector columnas: ");
//        for (int i = 0; i < row_ptr[n_rows]; i++) printf("%d ", col_ids[i]);
//        printf("\n");
//    }

    const unsigned int row = warp_id; ///< un warp por fila

    data_type sum = 0;
    if (row < n_rows) {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + lane; element < row_end; element += 32)
            sum += data[element] * x[col_ids[element]];
    }

    sum = warp_reduce(sum);

    if (lane == 0 && row < n_rows) {
        y[row] = sum;
    }
}


class jacobi {
public:
    jacobi(const CSR &m, const int &block_size_arg) : x_h(m.filas,1), BLOCK_SIZE(block_size_arg){
        if (m.filas == 0)
            throw std::invalid_argument("la matriz no está inicializada");

        matriz = m;
        y_h = new double[getFilas()];
        diagonal = calculaDiagonal();
        inversa = new double[getFilas()];
        inversaDiagonal();
        cudaMalloc(&A, sizeof(double) * matriz.val.size());
        cudaMalloc(&col_ind, sizeof(double) * matriz.col_ind.size());
        cudaMalloc(&row_ptr, sizeof(double) * matriz.row_ptr.size());
        cudaMalloc(&x_d, sizeof(double) * getColumnas());
        cudaMalloc(&y_d, sizeof(double) * getFilas());

        cudaMemcpy(A, matriz.val.data(), matriz.val.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(col_ind, matriz.col_ind.data(), matriz.col_ind.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(row_ptr, matriz.row_ptr.data(), matriz.row_ptr.size() * sizeof(double), cudaMemcpyHostToDevice);
    }


    ~jacobi() {
        free(y_h);
        free(diagonal);
        free(inversa);
        cudaFree(A);
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(col_ind);
        cudaFree(row_ptr);
    }

    double *getInversa() const;

    double *getDiagonal() const;

    double *multiplicacionMV();

    void inversaDiagonal();

    double *calculaDiagonal();

    double norma(const double *r);

    void diferencia(double *b, double *q, double *r, int n);

    int getFilas();

    int getColumnas();

    const CSR &getMatriz() const { return matriz; }

    CSR matriz;

    double *getYH();

    double &getXH(int i);

    const vector<double> &getXH() const;

private:
    double *diagonal;
    double *inversa;
    double *y_h;
    vector<double> x_h;

    double *A, *x_d, *y_d;
    unsigned int *col_ind, *row_ptr;

    const int BLOCK_SIZE;
};

#endif //JACOBI_H