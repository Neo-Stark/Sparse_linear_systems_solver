//
// Created by fran.
//
#ifndef JACOBI_H
#define JACOBI_H

#include <csr_formatter.h>
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

class jacobi {
public:
    jacobi(const CSR &m) {
        if (m.filas != 0)
            this->matriz = m;
        else
            throw std::invalid_argument("la matriz no está inicializada");

        diagonal = calculaDiagonal();
        inversa = new float[getFilas()];
        inversaDiagonal();
    }

    float *getInversa() const;

    float *getDiagonal() const;;

    void multiplicacionMV();

    void inversaDiagonal();

    float *calculaDiagonal();

    float norma(const float *r);

    void diferencia(float *b, float *q, float *r, int n);


    int getFilas();

    int getColumnas();

    const CSR &getMatriz() const {
        return matriz;
    }

    CSR matriz;
private:
    float *diagonal;
    float *inversa;

};

#endif //JACOBI_H