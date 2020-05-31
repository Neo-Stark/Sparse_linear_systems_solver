//
// Created by fran.
//
#ifndef JACOBI_H
#define JACOBI_H
#include <csr_formatter.h>
#include <cmath>

using namespace std;

class jacobi {
public:
    jacobi(const CSR &m) {
        if (m.N != 0)
            this->matriz = m;
        else
            throw std::invalid_argument("la matriz no está inicializada");

        diagonal = calculaDiagonal();
        inversa = new float[getN()];
        inversaDiagonal();
    }

    float *getInversa() const;

    float *getDiagonal() const;;

    void multiplicacionMV();

    void inversaDiagonal();

    float *calculaDiagonal();

    float norma(const float *r);

    void diferencia(float *b, float *q, float *r, int n);


    int getN();
    const CSR &getMatriz() const {
        return matriz;
    }

    CSR matriz;
private:
    float *diagonal;
    float *inversa;

};

#endif //JACOBI_H