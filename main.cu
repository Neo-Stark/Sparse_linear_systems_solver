#include <iostream>
#include "jacobi.cuh"
#include <rhsVector_reader.h>
#include <cpu_seconds.h>
#include <iomanip>

#define BLOCK_SIZE 512
#define K 1000
#define THRESHOLD 0.00000001
#define DEBUG 0
using namespace std;

int main(int argc, char **argv) {
    std::fstream is(argv[1]);
    CSR matriz = assemble_csr_matrix(is);
    auto b = rhsVector_reader<double>(argv[2], matriz.filas);
    jacobi jacobi(matriz, BLOCK_SIZE);
    int k = 0;
    double t1 = cpuSecond();
    while (k < K) {
        k++;
        jacobi.multiplicacionMV();
        jacobi.diferencia(b, jacobi.getYH(), jacobi.getYH(), jacobi.getFilas());
        if (jacobi.norma(jacobi.getYH()) <= THRESHOLD) break;
        for (size_t i = 0; i < jacobi.getColumnas(); i++)
            jacobi.getXH(i) = jacobi.getXH(i) + jacobi.getYH()[i] * jacobi.getInversa()[i];
    }
    double t2 = cpuSecond() - t1;

    string file(argv[1]);
    std::size_t i = file.rfind('/', file.length());
    cout << "Matriz: " << file.substr(i + 1, file.length() - 1) << endl;
    cout << "Block size  "  << "tiempo\n"
         << BLOCK_SIZE << setw(18) << t2 << endl;

    if (DEBUG) {
        cout << endl;
        cout << "Iteraciones K: " << k << endl;
        cout << "Vector x: ";
        for (auto x_i : jacobi.getXH()) cout << x_i << " ";
        cout << endl;
    }
}
