#include <iostream>
#include <jacobi.h>
#include <rhsVector_reader.h>
#include <cpu_seconds.h>
#include <iomanip>
#include <fstream>
#include <omp.h>

#define BLOCK_SIZE 512
#define K 10000
#define THRESHOLD 10e-6
#define DEBUG 0
using namespace std;

int main(int argc, char **argv) {
    fstream is(argv[1]);
    CSR matriz(is);
    cout << "size: " << matriz.getRowPtr().size() << endl;
    if (matriz.isDiagonallyDominant())
        cout << "La matriz es diagonalmente dominante" << endl;

    auto b = rhsVector_reader<double>(argv[2], matriz.getFilas());
    jacobi jacobi(matriz, BLOCK_SIZE);
    int k = 0;
    double t1 = cpuSecond();
    while (k < K) {
        k++;
        if (DEBUG) {
            cout << "V(x) _iter: " << k << "   ";
            for (auto x_i : jacobi.getX()) cout << x_i << " ";
            cout << endl;
        }
        jacobi.multiplicacionMV_CUDA();
        jacobi.calculaResiduo(b);
        jacobi.obtenerNuevaX();
        if (jacobi.normaInfinito_r() <= THRESHOLD) break;
        jacobi.actualizaX();
    }
    double t2 = cpuSecond() - t1;

    string file(argv[1]);
    std::size_t i = file.rfind('/', file.length());
    cout << "Matriz: " << file.substr(i + 1) << endl;
    cout << "Block size  " << "tiempo\n"
         << BLOCK_SIZE << setw(18) << t2 << endl;

    if (DEBUG) {
        cout << endl;
        cout << "Vector x: ";
        for (auto x_i : jacobi.getX()) cout << x_i << " ";
        cout << endl;
        cout << "Vector y: ";
        for (auto i = 0; i < jacobi.getFilas(); i++) cout << jacobi.getY(i) << " ";
        cout << endl;
        cout << "Vector b: ";
        for (auto i = 0; i < matriz.getFilas(); i++) cout << b[i] << " ";
        cout << endl;
        cout << "Iteraciones K: " << k << endl;
    }

    free(b);
}
