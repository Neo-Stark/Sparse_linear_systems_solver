#include <iostream>
#include <jacobi.h>
#include <readers.h>
#include <cpu_seconds.h>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <SRJ.h>

#define BLOCK_SIZE 128
#define K 1000000
#define THRESHOLD 1.e-6
#define DEBUG 1
using namespace std;

int main(int argc, char **argv) {
    fstream is(argv[1]);
    CSR matriz(is);
    if (matriz.isDiagonallyDominant())
        cout << "La matriz es diagonalmente dominante" << endl;

//    auto _b = rhsVector_reader<double>(argv[2], matriz.getFilas());
    auto _b = vector<double>(matriz.getFilas(), 0);
    int k = 0;
    double t1 = cpuSecond();

    jacobi *test;
    if (argc == 4) {
        auto srjSch = srjSch_reader<double>(argv[3]);
        test = new SRJ(matriz, srjSch, BLOCK_SIZE);
    } else
        test = new jacobi(matriz, BLOCK_SIZE);

    double residual = 1e36;
    while (residual > THRESHOLD) {
        auto y1500 = test->multiplicacionMV_CUDA()[1500];
        test->calculaResiduo(_b.data());
        test->obtenerNuevaX();
        residual = test->norma_CUDA();
        test->actualizaX();
        if (k % 1000 == 0) cout << "k: " << k << " residuo " << residual << " y1500 " << y1500 << endl;
        k++;
    }
    double t2 = cpuSecond() - t1;


    string file(argv[1]);
    std::size_t i = file.rfind('/', file.length());
    cout << "Matriz: " << file.substr(i + 1) << endl;
    cout << "Block size  " << "tiempo\n"
         << BLOCK_SIZE << setw(18) << t2 << endl;
    cout << "ITERS= " << k;

//    if (DEBUG) {
//        cout << endl;
//        cout << "Vector x: ";
//        for (auto x_i : test.getX()) cout << x_i << " ";
//        cout << endl;
//        test.multiplicacionMV_CUDA();
//        cout << "Vector y: ";
//        for (auto i = 0; i < test.getFilas(); i++) cout << test.getY(i) << " ";
//        cout << endl;
//        cout << "Vector b: ";
//        for (auto i = 0; i < matriz.getFilas(); i++) cout << b[i] << " ";
//        cout << endl;
//        cout << "Iteraciones K: " << k << endl;
//    }

    free(test);
//    free(_b);
}
