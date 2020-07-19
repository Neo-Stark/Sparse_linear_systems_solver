#include <iostream>
#include <jacobi.h>
#include <rhsVector_reader.h>
#include <cpu_seconds.h>
#include <iomanip>
#include <fstream>

#define BLOCK_SIZE 512
#define K 1000
#define THRESHOLD 0.00000001
#define DEBUG 0
using namespace std;

int main(int argc, char **argv) {
    std::fstream is(argv[1]);
    CSR matriz(is);
    auto b = rhsVector_reader<double>(argv[2], matriz.getFilas());
    jacobi jacobi(matriz, BLOCK_SIZE);
    int k = 0;
    double t1 = cpuSecond();
    while (k < K) {
        k++;
        if ( jacobi.multiplicacionMV_CUDA()[98] - jacobi.multiplicacionMV_OMP()[98] < THRESHOLD)
            cout << "k = " << k << " multiplicacion igual" << endl;
        jacobi::diferencia(b, jacobi.getY(), jacobi.getY(), jacobi.getFilas());
        if (jacobi.norma() <= THRESHOLD) break;
        jacobi.obtenerNuevaX();
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
        for (auto x_i : jacobi.getX()) cout << x_i << " ";
        cout << endl;
    }
}
