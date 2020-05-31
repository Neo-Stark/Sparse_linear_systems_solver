#include <iostream>
#include <fstream>
#include "jacobi.h"
#include <sstream>

using namespace std;

int main() {
    std::fstream is("../data/matrix4225.mtx");
    const string mm_file_content =
            "%%MatrixMarket matrix coordinate real symmetric\n"
            "%----------------------------------------------\n"
            "% Multi line comment \n"
            "4 4 4\n"
            "2 1 5\n"
            "2 2 8\n"
            "3 3 3\n"
            "4 2 6\n";
    istringstream iss (mm_file_content);

    CSR matriz = assemble_csr_matrix(iss);
    jacobi jacobi(matriz);

    cout << "********MATRIZ*******" << endl;
    printMatrix(matriz);
    cout << "N: " << jacobi.matriz.N << endl;
    cout << "getN(): " << jacobi.getN() << endl;
    float *diagonal = jacobi.getDiagonal();
    cout << "Diagonal: [ ";
    for (int i=0; i < jacobi.getN(); i++){
        cout << diagonal[i] << " ";
    }
    float *inversa = jacobi.getInversa();
    cout << "]\n";
cout << "Inversa: [ ";
    for (int i=0; i < jacobi.getN(); i++){
        cout << inversa[i] << " ";
    }
    cout << "]\n";
    std::vector<double>::iterator it;

    it = find(matriz.val.begin(), matriz.val.end(), -0.000669);
    if (it != matriz.val.end())
        std::cout << "Element found in matriz: " << *it << '\n';
    else
        std::cout << "Element not found in matriz\n";
    cout << matriz.N;
}
