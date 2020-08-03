#include <jacobi.h>
#include <kernels.cuh>
#include <iostream>
#include <sstream>
#include <fstream>


using namespace std;
#define BLOCKSIZE 512

struct matrices {
    stringstream matriz1;
    stringstream matriz2;

    matrices() :
            matriz1("%%matrix market matrix\n"
                    "10 10 12\n"
                    "1 1 1\n"
                    "1 2 1\n"
                    "2 2 1\n"
                    "3 3 1\n"
                    "4 4 1\n"
                    "5 5 1\n"
                    "6 6 1\n"
                    "7 7 1\n"
                    "8 8 1\n"
                    "9 9 1\n"
                    "10 1 1\n"
                    "10 10 1\n"),
            matriz2("%%matrix market matrix\n"
                    "4 4 13\n"
                    "1 1 4\n"
                    "1 2 -1\n"
                    "1 3 -6\n"
                    "2 1 -5\n"
                    "2 2 -4\n"
                    "2 3 10\n"
                    "2 4 8\n"
                    "3 2 9\n"
                    "3 3 4\n"
                    "3 4 -2\n"
                    "4 1 1\n"
                    "4 3 -7\n"
                    "4 4 5\n") {}
} matrices;

struct vectores_b {
    double _1[10] = {3, 2, 3, 4, 5, 6, 7, 8, 9, 11};
    double _2[4] = {2, 21, -12, -6};
} b;

struct vectores_x {
    double _1[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double _2[4] = {3, -2, 2, 1};
} v_x;

void test_multiplicacion_cuda() {

    CSR matriz(matrices.matriz2);
    double *x, *x_d, *y, *y_d, *A;
    unsigned int *col_ind, *row_ptr;
    x = v_x._2;
    y = new double[matriz.getFilas()];
    cudaMalloc(&A, sizeof(double) * matriz.getVal().size());
    cudaMalloc(&col_ind, sizeof(int) * matriz.getColInd().size());
    cudaMalloc(&row_ptr, sizeof(int) * matriz.getRowPtr().size());
    cudaMalloc(&x_d, sizeof(double) * matriz.getColumnas());
    cudaMalloc(&y_d, sizeof(double) * matriz.getFilas());
    cudaMemcpy(A, matriz.getVal().data(), matriz.getVal().size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.getColInd().data(), matriz.getColInd().size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.getRowPtr().data(), matriz.getRowPtr().size() * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Row ptr: ";
    for (auto row : matriz.getRowPtr()) cout << row << " ";
    cout << " size: " << matriz.getRowPtr().size();
    cout << endl;

    dim3 block_size(BLOCKSIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * matriz.getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(x_d, x, matriz.getColumnas() * sizeof(double), cudaMemcpyHostToDevice);

    csr_spmv_vector_kernel<double><<<grid_size, block_size>>>(matriz.getFilas(), col_ind, row_ptr, A, x_d, y_d);

    cudaMemcpy(y, y_d, matriz.getFilas() * sizeof(double), cudaMemcpyDeviceToHost);
    matriz.printMatrix();
    cout << "Solución multiplicación: " << endl;
    for (auto i = 0; i < matriz.getFilas(); i++) cout << y[i] << " ";
    cout << endl;
}

void test_reduce_max() {
    // test_reduce_max
    int n = 10;
    auto *h_v = new double[n];
    cout << "n = " << n << endl;
    for (int i = 0; i < n; i++) {
        h_v[i] = i;
    }
    double *d_vi, *d_vo;
    cudaMalloc(&d_vi, sizeof(double) * n);
    cudaMemcpy(d_vi, h_v, sizeof(double) * n, cudaMemcpyHostToDevice);
    dim3 block(BLOCKSIZE);
    dim3 grid = (n / 2 + block.x - 1) / block.x;
    cudaMalloc(&d_vo, sizeof(double) * grid.x);
    reduce_max<double, 512><<<grid, block, block.x*sizeof(double)>>>(d_vi, d_vo, n);
    double h_vo[grid.x];
    cudaMemcpy(h_vo, d_vo, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double maximo = 0.0;
    for (int i = 0; i < grid.x; i++) maximo = max(maximo, h_vo[i]);

    cout.precision(20);
    cout << "Máximo: " << maximo << endl;
    cudaFree(d_vi);
    cudaFree(d_vo);
    free(h_v);
}

void test_diagonal() {
    CSR matriz(matrices.matriz1);
    jacobi test(matriz, BLOCKSIZE);
    cout << "Diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getDiagonal()[i] << " ";
    cout << endl;
    cout << "Inversa diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getInversa(i) << " ";
    cout << endl;
    cout << "Inversa diagonal x Diagonal: ";
    for (int i = 0; i < matriz.getFilas(); i++) cout << test.getInversa(i) * test.getDiagonal()[i] << " ";
    cout << endl;
}

void test_jacobi() {
    CSR matriz(matrices.matriz2);
    matriz.printMatrix();
    cout << matriz.isDiagonallyDominant() << endl;
    cout << "vector b: ";
    for (double i : b._2)
        cout << i << " ";
    cout << endl;

    jacobi test(matriz, 64);
    int k = 0;
    while (k < 39) {
        k++;
        test.multiplicacionMV_CUDA();
        test.calculaResiduo(b._2);
        test.obtenerNuevaX();
        test.actualizaX();
        if (test.norma() <= 1e-5) break;
        for (int i = 0; i < test.getFilas(); i++)
            test.getX(i) = test.getR(i) * test.getInversa(i);
    }

    cout << endl;
    cout << "Vector x: ";
    for (auto x_i : test.getX()) cout << x_i << " ";
    cout << endl;
    cout << "Vector y: ";
    for (auto i = 0; i < test.getFilas(); i++) cout << test.getY(i) << " ";
    cout << endl;
    cout << "Vector b: ";
    for (auto i = 0; i < matriz.getFilas(); i++) cout << b._2[i] << " ";
    cout << endl;
    cout << "Iteraciones K: " << k << endl;
}

void ordenar_mtx(const string &nombre) {
    ofstream of(nombre + "_copia", ios::trunc);
    ifstream fin(nombre);
    string banner;
    while (fin.peek() == '%') {
        getline(fin, banner);
        of << banner << endl;
    }
    int M, N, L;
    fin >> M >> N >> L;
    of << M << " " << N << " " << L << endl;
    for (int l = 0; l < L; l++) {
        int row, col;
        double data;
        fin >> row >> col >> data;
        of << col << " " << row << " " << data << endl;
    }
    of.close();
    fin.close();
}

void test_multiplacion_omp() {
    CSR matriz(matrices.matriz2);
    jacobi test(matriz, {3, -2, 2, 1});

    auto y = test.multiplicacionMV_OMP();
    cout << "y(): ";
    for (int i = 0; i < test.getFilas(); i++) cout << y[i] << " ";
    cout << endl;
}

int main(int argc, char **argv) {
//    test_jacobi();
//    test_diagonal();
//    test_multiplicacion_cuda();
    test_multiplacion_omp();
//    test_reduce_max();
//    ordenar_mtx(argv[1]);
}