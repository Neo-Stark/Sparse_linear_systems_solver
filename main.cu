#include <iostream>
#include <fstream>
#include "jacobi.cuh"
#include <reduce.cuh>
#include <rhsVector_reader.h>
#include <sstream>

#define BLOCK_SIZE 32
using namespace std;

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
    istringstream iss(mm_file_content);

    CSR matriz = assemble_csr_matrix(iss);
    jacobi jacobi(matriz);
    vector<double> x_h = {1, 1, 1, 1};

    double *A, *x_d, *y_d, *y_h;
    unsigned int *col_ind, *row_ptr;

    y_h = new double[matriz.filas];
    cudaMalloc(&A, sizeof(double) * matriz.val.size());
    cudaMalloc(&col_ind, sizeof(double) * matriz.col_ind.size());
    cudaMalloc(&row_ptr, sizeof(double) * matriz.row_ptr.size());
    cudaMalloc(&x_d, sizeof(double) * x_h.size());
    cudaMalloc(&y_d, sizeof(double) * x_h.size());
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    const unsigned int warp_size = 32; /// One warp per row
    grid_size.x = (warp_size * jacobi.getFilas() + block_size.x - 1) / block_size.x;

    cudaMemcpy(A, matriz.val.data(), matriz.val.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, matriz.col_ind.data(), matriz.col_ind.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr, matriz.row_ptr.data(), matriz.row_ptr.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x_h.data(), x_h.size() * sizeof(double), cudaMemcpyHostToDevice);

    csr_spmv_vector_kernel<double><<<grid_size, block_size>>>(jacobi.getFilas(), col_ind, row_ptr, A, x_d, y_d);

    cudaMemcpy(y_h, y_d, matriz.filas * sizeof(double), cudaMemcpyDeviceToHost);

    printMatrix(matriz);
    cout << endl;
    cout << "Vector x: ";
    for (auto x_i : x_h) cout << x_i << " ";
    cout << endl;

    cout << "Resultado multiplicación:\n";
    cout << "y = [";
    for (int i = 0; i < matriz.filas - 1; i++) cout << y_h[i] << ", ";
    cout << y_h[matriz.filas - 1] << "]" << endl;
}
