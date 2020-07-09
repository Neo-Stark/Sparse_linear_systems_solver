#ifndef MV_MULTIPLICATION_CUH_
#define MV_MULTIPLICATION_CUH_

#define FULL_WARP_MASK 0xFFFFFFFF

template<class T>
__device__ T warp_reduce(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    return val;
}

template<class T>
__global__ void csr_spmv_vector_kernel(
        unsigned int n_rows,
        const unsigned int *col_ids,
        const unsigned int *row_ptr,
        const T *data,
        const T *x,
        T *y) {

    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = thread_id / 32;
    const unsigned int lane = thread_id % 32;
    const unsigned int row = warp_id; ///< un warp por fila
    T sum = 0;

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
}

template<class T>
__global__ void kernelNuevaX(int n, T *x, const T *y, const T *inversa_diag) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        x[thread_id] = x[thread_id] + y[thread_id] * inversa_diag[thread_id];
}
#endif