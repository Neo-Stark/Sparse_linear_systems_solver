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
__device__ T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
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
        for (unsigned int element = row_start + lane; element < row_end; element += 32) {
            sum += data[element] * x[col_ids[element]];
//            if (row == 8) printf("sum last row: %f \n", sum);
        }
    }
    sum = warp_reduce(sum);

    if (lane == 0 && row < n_rows) {
        y[row] = sum;
    }
}

template<class T>
__global__ void kernelNuevaX(int n, T *x, const T *r, const T *inversa_diag) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        x[thread_id] = x[thread_id] + r[thread_id] * inversa_diag[thread_id];
}

template<class T, unsigned int blockSize>
__global__ void reduce_max(const T *g_idata, T *g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    if (i < n && n < blockSize) sdata[tid] = g_idata[i];
    while (i < n && n > blockSize) {
        sdata[tid] = max(g_idata[i], g_idata[i + blockSize]);
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); }
        __syncthreads();
    }
    T maximo = sdata[tid];
    if (tid < 32) {
        maximo = max(sdata[tid], sdata[tid + 32]);
        __syncwarp();
        maximo = warp_reduce_max(maximo);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = maximo;
    }
}

#endif