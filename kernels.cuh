#ifndef MV_MULTIPLICATION_CUH_
#define MV_MULTIPLICATION_CUH_

#define FULL_WARP_MASK 0xFFFFFFFF

template<class T>
__device__ T reduction_sum_warp(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    return val;
}

template<class T>
__device__ T reduction_max_warp(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    return val;
}

template<class T>
__global__ void matrix_vector_multiplication(
        const T *val,
        const unsigned int *con_ind,
        const unsigned int *row_ptr,
        const T *x,
        T *y,
        unsigned int n_filas) {

    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = thread_id / 32;
    const unsigned int carril = thread_id % 32;
    const unsigned int fila = warp_id; //<-- un warp por fila
    T sum = 0;

    if (fila < n_filas) {
        const unsigned int row_start = row_ptr[fila];
        const unsigned int row_end = row_ptr[fila + 1];
        for (unsigned int i = row_start + carril; i < row_end; i += 32) {
            sum += val[i] * x[con_ind[i]];
        }
        sum = reduction_sum_warp(sum);

        if (carril == 0) {
            y[fila] = sum;
        }
    }
}

template<class T>
__global__ void nuevaX(int n, T *x, const T *r) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        x[thread_id] = x[thread_id] + r[thread_id];
}

template<class T>
__global__ void nuevaX_srj(int n, T *x, const T *r, const T omega) {
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < n)
        x[thread_id] = x[thread_id] + r[thread_id] * omega;
}

template<class T, unsigned int blockSize>
__global__ void reduction_max(const T *g_idata, T *g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    sdata[tid] = 0;
//    unsigned int gridSize = blockSize * 2 * gridDim.x;
//    if (i < n && n < blockSize) sdata[tid] = abs(g_idata[i]);
//    while (i < n && n > blockSize) {
//        sdata[tid] = max(abs(g_idata[i]), abs(g_idata[i + blockSize]));
//        i += gridSize;
//    }
    if (i + blockSize < n && n > blockSize)
        sdata[tid] = max(abs(g_idata[i]), abs(g_idata[i + blockSize]));
    else if (i < n)
        sdata[tid] = abs(g_idata[i]);
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
        if (blockSize >= 64)
            maximo = max(sdata[tid], sdata[tid + 32]);
        __syncwarp();
        maximo = reduction_max_warp(maximo);
    }
    if (tid == 0) {
        g_odata[blockIdx.x] = maximo;
    }
}

#endif