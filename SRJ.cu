//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#include "SRJ.h"
#include <kernels.cuh>

void SRJ::obtenerNuevaX() {
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size{};
    grid_size.x = (getFilas() + block_size.x - 1) / block_size.x;
    cudaMemcpy(r_d, r, getFilas() * sizeof(double), cudaMemcpyHostToDevice);
    nuevaX_srj<double><<<grid_size, block_size>>>(x.size(), x_d, r_d, srjSch[k%srjSch.size()]);
    k++;
}
