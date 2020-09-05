//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_UTILIDADES_H
#define SISTEMAS_LINEALES_UTILIDADES_H


class utilidades {
public:
    static double reduce_max_OMP(const double *v, int n);

    static double reduce_max_CUDA(const double *d_vi, int n, const int BLOCK_SIZE);

    static double reduce_max_sec(const double *v, int n);

    static double cpuSecond();
};


#endif //SISTEMAS_LINEALES_UTILIDADES_H
