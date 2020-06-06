//
// Created by fran on 1/6/20.
//

#ifndef SISTEMAS_LINEALES_CPU_SECONDS_H
#define SISTEMAS_LINEALES_CPU_SECONDS_H

#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

#endif //SISTEMAS_LINEALES_CPU_SECONDS_H
