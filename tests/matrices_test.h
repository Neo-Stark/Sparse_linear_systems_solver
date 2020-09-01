//
// Created by Francisco José González García.
// Copyright (c) 2020 Universidad de Granada. All rights reserved.
//

#ifndef SISTEMAS_LINEALES_MATRICES_TEST_H
#define SISTEMAS_LINEALES_MATRICES_TEST_H

#include <sstream>
#include <fstream>

using namespace std;

struct matrices {
    stringstream _1, _2, _3, _4;
    double m1[4][4], m2[4][4], m3[3][3], m4[8][8];
    matrices();
};

struct vectores_b {
    double _1[10];
    double _2[4];
    double _3[4];
    double _3_5[4];
    double _4[8];
    double _4_5[8];

    double _m3[3];

    vectores_b();
};

struct vectores_x {
    double _1[10];
    double _2[4];
    double _3[4];
    double _3_5[4];
    double _4[8];
    double _4_5[8];
    vectores_x();
};

extern matrices m;
extern vectores_x v_x;
extern vectores_b b;
#endif //SISTEMAS_LINEALES_MATRICES_TEST_H
