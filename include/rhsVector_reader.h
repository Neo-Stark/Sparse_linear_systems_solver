//
// Created by fran.
//

#ifndef RHSVECTOR_READER_H
#define RHSVECTOR_READER_H

#include <istream>
#include <fstream>
#include <algorithm>

using namespace std;

template<typename T>
static T *rhsVector_reader(string file, int row) {
    T *b = new T[row];
    fstream fin(file);
    int count = 0;
    count = std::count(std::istreambuf_iterator<char>(fin),
                  std::istreambuf_iterator<char>(), '\n');
    fin.seekg(0);

    if (row != count) {
        printf("Dimensions do not match.\nCode Terminated");
        abort();
    }
    for (int i = 0; i < row; i++) {
        fin >> b[i];
    }
    fin.close();
    return b;
}

#endif //RHSVECTOR_READER_H
