//
// Created by fran.
//

#ifndef READERS_H
#define READERS_H

#include <istream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

template<typename T>
static T *rhsVector_reader(string file, int row) {
    T *b = new T[row];
    ifstream fin(file);
    if (!fin.is_open()){
        cout << "no se puede leer el fichero " << file;
        abort();
    }
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

template<class T>
static vector<T> srjSch_reader(string file){
    vector<T> srjSch;
    T element;
    ifstream fin(file);
    if (!fin.is_open()){
        cout << "no se puede leer el fichero " << file;
        abort();
    }
    while (!fin.eof()){
        fin >> element;
        srjSch.push_back(element);
    }
    fin.close();
    return srjSch;
}

#endif //READERS_H
