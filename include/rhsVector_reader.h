//
// Created by fran.
//

#ifndef RHSVECTOR_READER_H
#define RHSVECTOR_READER_H
#include <istream>
using namespace std;
template <typename type>
static type* rhsVector_reader(string file, int row){
    type *b = new type[row];
    FILE *fp = fopen(file.c_str(), "r");
    int count = 0;
    double n = 0;
    while (fscanf(fp, "%lf", &n) != -1) {
        count++;
    }

    if (row != count) {
        printf("Dimensions do not match.\nCode Terminated");
        abort();
    }
    fseek(fp, 0, SEEK_SET);
    for (int i = 0; i < row; i++) {
        fscanf(fp, "%lf", &b[i]);
    }
    fclose(fp);

    return b;
}
#endif //RHSVECTOR_READER_H
