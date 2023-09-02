//
// Created by root on 8/12/23.
//
#include <iostream>
#include "tools.h"

using namespace std;

extern "C" {
int func(int x, int y) {
    cout << "x_func " << x << endl;
    cout << "y_func " << y << endl;
    return multi(x, y) + x + y;
}
}