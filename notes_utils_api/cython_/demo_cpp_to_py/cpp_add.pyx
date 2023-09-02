# cpp_add.pyx
cdef extern from "cpp_add.cpp":
    int add2(int x, int y)

def py_add(int x, int y):
    return add2(x, y)
