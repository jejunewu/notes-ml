#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <mathfunlib.hpp>


PyObject* py_addition(PyObject * self, PyObject* args) {
    int a, b;
    int ok = PyArg_ParseTuple(args, "ll", &a, &b);
    int result = addition(a, b);
    return PyLong_FromLong(result);
}

static PyMethodDef superfastcode_methods[] = {
    // The first property is the name exposed to Python, fast_tanh, the second is the C++
    // function name that contains the implementation.
    { "fast_addition", (PyCFunction)py_addition, METH_VARARGS, nullptr},

    // Terminate the array with an object containing nulls.
    { nullptr, nullptr, 0, nullptr }
};

static PyModuleDef superfastcode_module = {
    PyModuleDef_HEAD_INIT,
    "superfastcode",                        // Module name to use with Python import statements
    "Provides some functions, but faster",  // Module description
    0,
    superfastcode_methods                   // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_fastaddition() {
    return PyModule_Create(&superfastcode_module);
}