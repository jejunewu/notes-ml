cmake_minimum_required(VERSION 3.26)
project(cython_)

set(CMAKE_CXX_STANDARD 14)

add_executable(cython_
        demo_cpp_to_py/cpp_add.cpp
        demo_cpp_to_py/main.cpp
        demo_cpp_to_py/cpp_add.h
        demo_cpp_to_py/cpp_add.pyx
        demo2_c_to_py/main.c
        demo2_c_to_py/main.cpp
        demo2_c_to_py/tools.cpp
        demo2_c_to_py/tools.h)
