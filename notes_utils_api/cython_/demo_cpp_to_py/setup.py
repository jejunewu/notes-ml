from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("cpp_add", sources=["cpp_add.pyx", "cpp_add.cpp"])
setup(name="cpp_add", ext_modules=cythonize([ext]))