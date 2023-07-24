from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Tools App',
    ext_modules=cythonize("tools.py"),
)
