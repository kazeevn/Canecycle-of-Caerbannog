from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("canecycle/*.pyx"),
    include_dirs=[numpy.get_include()],
    name='canecycle',
    url='https://github.com/kazeevn/Canecycle-of-Caerbannog',
    license='GPLv3',
    packages=['canecycle']
)
