from cpython cimport bool
from item cimport Item
cimport numpy as np

cdef class Source(object):
    cdef np.int_t holdout
    cdef object iterator
    cdef bool is_ready
    # Impossible in Cython, but nevertheless needed
    # cdef void restart(self, int holdout)
    # def Item __next__(self)
