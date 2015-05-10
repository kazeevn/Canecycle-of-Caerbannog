from cpython cimport bool
from item cimport Item

cdef class Source:
    cdef int holdout
    cdef object iterator
    cdef bool is_ready
    # Impossible in Cython, but nevertheless needed
    # cdef void restart(self, int holdout)
    # def Item __next__(self)

            
            
