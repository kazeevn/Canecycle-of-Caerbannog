from item cimport Item

cdef class Source:
    cdef int holdout
    # Impossible in Cython, but nevertheless needed
    # cdef void restart(self, int holdout)
    # def Item __next__(self)
