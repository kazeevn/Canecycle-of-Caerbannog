from numpy cimport ndarray

cdef class Item:
    cdef public int label
    cdef public double weight
    cdef ndarray indexes
    cdef ndarray data
