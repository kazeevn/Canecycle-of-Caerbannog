from numpy cimport ndarray

cdef class Item:
    cdef public int label
    cdef public double weight
    cdef public ndarray indexes
    cdef public ndarray data
