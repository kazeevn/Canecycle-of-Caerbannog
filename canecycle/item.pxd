from numpy cimport ndarray, int_t, float_t

cdef class Item:
    cdef public int_t label
    cdef public float_t weight
    cdef public ndarray indexes
    cdef public ndarray data
