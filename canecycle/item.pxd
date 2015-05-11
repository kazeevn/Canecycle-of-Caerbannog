cimport numpy as np


cdef class Item(object):
    cdef public np.int_t label
    cdef public np.float_t weight
    cdef public np.ndarray indexes
    cdef public np.ndarray data
