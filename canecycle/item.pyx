import ctypes
from scipy.sparse import coo_matrix

cdef class Item(object):
    cpdef public int label
    cpdef public double weight
    cpdef public features
    
    def __cinit__(self, unsigned int features_number):
        self.label = 0
        self.weight = 1.
        self.features = coo_matrix(features_number, dtype=float)
    
