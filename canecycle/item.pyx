import ctypes
from scipy.sparse import csr_matrix

cdef class Item:
    cdef int label
    cdef double weight
    
    def __cinit__(self, unsigned int features_number):
        self.label = 0
        self.weight = 1.
        self.features = csr_matrix(features_number, dtype=float)
    