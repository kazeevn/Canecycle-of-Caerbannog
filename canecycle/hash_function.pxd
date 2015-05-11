cimport numpy as np


cdef class HashFunction(object):
    cdef readonly np.uint64_t hash_size
    cpdef np.uint64_t hash(self, str string)
