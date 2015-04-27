cimport numpy

cdef class HashFunction:
    cdef readonly numpy.uint64_t hash_size
    cpdef numpy.uint64_t hash(self, str string)
