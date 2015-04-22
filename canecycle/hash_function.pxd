cdef class HashFunction:
    cdef readonly unsigned long hash_size
    cpdef unsigned long hash(self, str string)
