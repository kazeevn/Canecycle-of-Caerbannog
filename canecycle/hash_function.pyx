import spooky
cimport numpy as np


cdef np.uint64_t MAX_HASH_SIZE = 2**63


cdef class HashFunction(object):
    def __cinit__(self, np.uint64_t hash_size):
        if hash_size > MAX_HASH_SIZE:
            raise ValueError(
                "We appreciate your hardware, but the maximum "
                "supported hash size is %d." % MAX_HASH_SIZE)
        if hash_size < 0:
            raise ValueError("Hash size should be positive.")
        self.hash_size = hash_size
    
    cpdef np.uint64_t hash(self, str string):
        return spooky.hash64(string) % self.hash_size
