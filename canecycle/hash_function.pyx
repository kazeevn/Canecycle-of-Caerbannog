import spooky
cimport numpy as np


cdef np.uint64_t MAX_HASH_SIZE = 2**63


cdef class HashFunction(object):
    """Class to calculate hash of string.
    Attributes:
        hash_size(unsigned long) - hash range is 0..hash_size-1
    """
    
    def __cinit__(self, np.uint64_t hash_size):
        """Stores given hash size."""
        if hash_size > MAX_HASH_SIZE:
            raise ValueError(
                "We appreciate your hardware, but the maximum "
                "supported hash size is %d." % MAX_HASH_SIZE)
        self.hash_size = hash_size
    
    cpdef np.uint64_t hash(self, str string):
        """Applies hash to the string given."""
        return spooky.hash64(string) % self.hash_size
