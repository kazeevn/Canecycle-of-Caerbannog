import spooky
import ctypes 

MAX_HASH_SIZE = 63

cdef class HashFunction:
    cdef unsigned long hash_size
    def __init__(self, hash_size):
        if hash_size > MAX_HASH_SIZE:
            raise ValueError(
                "We appreciate your hardware, but the maximum supported hash size is %d" %
		MAX_HASH_SIZE)
        self.hash_size = 2**hash_size
    
    cpdef unsigned long hash(self, string):
        return spooky.hash64(string) % self.hash_size

