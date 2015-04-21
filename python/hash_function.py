import spooky
import ctypes 

MAX_HASH_SIZE = 63


class HashFunction:
    def __init__(self, hash_size):
        if hash_size > MAX_HASH_SIZE:
            raise ValueError(
                "We appreciate your hardware, but the maximum supported hash size is 61")
        self.hash_size = hash_size
    
    def hash(self, string):
        return spooky.hash64(string) % self.hash_size

