import spooky
import ctypes 

class HashFunction:
    def __init__(hash_size):
        if hash_size > 63:
            raise ValueError(
                "We appreciate your hardware, but the maximum supported hash size is 61")
        self.hash_size = hash_size
    
    def hash(string):
        return spooky.hash64(string) % self.hash_size

