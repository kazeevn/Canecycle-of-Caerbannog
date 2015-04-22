from __future__ import division
import ctypes


cdef class WeightManager(object):
    cdef unsigned int ones
    cdef unsigned int zeros
    
    def __cinit__(self):
        # a priori values
        self.ones = 1
        self.zeros = 1
    
    
    cdef float get_weight(self, int label, float weight):
        #TODO: use weight
        if label == 1:
            self.ones += 1
            return 1. # self.ones / self.zeros # * weight
        elif label == -1:
            self.zeros += 1
            return 1. # self.zeros / self.ones # * weight
        else:
            raise ValueError("Unsupported label: %d" % label)
        
        return 0.

