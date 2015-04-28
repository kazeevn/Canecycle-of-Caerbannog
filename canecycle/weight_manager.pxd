cimport numpy as np

cdef class WeightManager(object):
    cdef np.float_t ones
    cdef np.float_t zeros
    
    cpdef np.float_t get_weight(self, np.int_t label, np.float_t weight)

