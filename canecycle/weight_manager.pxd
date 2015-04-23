
cdef class WeightManager(object):
    cdef float ones
    cdef float zeros
    
    cpdef float get_weight(self, int label, float weight)

