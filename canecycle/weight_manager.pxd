
cdef class WeightManager(object):
    cdef float ones
    cdef float zeros
    
    cdef float get_weight(self, int label, float weight)

