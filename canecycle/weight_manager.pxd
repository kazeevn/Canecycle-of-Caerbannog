
cdef class WeightManager(object):
    cdef float ones
    cdef float zeros
    cdef int sum
    
    cpdef float get_weight(self, int label, float weight)

