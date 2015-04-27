#cython: profile=True
#cython: cdivision=True

cdef class WeightManager(object):
    #(shiryaev) maybe (p,n) format would have been more intuitive here?..
    def __cinit__(self, float apriori_mean=0.5, float apriori_variance=0.25):
        """If you have some idea about label distribution,
        you can apply it here.
        
        """
        # np
        self.ones = apriori_mean # 0.5 by default
        # nq = npq/(1-q) = npq/(1-npq/np)
        self.zeros = apriori_variance / (1 - apriori_variance/apriori_mean) # 0.5 by default
    
    
    cpdef float get_weight(self, int label, float weight):
        #TODO(shiryaev) use weight
        if label == 1:
            self.ones += 1
            return self.zeros / self.ones # * weight
        elif label == -1:
            self.zeros += 1
            return self.ones / self.zeros # * weight
        else:
            raise ValueError("Unsupported label: %d" % label)

