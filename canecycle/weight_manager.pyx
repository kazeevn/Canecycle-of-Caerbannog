#cython: profile=True
#cython: cdivision=True

cimport numpy as np

cdef class WeightManager(object):
    #(shiryaev) maybe (p,n) format would have been more intuitive here?..
    def __cinit__(self, np.float_t apriori_mean=0.5, np.float_t apriori_variance=0.25):
        """If you have some idea about label distribution,
        you can apply it here.
        
        """
        # np
        self.ones = apriori_mean # 0.5 by default
        # nq = npq/(1-q) = npq/(1-npq/np)
        self.zeros = apriori_variance / (1 - apriori_variance/apriori_mean) # 0.5 by default
    
    
    cpdef np.float_t get_weight(self, np.int_t label, np.float_t weight):
        #TODO(shiryaev) use weight
        if label == 1:
            self.ones += 1
            return self.zeros / self.ones # * weight
        elif label == -1:
            self.zeros += 1
            return self.ones / self.zeros # * weight
        else:
            raise ValueError("Unsupported label: %d" % label)

