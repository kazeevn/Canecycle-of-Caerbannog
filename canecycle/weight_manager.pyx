#cython: profile=True
#cython: cdivision=True


cimport numpy as np


cdef class WeightManager(object):
    """Class for storing information about class sizes
    in binary classification.
    Attributes:
        ones(float) - number of items with label 1
        zeros(float) - number of items with label -1
    """

    def __cinit__(self, np.float_t apriori_mean=0.5, np.float_t apriori_variance=0.25):
        """If you have some idea about label distribution,
        you can apply it here.
        Args:
            apriori_mean(float): expectation of quantity of ones
            apriori_variance(float): variance of quantity of ones
        """
        # np
        self.ones = apriori_mean
        # nq = npq/(1-q) = npq/(1-npq/np)
        self.zeros = apriori_variance / (1 - apriori_variance/apriori_mean)
    
    
    cpdef np.float_t get_weight(self, np.int_t label, np.float_t weight):
        """Updates itself with new label and returns recommended weight.
        Args:
            label(int): item label, 1 or -1
            weight(float): item weight (not used)
        """
        if label == 1:
            self.ones += 1
            return self.zeros / self.ones
        elif label == -1:
            self.zeros += 1
            return self.ones / self.zeros
        else:
            raise ValueError("Unsupported label: %d" % label)
