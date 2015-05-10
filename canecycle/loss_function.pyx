# cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


from canecycle.item cimport Item
import numpy as np
cimport numpy as np
from itertools import izip


cdef class LossFunction(object):

    cpdef np.float_t get_proba(self, Item item, 
                               np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t feature_value, dot_product
        cdef np.uint64_t index
        dot_product = 0
        for index, feature_value in izip(item.indexes, item.data):
            dot_product += feature_value * weights[index]
        return 1. / (1. + np.exp(-dot_product))



    cpdef np.int_t get_decision(self, Item item, np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t loss = self.get_loss(item, weights)
        return loss > 0.5

    cpdef np.float_t get_loss(self, Item item,
                                   np.ndarray[np.float_t, ndim=1] weights):
        cdef np.int_t label = int(item.label)
        cdef np.float_t prediction = self.get_proba(item, weights)
        cdef np.float_t loss
        loss = -np.log(prediction) if label == 1 else -np.log(1. - prediction)
        return loss

    cpdef np.ndarray[np.float_t, ndim=1] get_gradient(self, Item item,
         np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t sigmoid_values = self.get_loss(item, weights)
        cdef np.float_t label = float(item.label)
        return (sigmoid_values - label) * item.data
    
    def __reduce__(self):
        return (LossFunction, (), {})
    
    def __setstate__(self, params):
        pass

