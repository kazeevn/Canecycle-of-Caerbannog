#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


import numpy as np
cimport numpy as np
from itertools import izip

from canecycle.item cimport Item


cdef class LossFunction(object):
    cpdef np.float_t get_log_proba(self, Item item,
                                   np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t dot_product = np.dot(
            item.data, weights[item.indexes])
        return -np.logaddexp(0., -dot_product)
            

    cpdef np.float_t get_log_one_minus_proba(self, Item item,
                                             np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t dot_product = np.dot(
            item.data, weights[item.indexes])
        return -np.logaddexp(0., dot_product)

    cpdef np.int_t get_decision(self, Item item,
                                np.ndarray[np.float_t, ndim=1] weights):
        return self.get_loss(item, weights) > 0.5

    cpdef np.float_t get_loss(self, Item item,
                              np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t dot_product = np.dot(
            item.data, weights[item.indexes])

        if item.label == 1:
            return np.logaddexp(0., -dot_product)
        else:
            return np.logaddexp(0., dot_product)

    cpdef np.ndarray[np.float_t, ndim=1] get_gradient(self, Item item,
                                                      np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t sigmoid_values = self.get_loss(item, weights)
        return (sigmoid_values - item.label) * item.data
    
    def __reduce__(self):
        return (LossFunction, (), {})
    
    def __setstate__(self, params):
        pass


