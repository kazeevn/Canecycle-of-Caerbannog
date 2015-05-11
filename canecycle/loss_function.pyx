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
        # http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
        if dot_product < 0:
            return dot_product - np.logaddexp(0., dot_product)
        else:
            return -np.logaddexp(0., -dot_product)

    cpdef np.float_t get_log_one_minus_proba(self, Item item,
                                             np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t dot_product = np.dot(
            item.data, weights[item.indexes])
        # http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
        if dot_product > 0:
            return -dot_product - np.logaddexp(0, -dot_product)
        else:
            return -np.logaddexp(0.,dot_product)

    cpdef np.int_t get_decision(self, Item item,
                                np.ndarray[np.float_t, ndim=1] weights):
        return self.get_loss(item, weights) > 0.5

    cpdef np.float_t get_loss(self, Item item,
                              np.ndarray[np.float_t, ndim=1] weights):
        if item.label == 1:
            return -self.get_log_proba(item, weights)
        else:
            return -self.get_log_one_minus_proba(item, weights)

    cpdef np.ndarray[np.float_t, ndim=1] get_gradient(self, Item item,
                                                      np.ndarray[np.float_t, ndim=1] weights):
        cdef np.float_t sigmoid_values = self.get_loss(item, weights)
        cdef np.float_t label = float(item.label)
        return (sigmoid_values - label) * item.data
