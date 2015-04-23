#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


from canecycle.item cimport Item
import numpy as np
cimport numpy as np

cdef class LossFunction(object):
    cdef cDOUBLE get_loss(self, Item item, 
                                 np.ndarray[cDOUBLE, ndim=1] weights):
        cdef np.ndarray[cDOUBLE, ndim=1] features = item.features
        return 1. / (1. + np.exp(-np.dot(weights, features)))

    cdef int get_decision(self, Item item, np.ndarray[cDOUBLE, ndim=1] weights):
        cdef cDOUBLE loss = self.get_loss(item, weights)
        return loss > 0.5

    cdef np.ndarray[cDOUBLE, ndim=1] get_gradient(self, 
                                        np.ndarray[cDOUBLE, ndim=1] weights, Item item):
        cdef cDOUBLE sigmoid_values = self.get_loss(item, weights)
        cdef cINT32 label = item.label
        cdef np.ndarray[cDOUBLE, ndim=1] features = item.features
        return (label - self.get_loss(item, weights)) * features


