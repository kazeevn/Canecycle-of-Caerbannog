# cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


from canecycle.item cimport Item
import numpy as np
cimport numpy as np


cdef class LossFunction(object):

    cpdef cDOUBLE get_proba(self, Item item, 
                                 np.ndarray[cDOUBLE, ndim=1] weights):
        cdef np.ndarray[cINT32, ndim=1] features_indices = item.indexes
        cdef np.ndarray[cDOUBLE, ndim=1] features_values = item.data
        cdef cDOUBLE feature_value, dot_product
        cdef int index
        
        dot_product = 0.0
        for index, feature_value in enumerate(features_values):
            dot_product += feature_value * weights[features_indices[index]]
        return 1. / (1. + np.exp(-dot_product))


    cpdef int get_decision(self, Item item, np.ndarray[cDOUBLE, ndim=1] weights):
        cdef cDOUBLE loss = self.get_loss(item, weights)
        return loss > 0.5

    cpdef cDOUBLE get_loss(self, Item item,
                                   np.ndarray[cDOUBLE, ndim=1] weights):
        cdef cINT32 label = int(item.label)
        cdef cDOUBLE prediction = self.get_proba(item, weights)
        cdef cDOUBLE loss = -np.log(prediction) if label == 1 else -np.log(1. - prediction)
        return loss

    cpdef np.ndarray[cDOUBLE, ndim=1] get_gradient(self, np.ndarray[cDOUBLE, ndim=1] weights, Item item):
        cdef cDOUBLE sigmoid_values = self.get_loss(item, weights)
        cdef cDOUBLE label = float(item.label)
        return -(sigmoid_values - label) * item.data



