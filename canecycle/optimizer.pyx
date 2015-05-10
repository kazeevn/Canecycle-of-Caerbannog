#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from item import Item


from canecycle.loss_function cimport LossFunction

cdef class Optimizer(object):

    cdef np.float_t l1Regularization
    cdef np.float_t l2Regularization
    cdef np.float_t alpha
    cdef np.float_t betta
    cdef np.ndarray z
    cdef np.ndarray n
    cdef LossFunction loss_function

    def __cinit__(self, l1Regularization, l2Regularization, feature_space_size, 
                  alpha, betta, loss_function):
        self.l1Regularization = l1Regularization
        self.l2Regularization = l2Regularization
        self.z = np.zeros(feature_space_size)
        self.n = np.zeros(feature_space_size)
        self.alpha = alpha
        self.betta = betta
        self.loss_function = loss_function

    cpdef np.ndarray[np.float_t, ndim=1] step(self, item, 
                                              np.ndarray[np.float_t, ndim=1] weights):
        cdef np.ndarray[np.uint64_t, ndim=1] non_zero_indices = item.indexes
        cdef np.ndarray[np.float_t, ndim=1] non_zero_values = item.data
        cdef np.uint64_t index, index_in_vector
        for index, index_in_vector in enumerate(non_zero_indices):
            if abs(self.z[index_in_vector]) < self.l1Regularization:
                weights[index_in_vector] = 0.0
            else:
                weights[index_in_vector] = self.betta + np.sqrt(self.n[index_in_vector])
                weights[index_in_vector] /= self.alpha
                weights[index_in_vector] += self.l2Regularization
                weights[index_in_vector] = -1. / weights[index_in_vector]
                weights[index_in_vector] *= (self.z[index_in_vector] - 
                                 np.sign(self.z[index_in_vector]) * self.l1Regularization)
        cdef np.ndarray[np.float_t, ndim=1] gradient = self.loss_function.get_gradient(item, weights)
        cdef np.ndarray[np.float_t, ndim=1] sigma = np.sqrt(self.n[non_zero_indices] + gradient ** 2)
        sigma -= np.sqrt(self.n[non_zero_indices])
        sigma /= self.alpha
        self.z[non_zero_indices] += gradient
        self.z[non_zero_indices] -= sigma * weights[non_zero_indices]
        self.n[non_zero_indices] += gradient ** 2
        return weights


