#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False


import numpy as np
cimport numpy as np

from canecycle.item import Item
from canecycle.item cimport Item
from canecycle.loss_function cimport LossFunction


cdef class Optimizer(object):
    def __cinit__(self, np.float_t l1Regularization, np.float_t l2Regularization,
                  np.uint64_t feature_space_size, np.float_t alpha, np.float_t betta,
                  LossFunction loss_function):
        self.l1Regularization = l1Regularization
        self.l2Regularization = l2Regularization
        self.z = np.zeros(feature_space_size)
        self.n = np.zeros(feature_space_size)
        self.alpha = alpha
        self.betta = betta
        self.loss_function = loss_function

    cpdef np.ndarray[np.float_t, ndim=1] step(self, Item item, 
                                              np.ndarray[np.float_t, ndim=1] weights):
        cdef np.ndarray l1_survived = np.abs(self.z[item.indexes]) > self.l1Regularization
        weights[item.indexes[-l1_survived]] = 0.0
        cdef np.ndarray[np.uint64_t, ndim=1] l1_survived_indexes = item.indexes[l1_survived]
        cdef np.int_t n_steps = int(np.ceil(item.weight-1e-3))
        cdef np.ndarray[np.float_t, ndim=1] gradient
        cdef np.ndarray[np.float_t, ndim=1] sigma
        cdef np.uint32_t step_index
        for step_index in range(n_steps):
            weights[l1_survived_indexes] = self.betta + np.sqrt(self.n[l1_survived_indexes])
            weights[l1_survived_indexes] /= self.alpha
            weights[l1_survived_indexes] += self.l2Regularization
            weights[l1_survived_indexes] = -1. / weights[l1_survived_indexes]
            weights[l1_survived_indexes] *= (self.z[l1_survived_indexes] - 
                            np.sign(self.z[l1_survived_indexes]) * self.l1Regularization)
    
            gradient = self.loss_function.get_gradient(item, weights)
            sigma = np.sqrt(self.n[l1_survived_indexes] + gradient[l1_survived] ** 2)
            sigma -= np.sqrt(self.n[l1_survived_indexes])
            sigma /= self.alpha
            self.z[item.indexes] += gradient
            self.z[l1_survived_indexes] -= sigma * weights[l1_survived_indexes]
            self.n[item.indexes] += gradient ** 2
        
        return weights
