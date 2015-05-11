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
        weights[item.indexes] = self.betta + np.sqrt(self.n[item.indexes])
        weights[item.indexes] /= self.alpha
        weights[item.indexes] += self.l2Regularization
        weights[item.indexes] = -1. / weights[item.indexes]
        weights[item.indexes] *= (self.z[item.indexes] - 
                         np.sign(self.z[item.indexes]) * self.l1Regularization)
        weights[item.indexes][np.abs(self.z[item.indexes]) <= self.l1Regularization] = 0.0

        cdef np.ndarray[np.float_t, ndim=1] gradient = \
            self.loss_function.get_gradient(item, weights)
        cdef np.ndarray[np.float_t, ndim=1] sigma = np.sqrt(self.n[item.indexes] + gradient ** 2)
        sigma -= np.sqrt(self.n[item.indexes])
        sigma /= self.alpha
        self.z[item.indexes] += gradient
        self.z[item.indexes] -= sigma * weights[item.indexes]
        self.n[item.indexes] += gradient ** 2
        return weights
