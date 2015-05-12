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
    """FTRL-Proximal algorithm realization.
    Attributes:
        l1_regularisation(float)
        l2_regularisation(float)
        alpha(float)
        beta(float)
        z(array)
        n(aray)
        loss_function(LossFunction) -- function that calculates logarithmic loss
    See http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    for details.
    """
    
    def __cinit__(self, np.float_t l1_regularisation, np.float_t l2_regularisation,
                  np.uint64_t feature_space_size, np.float_t alpha, np.float_t beta,
                  LossFunction loss_function):
        self.l1_regularisation = l1_regularisation
        self.l2_regularisation = l2_regularisation
        self.z = np.zeros(feature_space_size)
        self.n = np.zeros(feature_space_size)
        self.alpha = alpha
        self.beta = beta
        self.loss_function = loss_function

    cpdef np.ndarray[np.float_t, ndim=1] step(self, Item item, 
                                              np.ndarray[np.float_t, ndim=1] weights):
        """Updates model weights to the item given.
        Args:
            item(Item) -- piece of data, see Item description
            weights(array) - model weights, shape(weights)==shapes(item.data)
        """
        cdef np.ndarray l1_survived
        cdef np.ndarray[np.uint64_t, ndim=1] l1_survived_indices
        cdef np.ndarray[np.float_t, ndim=1] gradient
        cdef np.ndarray[np.float_t, ndim=1] sigma
        cdef np.uint32_t n_steps = int(np.ceil(item.weight*0.9))
        cdef np.uint32_t step_index
        for step_index in xrange(n_steps):
            l1_survived = np.abs(self.z[item.indices]) > self.l1_regularisation
            weights[item.indices[-l1_survived]] = 0.0
            l1_survived_indices = item.indices[l1_survived]
            weights[l1_survived_indices] = self.beta + np.sqrt(self.n[l1_survived_indices])
            weights[l1_survived_indices] /= self.alpha
            weights[l1_survived_indices] += self.l2_regularisation
            weights[l1_survived_indices] = -1. / weights[l1_survived_indices]
            weights[l1_survived_indices] *= (self.z[l1_survived_indices] - 
                            np.sign(self.z[l1_survived_indices]) * self.l1_regularisation)
    
            gradient = self.loss_function.get_gradient(item, weights)
            sigma = np.sqrt(self.n[l1_survived_indices] + gradient[l1_survived] ** 2)
            sigma -= np.sqrt(self.n[l1_survived_indices])
            sigma /= self.alpha
            self.z[item.indices] += gradient
            self.z[l1_survived_indices] -= sigma * weights[l1_survived_indices]
            self.n[item.indices] += gradient ** 2
        
        return weights
