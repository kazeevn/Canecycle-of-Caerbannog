#cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from item import Item


from canecycle.loss_function cimport LossFunction


cdef class Optimizer(object):

    cdef double l1Regularization
    cdef double l2Regularization
    cdef double stepSize
    cdef double scaleDown
    cdef LossFunction loss_function

    def __init__(self, l1Regularization, l2Regularization, stepSize, scaleDown, loss_function):
        self.l1Regularization = l1Regularization
        self.l2Regularization = l2Regularization
        self.stepSize = stepSize
        self.scaleDown = scaleDown
        self.loss_function = loss_function

    cpdef np.ndarray[np.float_t, ndim=1] step(self, item, np.uint64_t step_number, 
             np.ndarray[np.float_t, ndim=1] weights):
        cdef np.uint64_t index, element_index
        cdef double step_size
        cdef np.ndarray[np.uint64_t, ndim=1] col
        cdef np.ndarray[np.float_t, ndim=1]  gradient
        step_size = self.stepSize * self.scaleDown ** step_number
        gradient = self.loss_function.get_gradient(weights, item)
        newStep = gradient * step_size
        col = item.indexes
        for index in range(np.shape(newStep)[0]):
            candidate_weight = weights[col[index]] - newStep[index]
            if candidate_weight > 0:
                candidate_step = candidate_weight - self.l1Regularization * step_size
                weights[col[index]] = max(0.0, candidate_step)
            else:
                candidate_step = candidate_weight + self.l1Regularization * step_size
                weights[col[index]] = min(0.0, candidate_step)
        return weights


