#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False

import numpy as np
cimport numpy as np
from item import Item

from scipy.sparse import coo_matrix

from canecycle.loss_function cimport LossFunction

ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE


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

    cpdef object step(self, item, unsigned int step_number, weights):
        cdef int index, element_index
        cdef double step_size
        cdef np.ndarray[cINT32, ndim=1] col
        cdef np.ndarray[cDOUBLE, ndim=1] data
        cdef object gradient
        step_size = self.stepSize * self.scaleDown ** step_number
        gradient = self.loss_function.get_gradient(weights, item)
        newStep = gradient * step_size
        col = newStep.row
        data = newStep.data
        for index in range(np.shape(data)[0]):
            candidate_weight = weights[col[index]] - newStep.data[index]
            if candidate_weight > 0:
                candidate_step = candidate_weight - self.l1Regularization * gradient.data[index]
                weights[col[index]] = max(0.0, candidate_step)
            else:
                candidate_step = candidate_weight + self.l1Regularization * gradient.data[index]
                weights[col[index]] = min(0.0, candidate_step)
        return weights


