cimport numpy as np

from canecycle.item cimport Item
from canecycle.loss_function cimport LossFunction

cdef class Optimizer(object):
    cdef np.float_t l1Regularization
    cdef np.float_t l2Regularization
    cdef np.float_t alpha
    cdef np.float_t betta
    cdef np.ndarray z
    cdef np.ndarray n
    cdef LossFunction loss_function
    
    cpdef np.ndarray[np.float_t, ndim=1] step(self,Item item,
                                              np.ndarray[np.float_t, ndim=1] weights)
