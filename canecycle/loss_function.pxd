cimport numpy as np

from canecycle.item cimport Item


cdef class LossFunction(object):
    cpdef np.float_t get_log_proba(self, Item item, np.ndarray[np.float_t, ndim=1] weights)
    cpdef np.float_t get_log_one_minus_proba(self, Item item
                                             np.ndarray[np.float_t, ndim=1] weights)
    cpdef np.float_t get_loss(self, Item item, np.ndarray[np.float_t, ndim=1] weights)
    cpdef np.int_t get_decision(self, Item item, np.ndarray[np.float_t, ndim=1] weights)
    cpdef np.ndarray[np.float_t, ndim=1] get_gradient(self, Item item,
                                                      np.ndarray[np.float_t, ndim=1])
