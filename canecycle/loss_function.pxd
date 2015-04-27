from item cimport Item
cimport numpy

cdef class LossFunction(object):
    cpdef numpy.float_t get_loss(self, Item item, numpy.ndarray[numpy.float_t, ndim=1] weights)
    cpdef numpy.int_t get_decision(self, Item item, numpy.ndarray[numpy.float_t, ndim=1] weights)
    cpdef numpy.float_t get_proba(self, Item item, numpy.ndarray[numpy.float_t, ndim=1] weights) except *
    cpdef numpy.ndarray[numpy.float_t, ndim=1] get_gradient(self, numpy.ndarray[numpy.float_t, ndim=1] weights, Item item)
