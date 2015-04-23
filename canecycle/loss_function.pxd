from item cimport Item
cimport numpy

ctypedef numpy.double_t cDOUBLE
ctypedef numpy.int32_t cINT32

cdef class LossFunction(object):
    cdef cDOUBLE get_loss(self, Item item, numpy.ndarray[cDOUBLE, ndim=1] weights)
    cdef int get_decision(self, Item item, numpy.ndarray[cDOUBLE, ndim=1] weights)
    cdef object get_gradient(self, numpy.ndarray[cDOUBLE, ndim=1] weights, Item item)
