from item cimport Item
cimport numpy

ctypedef numpy.double_t cDOUBLE
ctypedef numpy.int32_t cINT32

cdef class LossFunction(object):
    cpdef cDOUBLE get_loss(self, Item item, numpy.ndarray[cDOUBLE, ndim=1] weights)
    cpdef int get_decision(self, Item item, numpy.ndarray[cDOUBLE, ndim=1] weights)
    cpdef cDOUBLE get_proba(self, Item item, numpy.ndarray[cDOUBLE, ndim=1] weights)
    cpdef numpy.ndarray get_gradient(self, numpy.ndarray[cDOUBLE, ndim=1] weights, Item item)
