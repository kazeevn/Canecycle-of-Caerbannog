from item cimport Item
from numpy cimport ndarray

cdef class LossFunction:
    cdef inline double get_loss(self, Item item, ndarray weights):
        return 0.5
    
    cdef inline int get_decision(self, Item item, ndarray weights):
        return 1
    
    cdef inline object get_gradient(self, object features, int label):
        return features
    
