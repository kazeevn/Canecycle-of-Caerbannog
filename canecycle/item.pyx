#import scipy.sparse

cdef class Item:
    def __cinit__(self):
        self.label = 0
        self.weight = 1.
#        self.features = scipy.sparse.coo_matrix(
#            (features_number, 1), dtype=float)    
