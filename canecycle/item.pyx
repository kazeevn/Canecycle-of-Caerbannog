import scipy.sparse

cdef class Item:
    cdef public int label
    cdef public double weight
    cdef public object features 

    def __cinit__(self, unsigned int features_number):
        self.label = 0
        self.weight = 1.
        self.features = scipy.sparse.dok_matrix((features_number, 1), dtype=float)
    
