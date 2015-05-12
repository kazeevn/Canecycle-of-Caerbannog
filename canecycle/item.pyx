

cdef class Item(object):
    """Class to store a piece of data.
    Attributes:
        label(int) - item label, 1 or -1
        weight(float) - item importance for optimizer
        data(array) - item features, supposed to be sparse
        indices(array) - indices of non-zero item features,
            supposed to be sparse
    """
    def __cinit__(self):
        self.label = 0
        self.weight = 1.   
