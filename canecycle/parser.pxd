import numpy as np
cimport numpy as np

from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction


cdef class Parser(object):
    cdef HashFunction hash_function
    cdef list format
    cdef list column_names
    cdef list numeric_hashes
    cdef public np.uint64_t feature_columns_count
    cpdef Item parse(self, str line)
    cpdef np.uint64_t get_features_count(self)
