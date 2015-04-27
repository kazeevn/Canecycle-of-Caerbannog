from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction
from numpy cimport uint64_t

cdef class Parser:
    cdef HashFunction hash_function
    cdef list format
    cdef list column_names
    cdef list numeric_hashes
    cdef uint64_t feature_columns_count
    cpdef Item parse(self, str line)
    cpdef uint64_t get_features_count(self)

