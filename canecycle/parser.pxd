from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction

cdef class Parser:
    cdef HashFunction hash_function
    cdef list format
    cdef list column_names
    cdef list numeric_hashes
    cdef unsigned int features_count
    cpdef Item parse(self, str line)
    cpdef unsigned int get_features_count(self)

