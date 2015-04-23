from canecycle.item cimport Item
from canecycle.hash_function cimport HashFunction

cdef class Parser:
    cdef HashFunction hash_function
    cdef list format
    cpdef Item parse(self, str line)
    cpdef unsigned int get_features_count(self)
