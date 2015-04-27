from cpython cimport bool
from canecycle.source cimport Source
from canecycle.parser cimport Parser
from canecycle.parser import read_shad_lsml_header
from numpy cimport uint64_t

cpdef Reader from_shad_lsml(str filename, uint64_t hash_size)

cdef class Reader(Source):
    cdef object file
    cdef Parser parser
    cdef str filename
    cdef bool is_ready
    cdef uint64_t holdout_counter
    cdef uint64_t skip
    
    cpdef restart(self, int holdout)
    cpdef uint64_t get_features_count(self)
