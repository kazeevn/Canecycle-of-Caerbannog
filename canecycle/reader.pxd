from cpython cimport bool
from canecycle.source cimport Source
from canecycle.parser cimport Parser
from canecycle.parser import read_shad_lsml_header
from canecycle.item cimport Item

cpdef Reader from_shad_lsml(str filename, unsigned int hash_size)

cdef class Reader(Source):
    cdef object file
    cdef Parser parser
    cdef str filename
    cdef bool is_ready
    cdef unsigned int holdout_counter
    cdef unsigned int skip
    
    cpdef restart(self, int holdout)
    cpdef unsigned int get_features_count(self)
