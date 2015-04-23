from cpython cimport bool
from canecycle.source cimport Source
from canecycle.parser cimport Parser
from canecycle.item cimport Item


cdef class Reader(Source):
    cdef object file
    cdef Parser parser
    cdef str filename
    cdef bool is_ready
    cdef unsigned int holdout_counter
    cdef unsigned int skip
    
    cpdef restart(self, int holdout)
